#!/usr/bin/env python3
"""
R1.2 — Panel-augmented fine-tuning (GPU server).

The reviewer's remaining requested ablation: "fine-tuned SLMs receiving
equivalent panel information." The original and tuned fine-tuning runs
trained the SLMs on the sentence alone, while the zero-shot layer receives
the sentence plus the full advisory-panel signal block. This script closes
that asymmetry: it fine-tunes both SLMs on prompts that contain EXACTLY the
zero-shot advisory-panel prompt (sentence + predictions + probabilities +
confidence + entropy + driving words + panel status), with the gold label
as the training target.

Design decisions (all mirror the paper's established protocols):

  1. LEAKAGE-FREE SIGNALS: the panel signals for the 1,584 training rows are
     generated OUT-OF-FOLD (5-fold stratified CV on the train+validation
     pool, vectorizer fit per fold on the training partition only --- the
     identical protocol to the stacking baseline). The validation and test
     rows use the FINAL panel signals (models fit on the full 1,924-row
     pool --- the exact signals stored in test.csv that the zero-shot layer
     receives). The XGBoost feature list is the panel's GLOBAL ranking
     (identical across sentences, by design of the framework), so it is
     reused from the stored artifacts for every row.

  2. NO RETUNING: the configuration already selected by the validation grid
     is used verbatim (LoRA rank 16, 480 steps for both models; see
     Classic/results/ft_tuning/tuning_<model>_selected.json). The point of
     this experiment is an input-matched comparison, not a second search.

  3. LABEL-ONLY TRAINING TARGET: identical to the tuned-FT protocol
     ('Label: <label>' + EOS; label-only outputs accepted by the lenient
     parser), so the comparison against the sentence-only tuned runs is
     apples-to-apples on label accuracy.

Outputs (server, from project root):
  Models/panel_aug/<model>/            LoRA adapters
  Classic/results/ft_panel_aug/panel_aug_<model>.json       metrics record
  Classic/results/ft_panel_aug/val_<model>.csv              validation predictions
  Datasets/financial_phrasebank/test.csv  new columns:
      <model>_ft_panaug_label / _explanation / _time_sec
  (original and tuned *_ft_* columns untouched)

Usage (server, from project root):
    source ./activate_project.csh
    python finetune_panel_aug.py --model qwen           # full run
    python finetune_panel_aug.py --model gemma
    python finetune_panel_aug.py --model qwen --smoke   # 10 steps, 20 val+20 test rows

Pipeline:
    phase 1  generate leakage-free panel signals for train rows (CPU, ~1 min)
    phase 2  train LoRA on panel-augmented prompts (GPU, ~2.5 h/model)
    phase 3  evaluate on validation (340 rows, sanity check only — NOT selection)
    phase 4  evaluate once on test (340 rows) -> new test.csv columns + metrics
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_CSV = Path("Datasets/financial_phrasebank/train.csv")
VAL_CSV = Path("Datasets/financial_phrasebank/validation.csv")
TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
ARTIFACTS = Path("Classic/artifacts")
TUNING_DIR = Path("Classic/results/ft_tuning")
OUT_DIR = Path("Classic/results/ft_panel_aug")
SIGNALS_CACHE = Path("Classic/results/ft_panel_aug/oof_train_signals.csv")
ADAPTER_ROOT = Path("Models/panel_aug")
ALLOWED_LABELS = {"positive", "negative", "neutral"}
CLASSES = ("negative", "neutral", "positive")
SEED = 42
TOP_K_FEATURES = 5

QWEN_MODEL = "unsloth/Qwen3.5-4B"
GEMMA_MODEL = "unsloth/gemma-3-4b-it"


# ═════════════════════════════════════════════════════════════════════════════
# Config — identical training hyperparameters to finetune_tuned.py
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BaseCfg:
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    logging_steps: int = 1
    random_state: int = 42
    generation_retry_attempts: int = 4
    max_new_tokens: int = 96
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 64  # only used by gemma


def selected_config(model_key: str) -> tuple[int, int]:
    """Load the grid-selected (lora_r, max_steps) from the tuned-FT records."""
    path = TUNING_DIR / f"tuning_{model_key}_selected.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run finetune_tuned.py first (or hardcode "
            "LORA_R / MAX_STEPS if you deliberately want another config)")
    sel = json.loads(path.read_text())["selected_config"]
    return int(sel["lora_r"]), int(sel["max_steps"])


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — leakage-free panel signals (mirrors stacking_baseline.py + the
# per-row signal construction of predict_classic_models.py)
# ═════════════════════════════════════════════════════════════════════════════

def entropy_of_probs(prob_row) -> float:
    p = np.clip(np.asarray(prob_row, dtype=float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def top_tokens_from_linear_row(x_row, coef_row, feat_names, top_k) -> str:
    contrib = x_row.multiply(coef_row)
    arr = np.asarray(contrib.toarray()).ravel()
    nz = np.flatnonzero(arr)
    if nz.size == 0:
        return ""
    best = nz[np.argsort(arr[nz])[-top_k:]][::-1]
    return " | ".join(f"{feat_names[i]}:{arr[i]:.4f}" for i in best)


def make_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                           min_df=2, max_df=0.98, sublinear_tf=True)


def oof_signals(smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold panel signals for the TRAIN rows (1,584) and the
    VALIDATION rows (340).

    5-fold stratified CV over the train+validation pool (identical to the
    stacking baseline): for each fold, all three models + the TF-IDF
    vectorizer are fit on the fold's training partition, and the signals
    are computed for the held-out rows. A train (or validation) row
    therefore never sees a signal from a panel model that was trained on
    that row --- the validation sanity check is leakage-free in every
    respect, matching the stacking baseline's construction. The TEST rows
    keep the stored final-pool signals from test.csv (the exact deployment
    condition the zero-shot layer received).

    Returns (train_signals, val_signals).
    """
    if SIGNALS_CACHE.exists():
        cached = pd.read_csv(SIGNALS_CACHE)
        n_train = (cached["split"] == "train").sum()
        print(f"[signals] cached: {SIGNALS_CACHE} "
              f"({n_train} train + {len(cached) - n_train} val rows)")
        return (cached[cached["split"] == "train"].drop(columns="split"),
                cached[cached["split"] == "val"].drop(columns="split"))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC
    from sklearn.utils.extmath import softmax as sklearn_softmax
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    n_train = len(train_df)
    pool = pd.concat([train_df[["sentence", "label"]],
                      val_df[["sentence", "label"]]], ignore_index=True)
    pool["sentence"] = pool["sentence"].astype(str).str.strip()
    pool["label"] = pool["label"].astype(str).str.strip().str.lower()
    y_enc = np.array([CLASSES.index(t) for t in pool["label"]])

    # XGBoost global feature list — identical across sentences BY DESIGN
    # (the framework supplies the panel's global ranking in every prompt);
    # taken from the stored artifacts so it matches the test-row prompts.
    xgb_bundle = __import__("joblib").load(ARTIFACTS / "xgboost_tfidf.joblib")
    global_feats = " | ".join(
        f"{d['feature']}:{d['importance']:.4f}"
        for d in xgb_bundle["top_features"][:TOP_K_FEATURES])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sig = {f"{pfx}_{k}": np.zeros(len(pool)) for pfx in ("logreg", "svm", "xgb")
           for k in ("pred_enc", "confidence", "entropy",
                     "prob_negative", "prob_neutral", "prob_positive")}
    lr_feats = [""] * len(pool)
    sv_feats = [""] * len(pool)
    texts = pool["sentence"].tolist()

    for fold, (tr, te) in enumerate(skf.split(texts, y_enc)):
        vec = make_vectorizer()
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        feat_names = np.array(vec.get_feature_names_out())
        ytr = y_enc[tr]

        lr = LogisticRegression(C=4.0, class_weight="balanced",
                                max_iter=4000, random_state=SEED).fit(Xtr, ytr)
        svm = LinearSVC(C=1.0, class_weight="balanced",
                        random_state=SEED).fit(Xtr, ytr)
        xgb = XGBClassifier(objective="multi:softprob", n_estimators=350,
                            max_depth=6, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, n_jobs=-1, random_state=SEED,
                            eval_metric="mlogloss", verbosity=0).fit(Xtr, ytr)

        lr_p = lr.predict_proba(Xte)
        sv_p = sklearn_softmax(svm.decision_function(Xte))
        xg_p = xgb.predict_proba(Xte)
        # all three models are fit on integer-encoded labels (CLASSES index),
        # so classes_ holds the code for CLASSES[i]
        lr_cls = [CLASSES[int(c)] for c in lr.classes_]
        sv_cls = [CLASSES[int(c)] for c in svm.classes_]
        xg_cls = [CLASSES[int(c)] for c in xgb.classes_]

        for name, P, cls_order in (("logreg", lr_p, lr_cls),
                                   ("svm", sv_p, sv_cls),
                                   ("xgb", xg_p, xg_cls)):
            argmax = P.argmax(axis=1)
            # map probability columns to CLASSES order via the model's class order
            probs_cls = np.zeros_like(P)
            for local_i, cls_i in enumerate(cls_order):
                probs_cls[:, CLASSES.index(cls_i)] = P[:, local_i]
            for k, cls in enumerate(CLASSES):
                sig[f"{name}_prob_{cls}"][te] = probs_cls[:, k]
            sig[f"{name}_pred_enc"][te] = [CLASSES.index(cls_order[a]) for a in argmax]
            sig[f"{name}_confidence"][te] = P.max(axis=1)
            sig[f"{name}_entropy"][te] = [entropy_of_probs(r) for r in P]

        # per-sentence linear driving words (predicted-class coefficients)
        lr_argmax_local = lr_p.argmax(axis=1)
        sv_argmax_local = sv_p.argmax(axis=1)
        for name, model, argmax_local in (
                ("logreg", lr, lr_argmax_local),
                ("svm", svm, sv_argmax_local)):
            for j, i in enumerate(te):
                cls_i = int(argmax_local[j])   # local coef row index
                coef_row = model.coef_[cls_i]
                tok = top_tokens_from_linear_row(Xte[j], coef_row, feat_names,
                                                 TOP_K_FEATURES)
                if name == "logreg":
                    lr_feats[i] = tok
                else:
                    sv_feats[i] = tok
        print(f"  fold {fold + 1}/5 done")

    out = pd.DataFrame({
        "split": ["train"] * n_train + ["val"] * (len(pool) - n_train),
        "sentence": texts,
        "label": pool["label"].tolist(),
        **{f"logreg_prob_{c}": sig[f"logreg_prob_{c}"] for c in CLASSES},
        **{f"svm_prob_{c}": sig[f"svm_prob_{c}"] for c in CLASSES},
        **{f"xgb_prob_{c}": sig[f"xgb_prob_{c}"] for c in CLASSES},
    })
    out["logreg_pred"] = [CLASSES[int(e)] for e in sig["logreg_pred_enc"]]
    out["svm_pred"] = [CLASSES[int(e)] for e in sig["svm_pred_enc"]]
    out["xgb_pred"] = [CLASSES[int(e)] for e in sig["xgb_pred_enc"]]
    for name in ("logreg", "svm", "xgb"):
        out[f"{name}_confidence"] = sig[f"{name}_confidence"]
        out[f"{name}_entropy"] = sig[f"{name}_entropy"]
    out["logreg_top_features"] = lr_feats
    out["svm_top_features"] = sv_feats
    out["xgb_top_features_global"] = global_feats

    if not smoke:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(SIGNALS_CACHE, index=False)
        print(f"[signals] written: {SIGNALS_CACHE} ({len(out)} rows)")
    tr = out[out["split"] == "train"].drop(columns="split").reset_index(drop=True)
    vl = out[out["split"] == "val"].drop(columns="split").reset_index(drop=True)
    return tr, vl


# ═════════════════════════════════════════════════════════════════════════════
# Prompting — the ZERO-SHOT advisory-panel prompt (imported verbatim from
# zero_shot_slm_predictions.py) + tuned-FT training-format helpers
# ═════════════════════════════════════════════════════════════════════════════

def import_zero_shot_module():
    sys_path = str(Path("Classic").resolve())
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    import zero_shot_slm_predictions as zsm
    return zsm


class FormatError(RuntimeError):
    pass


def parse_prediction_label(raw: str) -> tuple[str, str]:
    """Lenient parse (label required, explanation optional) — identical to
    finetune_tuned.py."""
    text = str(raw).strip()
    # quote-tolerant: also matches JSON-style "label": "negative" (the
    # panel prompt instructs a JSON response; an under-trained model may
    # comply with the prompt instead of the trained 'Label:' format)
    label_match = re.search(
        r"label\s*[\"']?\s*:\s*[\"']?\s*(positive|negative|neutral)",
        text, flags=re.IGNORECASE)
    if not label_match:
        raise FormatError(f"Invalid response format: {text!r}")
    label = label_match.group(1).lower().strip()
    if label not in ALLOWED_LABELS:
        raise FormatError(f"Invalid label: {text!r}")
    expl_match = re.search(
        r"explanation\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    explanation = expl_match.group(1).strip() if expl_match else ""
    return label, explanation


def to_chat_content_blocks(messages):
    block_messages = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            block_messages.append(
                {"role": message["role"],
                 "content": [{"type": "text", "text": content}]})
        else:
            block_messages.append(message)
    return block_messages


def apply_chat_template_compat(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "string indices must be integers" not in str(exc):
            raise
        return tokenizer.apply_chat_template(to_chat_content_blocks(messages), **kwargs)


THINK_OPEN_TAGS = ("<think>", "<|im_start|>think")
THINK_CLOSE_TAG = "</think>"

EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"
THINK_CLOSE_BLOCK = "\n\n</think>\n\n"


def _force_no_think_prefill(tokenizer, inputs):
    """Token-level guarantee that Qwen thinking mode is OFF.

    Transformers 5.3.0 processors may ACCEPT the enable_thinking kwarg but
    IGNORE it with a warning ('not a valid argument for this processor'),
    leaving thinking mode active; the model then spends the whole
    generation budget reasoning before the answer. The deterministic fix:
    inspect the rendered generation prompt — if it does not already end
    with the (closed) empty think block, append the missing tokens so the
    model continues directly with the answer.
    """
    try:
        ids = inputs["input_ids"]
    except (KeyError, TypeError):
        return inputs
    tail = tokenizer.decode(ids[0, -12:], skip_special_tokens=False)
    if THINK_CLOSE_TAG in tail:
        return inputs  # flag was honoured; closed block present
    if any(tag in tail for tag in THINK_OPEN_TAGS):
        # thinking-mode template prefilled the opening tag; close it
        block, what = THINK_CLOSE_BLOCK, "closing think block appended"
    else:
        block, what = EMPTY_THINK_BLOCK, "empty think block appended"
    try:
        block_ids = tokenizer.encode(block, add_special_tokens=False)
    except (AttributeError, TypeError):
        block_ids = tokenizer(block, add_special_tokens=False)["input_ids"]
    if not block_ids:
        return inputs
    import torch
    inputs = dict(inputs)
    add = torch.tensor([block_ids], dtype=ids.dtype, device=ids.device)
    inputs["input_ids"] = torch.cat([ids, add], dim=1)
    if "attention_mask" in inputs:
        ones = torch.ones((1, len(block_ids)),
                          dtype=inputs["attention_mask"].dtype,
                          device=ids.device)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], ones],
                                             dim=1)
    print(f"    [no-think] {what} to generation prompt")
    return inputs


def apply_chat_template_no_think(tokenizer, messages, **kwargs):
    """Qwen template with thinking mode disabled — flag-based shim PLUS a
    token-level guarantee: even if the processor silently ignores
    enable_thinking (Transformers 5.3.0 warns and ignores), the empty think
    block is prefilled so no generation budget is spent reasoning."""
    call_kwargs = dict(enable_thinking=False, **kwargs)
    result = None
    try:
        result = tokenizer.apply_chat_template(messages, **call_kwargs)
    except TypeError as exc:
        err = str(exc)
        if "string indices must be integers" not in err and "enable_thinking" not in err:
            raise
        if "enable_thinking" in err:
            try:
                result = tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError as exc2:
                if "string indices must be integers" not in str(exc2):
                    raise
        if result is None:
            block_msgs = []
            for m in messages:
                content = m.get("content", "")
                block_msgs.append(
                    {"role": m["role"],
                     "content": [{"type": "text", "text": content}]
                     if isinstance(content, str) else content})
            try:
                result = tokenizer.apply_chat_template(block_msgs, **call_kwargs)
            except TypeError:
                result = tokenizer.apply_chat_template(block_msgs, **kwargs)
    if (kwargs.get("tokenize") and kwargs.get("return_dict")
            and kwargs.get("add_generation_prompt")
            and isinstance(result, dict) and "input_ids" in result):
        result = _force_no_think_prefill(tokenizer, result)
    return result


def normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    if text in ALLOWED_LABELS:
        return text
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return "neutral"


def panel_aug_user_prompt(build_prompt, signals_row: pd.Series) -> str:
    """The advisory-panel prompt built from a signals row.

    NOTE: the zero-shot prompt ends with an instruction to respond with a
    JSON object; for fine-tuning we keep the prompt VERBATIM (the model must
    learn to answer the real deployment prompt) and supply the training
    target in the 'Label: X' format used across all fine-tuning runs, so
    label accuracy remains comparable across conditions.
    """
    return build_prompt(signals_row)


def qwen_training_text(tokenizer, build_prompt, signals_row, label: str) -> str:
    messages = [
        {"role": "system",
         "content": "You classify financial sentiment into positive, negative, neutral."},
        {"role": "user", "content": panel_aug_user_prompt(build_prompt, signals_row)},
        {"role": "assistant", "content": f"Label: {label}"},
    ]
    return apply_chat_template_compat(
        tokenizer, messages, tokenize=False, add_generation_prompt=False)


def gemma_training_text(tokenizer, build_prompt, signals_row, label: str) -> str:
    messages = [
        {"role": "user",
         "content": [{"type": "text",
                      "text": panel_aug_user_prompt(build_prompt, signals_row)}]},
        {"role": "assistant",
         "content": [{"type": "text", "text": f"Label: {label}"}]},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return formatted.removeprefix("<bos>")


# ═════════════════════════════════════════════════════════════════════════════
# Model wrappers (identical to finetune_tuned.py)
# ═════════════════════════════════════════════════════════════════════════════

def resolve_local_snapshot(repo_id: str) -> str:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser(
        os.path.join("~", ".cache", "huggingface"))
    model_dir = os.path.join(hf_home, "hub", "models--" + repo_id.replace("/", "--"))
    ref = os.path.join(model_dir, "refs", "main")
    if os.path.isfile(ref):
        try:
            commit = open(ref).read().strip()
        except OSError:
            commit = ""
        snap = os.path.join(model_dir, "snapshots", commit)
        if os.path.isfile(os.path.join(snap, "config.json")):
            return snap
    import glob
    hits = sorted(glob.glob(os.path.join(model_dir, "snapshots", "*", "config.json")))
    if hits:
        return os.path.dirname(hits[-1])
    return repo_id


def load_model(model_key: str, cfg: BaseCfg, adapter_path: str | None = None):
    import torch
    if model_key == "qwen":
        from unsloth import FastLanguageModel
        src = adapter_path or resolve_local_snapshot(QWEN_MODEL)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=src, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit)
        if adapter_path:
            FastLanguageModel.for_inference(model)
    else:
        from unsloth import FastModel
        src = adapter_path or resolve_local_snapshot(GEMMA_MODEL)
        model, tokenizer = FastModel.from_pretrained(
            model_name=src, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit, load_in_8bit=False,
            full_finetuning=False)
        if adapter_path:
            FastModel.for_inference(model)
    return model, tokenizer


def attach_lora(model_key: str, model, lora_r: int, random_state: int):
    if model_key == "qwen":
        from unsloth import FastLanguageModel
        return FastLanguageModel.get_peft_model(
            model, r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_r, lora_dropout=0.0, bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=random_state, use_rslora=False)
    from unsloth import FastModel
    return FastModel.get_peft_model(
        model, r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_r, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_state, use_rslora=False)


# ═════════════════════════════════════════════════════════════════════════════
# Phases 2-4 — train, validate, test
# ═════════════════════════════════════════════════════════════════════════════

def prefix_for(model_key: str, smoke: bool) -> str:
    return f"{model_key}_ft_panaug{'_smoke' if smoke else ''}"


def train_panel_aug(model_key: str, cfg: BaseCfg, smoke: bool) -> Path:
    zsm = import_zero_shot_module()
    lora_r, max_steps = selected_config(model_key)
    if smoke:
        lora_r, max_steps = 16, 10
    cid = f"{model_key}_panaug_r{lora_r}_s{max_steps}{'_smoke' if smoke else ''}"
    adapter_dir = ADAPTER_ROOT / cid
    marker = adapter_dir / "TRAIN_DONE"

    if marker.exists():
        print(f"[{cid}] already trained — skipping")
        return adapter_dir

    sig, _val_sig = oof_signals(smoke=smoke)
    train_df = sig if not smoke else sig.head(50)
    print(f"[{cid}] training on {len(train_df)} panel-augmented rows "
          f"(lora_r={lora_r}, steps={max_steps})...")

    model, tokenizer = load_model(model_key, cfg)
    model = attach_lora(model_key, model, lora_r, cfg.random_state)

    text_fn = (qwen_training_text if model_key == "qwen"
               else gemma_training_text)
    from datasets import Dataset
    rows = []
    for _, row in train_df.iterrows():
        rows.append({"text": text_fn(tokenizer, zsm.build_prompt,
                                     row, normalize_label(row["label"]))})
    train_ds = Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)

    from trl import SFTConfig, SFTTrainer
    sft_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        max_steps=max_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=cfg.random_state,
        output_dir=str(OUT_DIR / "trainer_states" / cid),
        save_strategy="no",
        report_to="none",
    )
    kwargs = {"model": model, "train_dataset": train_ds, "args": sft_args}
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **kwargs)

    t0 = time.perf_counter()
    out = trainer.train()
    elapsed = time.perf_counter() - t0
    print(f"[{cid}] train_loss={out.metrics.get('train_loss'):.4f} "
          f"elapsed={elapsed/60:.1f} min")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (adapter_dir / "training_summary.json").write_text(json.dumps({
        "config_id": cid, "lora_r": lora_r, "max_steps": max_steps,
        "effective_epochs": round(max_steps * 2 * 4 / len(train_df), 3),
        "train_loss": out.metrics.get("train_loss"),
        "train_runtime_sec": elapsed,
        "prompt": "zero-shot advisory-panel prompt (sentence + panel signals)",
        "target": "Label: <gold label>",
        "panel_signals": "out-of-fold, 5-fold stratified on train+val pool",
        "base_config": asdict(cfg),
    }, indent=2))
    del model, tokenizer, trainer
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    marker.write_text("done")
    return adapter_dir


def infer_rows(model_key: str, model, tokenizer, prompts: list[str], cfg: BaseCfg,
               progress_every: int = 5):
    import torch
    preds, expls, times = [], [], []
    use_top_k = (model_key == "gemma")
    print(f"  generating {len(prompts)} predictions "
          f"(first call includes kernel warm-up: a few quiet minutes is NORMAL)...")
    for row_i, prompt in enumerate(prompts):
        if model_key == "qwen":
            messages = [
                {"role": "system",
                 "content": "Return only Label and Explanation fields exactly."},
                {"role": "user", "content": prompt},
            ]
            inputs = apply_chat_template_no_think(
                tokenizer, messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True)
        else:
            messages = [{"role": "user",
                         "content": [{"type": "text", "text": prompt}]}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        label, explanation, elapsed = None, None, 0.0
        last_raw = ""
        attempts = 0
        for _ in range(max(1, cfg.generation_retry_attempts)):
            attempts += 1
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id)
                if use_top_k:
                    gen_kwargs["top_k"] = cfg.top_k
                output = model.generate(**inputs, **gen_kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            generated = output[0][inputs["input_ids"].shape[1]:]
            last_raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
            try:
                label, explanation = parse_prediction_label(last_raw)
                break
            except FormatError:
                snippet = " ".join(str(last_raw).split())[:120]
                print(f"    row {row_i}: attempt {attempts} parse failure, "
                      f"retrying... raw[:120]={snippet!r}")
                continue
        if label is None:
            label, explanation = "[parse_error]", f"[parse_error] {last_raw[:200]}"
        preds.append(label)
        expls.append(explanation)
        times.append(elapsed)
        if (row_i + 1) % progress_every == 0 or (row_i + 1) == len(prompts):
            print(f"    {row_i + 1}/{len(prompts)} rows done "
                  f"(last row took {elapsed:.1f}s)")
    return preds, expls, times


def evaluate_split(model_key: str, adapter_dir: Path, cfg: BaseCfg,
                   split: str, smoke: bool):
    """Evaluate the trained adapter on validation (sanity) or test (final).

    Validation rows use the SAME out-of-fold signals the training rows use
    (they are part of the 5-fold CV pool, so their signals are computed
    leakage-free exactly like the stacking baseline). Test rows use the
    stored final-pool signals from test.csv --- byte-identical to what the
    zero-shot layer received (the deployment condition), and untouched by
    any training decision.
    """
    zsm = import_zero_shot_module()
    if split == "test":
        df = pd.read_csv(TEST_CSV)
        n = 20 if smoke else len(df)
        df = df.head(n).copy()
        required = [c for c in ("logreg_pred", "svm_pred", "xgb_pred",
                                "logreg_prob_negative", "xgb_entropy",
                                "logreg_top_features", "xgb_top_features_global")
                    if c not in df.columns]
        if required:
            raise FileNotFoundError(
                f"test.csv is missing panel signal columns: {required}")
    else:
        _tr_sig, val_sig = oof_signals(smoke=smoke)
        n = 20 if smoke else len(val_sig)
        df = val_sig.head(n).copy()

    prompts = [zsm.build_prompt(row) for _, row in df.iterrows()]
    model, tokenizer = load_model(model_key, cfg, adapter_path=str(adapter_dir))
    preds, expls, times = infer_rows(model_key, model, tokenizer, prompts, cfg)
    truth = df["label"].astype(str).str.lower().tolist()
    acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
    n_parse = sum(str(p).startswith("[parse_error") for p in preds)
    print(f"[{split}] accuracy: {acc*100:.2f}% (parse_errors={n_parse})")
    out = pd.DataFrame({"sentence": df["sentence"].astype(str).tolist(),
                        "label": df["label"].astype(str).tolist(),
                        "pred": preds, "explanation": expls, "time_sec": times})
    del model, tokenizer
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, acc, n_parse


def final_metrics(model_key: str, test_out: pd.DataFrame) -> dict:
    """Accuracy + macro F1 for the test predictions."""
    from collections import Counter
    y_true = [normalize_label(t) for t in test_out["label"]]
    y_pred = [str(p) for p in test_out["pred"]]
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
    f1s = []
    for cls in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s)}


def write_test_columns(model_key: str, test_out: pd.DataFrame, smoke: bool):
    prefix = prefix_for(model_key, smoke)
    df = pd.read_csv(TEST_CSV)
    for col in ("label", "explanation", "time_sec"):
        if f"{prefix}_{col}" not in df.columns:
            df[f"{prefix}_{col}"] = None
    for i, (_, r) in enumerate(test_out.iterrows()):
        idx = int(i)  # test_out preserves head(n) order == positional index
        df.at[idx, f"{prefix}_label"] = r["pred"]
        df.at[idx, f"{prefix}_explanation"] = r["explanation"]
        df.at[idx, f"{prefix}_time_sec"] = r["time_sec"]
    df.to_csv(TEST_CSV, index=False)
    print(f"test.csv updated: {prefix}_* columns "
          f"({len(test_out)} rows)")


def run_model(model_key: str, cfg: BaseCfg, smoke: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter_dir = train_panel_aug(model_key, cfg, smoke)

    # validation — sanity check only (selection was done in ft_tuning)
    val_cache = OUT_DIR / f"val_{model_key}{'_smoke' if smoke else ''}.csv"
    if val_cache.exists():
        val_out = pd.read_csv(val_cache)
        val_acc = (val_out["pred"].astype(str).str.lower()
                   == val_out["label"].astype(str).str.lower()).mean()
        print(f"[{model_key}] cached validation accuracy: {val_acc*100:.2f}%")
    else:
        val_out, val_acc, val_parse = evaluate_split(
            model_key, adapter_dir, cfg, "validation", smoke)
        val_out.to_csv(val_cache, index=False)

    # test — evaluated once
    prefix = prefix_for(model_key, smoke)
    test_df = pd.read_csv(TEST_CSV)
    n_test = 20 if smoke else len(test_df)
    if (f"{prefix}_label" in test_df.columns
            and test_df[f"{prefix}_label"].notna().sum() >= n_test):
        print(f"[{model_key}] test predictions already complete — reusing")
        test_out = pd.read_csv(OUT_DIR / f"test_{model_key}{'_smoke' if smoke else ''}.csv")
    else:
        test_out, test_acc, test_parse = evaluate_split(
            model_key, adapter_dir, cfg, "test", smoke)
        test_out.to_csv(OUT_DIR / f"test_{model_key}{'_smoke' if smoke else ''}.csv",
                        index=False)
        write_test_columns(model_key, test_out, smoke)
        val_acc_final = val_acc
        metrics = {
            "model": model_key,
            "config": "panel-augmented (selected tuned config)",
            "selected_config": dict(zip(("lora_r", "max_steps"),
                                        (selected_config(model_key)
                                         if not smoke else (16, 10)))),
            "prompt": "zero-shot advisory-panel prompt (sentence + panel signals)",
            "panel_signals_train": "out-of-fold (5-fold stratified, train+val pool)",
            "panel_signals_eval": "final stored artifacts (deployment condition)",
            "validation_accuracy": val_acc,
            "test_accuracy": test_acc,
            "test_parse_errors": test_parse,
            **{f"test_{k}": v for k, v in final_metrics(model_key, test_out).items()},
        }
        metrics_path = OUT_DIR / f"panel_aug_{model_key}{'_smoke' if smoke else ''}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"written: {metrics_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "gemma"), required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="10 train steps, 50 train rows, 20 val/test rows")
    args = ap.parse_args()

    for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not p.exists():
            raise FileNotFoundError(p)

    cfg = BaseCfg()
    if args.model == "gemma":
        cfg.temperature = 0.7
        cfg.top_p = 0.95
        cfg.top_k = 64

    if not args.smoke:
        # require the tuned-FT selection records (no retuning by design)
        selected_config(args.model)

    run_model(args.model, cfg, smoke=args.smoke)
    print("\nDONE.")


if __name__ == "__main__":
    main()