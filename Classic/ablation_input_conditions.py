#!/usr/bin/env python3
"""
Input-ablation study: sentence-only (LLM-only) and panel-only zero-shot SLM
conditions (R1.2 / R2.2 / R2.1).
=============================================================================
The original framework (zero_shot_slm_predictions.py) gives each SLM the
sentence verbatim PLUS the three classical models' signals. This script runs
the two ablation conditions that isolate each information source:

  sentence-only ("LLM-only baseline"): prompt = task instruction + sentence
      + JSON instruction. NO panel signals. This is the LLM-only baseline the
      reviewers requested: how well does the SLM do the task alone?

  panel-only: prompt = the classical panel signals + agreement status + JSON
      instruction, WITHOUT the sentence. Tests how much of the SLM's accuracy
      comes from arbitrating second-order signals alone.

Together with the existing sentence+panel results (qwen_zs_*, gemma_zs_*),
this completes the ablation triad.

Procedure is identical to zero_shot_slm_predictions.py: same models
(unsloth/Qwen3.5-4B, unsloth/gemma-3-4b-it), same sampling parameters
(Qwen t=0.3/top_p=0.9/512 tok, enable_thinking=False; Gemma t=0.7/top_p=0.95/
top_k=64/256 tok), same JSON parsing with 4 retries, checkpoint every 10 rows.

New columns written to test.csv (existing columns untouched):
  qwen_sentonly_zs_{label,explanation,recommendation,time_sec}
  gemma_sentonly_zs_{...}
  qwen_panelonly_zs_{...}
  gemma_panelonly_zs_{...}

Fallback on persistent parse failure:
  sentence-only -> majority class of the training corpus ("neutral"), marked
  [parse_error] in the explanation (metrics exclude these rows);
  panel-only    -> classical panel majority vote (as in the original script).

Metrics saved to Classic/results/test_metrics_zs_ablation.csv; paired McNemar
tests against the full-information condition (qwen_zs_label) are printed and
saved to Classic/results/ablation_mcnemar.json.

Usage (GPU server):
    python Classic/ablation_input_conditions.py --smoke            # 20 rows, env check
    python Classic/ablation_input_conditions.py --condition sentence --slm qwen
    python Classic/ablation_input_conditions.py --condition sentence --slm gemma
    python Classic/ablation_input_conditions.py --condition panel --slm qwen
    python Classic/ablation_input_conditions.py --condition panel --slm gemma
    python Classic/ablation_input_conditions.py                   # all four runs
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:  # torch only needed on the GPU server at runtime
    torch = None

ALLOWED_LABELS = {"positive", "negative", "neutral"}
SAVE_EVERY = 10

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
RESULTS_DIR = Path("Classic/results")

JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object — no other text, no markdown fences:\n'
    '{"label": "<positive|negative|neutral>", '
    '"explanation": "<2-3 sentences>", '
    '"recommendation": "<1-2 sentences of actionable advice for a financial decision-maker>"}'
)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

class FormatError(RuntimeError):
    pass


def _agreement_status(preds: list[str]) -> str:
    unique = set(preds)
    if len(unique) == 1:
        return "Full agreement — all 3 models agree"
    if len(unique) == 2:
        counts = Counter(preds)
        top = counts.most_common()
        return (f"Majority agreement: {top[0][0]} ({top[0][1]}/3 models) "
                f"vs {top[1][0]} ({top[1][1]}/3)")
    return "Split — all 3 models disagree"


def build_sentence_only_prompt(row: pd.Series) -> str:
    """LLM-only baseline: task instruction + sentence + JSON instruction."""
    sentence = str(row["sentence"]).strip()
    return (
        "You are a senior financial sentiment analyst. Classify the sentiment "
        "of the following sentence from a financial news item.\n\n"
        f'SENTENCE: "{sentence}"\n\n'
        "Determine the correct sentiment (positive, negative, or neutral), "
        "explain your reasoning clearly by referencing the sentence content, "
        "and provide an actionable business recommendation for a financial "
        "decision-maker.\n\n"
        + JSON_INSTRUCTION
    )


def build_panel_only_prompt(row: pd.Series) -> str:
    """Panel-only condition: the original advisory prompt WITHOUT the sentence."""
    preds = [str(row["logreg_pred"]), str(row["svm_pred"]), str(row["xgb_pred"])]
    agree = _agreement_status(preds)

    lr_neg = float(row.get("logreg_prob_negative", 0)) * 100
    lr_neu = float(row.get("logreg_prob_neutral", 0)) * 100
    lr_pos = float(row.get("logreg_prob_positive", 0)) * 100
    sv_neg = float(row.get("svm_prob_negative", 0)) * 100
    sv_neu = float(row.get("svm_prob_neutral", 0)) * 100
    sv_pos = float(row.get("svm_prob_positive", 0)) * 100
    xg_neg = float(row.get("xgb_prob_negative", 0)) * 100
    xg_neu = float(row.get("xgb_prob_neutral", 0)) * 100
    xg_pos = float(row.get("xgb_prob_positive", 0)) * 100

    lr_conf = float(row.get("logreg_confidence", 0)) * 100
    sv_conf = float(row.get("svm_confidence", 0)) * 100
    xg_conf = float(row.get("xgb_confidence", 0)) * 100

    lr_ent = float(row.get("logreg_entropy", 0))
    sv_ent = float(row.get("svm_entropy", 0))
    xg_ent = float(row.get("xgb_entropy", 0))

    lr_feats = str(row.get("logreg_top_features", "N/A"))
    sv_feats = str(row.get("svm_top_features", "N/A"))
    xg_feats = str(row.get("xgb_top_features_global", "N/A"))

    return (
        "You are a senior financial sentiment analyst reviewing the output of a "
        "three-model advisory panel that has analyzed a financial news "
        "sentence. The sentence itself is not shown; judge only from the "
        "panel's signals below.\n\n"
        "━━━ ADVISORY PANEL SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "[1] LOGISTIC REGRESSION  (calibrated probability expert)\n"
        f"    Prediction   : {row['logreg_pred']}\n"
        f"    Confidence   : {lr_conf:.1f}%\n"
        f"    Probabilities: negative {lr_neg:.1f}% | neutral {lr_neu:.1f}% | positive {lr_pos:.1f}%\n"
        f"    Uncertainty  : {lr_ent:.4f}  (entropy — lower = more certain)\n"
        f"    Driving words: {lr_feats}\n\n"
        "[2] LINEAR SVM  (maximum-margin boundary expert)\n"
        f"    Prediction   : {row['svm_pred']}\n"
        f"    Confidence   : {sv_conf:.1f}%\n"
        f"    Probabilities: negative {sv_neg:.1f}% | neutral {sv_neu:.1f}% | positive {sv_pos:.1f}%\n"
        f"    Uncertainty  : {sv_ent:.4f}\n"
        f"    Driving words: {sv_feats}\n\n"
        "[3] XGBOOST  (non-linear keyword-interaction expert)\n"
        f"    Prediction   : {row['xgb_pred']}\n"
        f"    Confidence   : {xg_conf:.1f}%\n"
        f"    Probabilities: negative {xg_neg:.1f}% | neutral {xg_neu:.1f}% | positive {xg_pos:.1f}%\n"
        f"    Uncertainty  : {xg_ent:.4f}\n"
        f"    Key features : {xg_feats}\n\n"
        f"Panel status : {agree}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Based on all panel signals, determine the correct final sentiment, "
        "explain your reasoning clearly by referencing the model signals and "
        "key words, and provide an actionable business recommendation for a "
        "financial decision-maker.\n\n"
        + JSON_INSTRUCTION
    )


def parse_response(raw: str) -> tuple[str, str, str]:
    """Same semantics as zero_shot_slm_predictions.parse_response."""
    text = str(raw).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    json_m = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_m:
        raise FormatError(f"No JSON object found in: {text!r}")
    try:
        data = json.loads(json_m.group())
    except json.JSONDecodeError as exc:
        raise FormatError(f"JSON decode error ({exc}) in: {json_m.group()!r}")
    label = str(data.get("label", "")).strip().lower()
    explanation = str(data.get("explanation", "")).strip()
    recommendation = str(data.get("recommendation", "")).strip()
    if label not in ALLOWED_LABELS:
        raise FormatError(f"Invalid label '{label}' in: {data!r}")
    if not explanation or not recommendation:
        raise FormatError(f"Empty explanation or recommendation in: {data!r}")
    return label, explanation, recommendation


# ──────────────────────────────────────────────────────────────────────────────
# SLM inference (same protocol as zero_shot_slm_predictions.py)
# ──────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class SLMConfig:
    qwen_model_name: str = "unsloth/Qwen3.5-4B"
    gemma_model_name: str = "unsloth/gemma-3-4b-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    qwen_max_new_tokens: int = 512
    qwen_temperature: float = 0.3
    qwen_top_p: float = 0.9
    gemma_max_new_tokens: int = 256
    gemma_temperature: float = 0.7
    gemma_top_p: float = 0.95
    gemma_top_k: int = 64
    generation_retry_attempts: int = 4


def _apply_chat_template_compat(tokenizer, messages, enable_thinking=False, **kwargs):
    """Same compatibility shim as zero_shot_slm_predictions.py."""
    call_kwargs = dict(enable_thinking=enable_thinking, **kwargs)
    try:
        return tokenizer.apply_chat_template(messages, **call_kwargs)
    except TypeError as exc:
        err = str(exc)
        if "string indices must be integers" not in err and "enable_thinking" not in err:
            raise
        if "enable_thinking" in err:
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError as exc2:
                if "string indices must be integers" not in str(exc2):
                    raise
        block_msgs = []
        for m in messages:
            content = m.get("content", "")
            block_msgs.append({
                "role": m["role"],
                "content": [{"type": "text", "text": content}]
                if isinstance(content, str) else content,
            })
        try:
            return tokenizer.apply_chat_template(block_msgs, **call_kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(block_msgs, **kwargs)


def _generate_and_parse(model, tokenizer, messages, gen_kwargs, cfg, tag):
    if torch is None:
        raise RuntimeError("torch is required on the GPU server")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if gen_kwargs.get("_gemma"):
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True)
    else:
        inputs = _apply_chat_template_compat(
            tokenizer, messages, enable_thinking=False,
            add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    total_time = 0.0
    max_attempts = max(1, cfg.generation_retry_attempts)
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=gen_kwargs["max_new_tokens"],
                do_sample=True,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs.get("top_p"),
                top_k=gen_kwargs.get("top_k"),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        generated = output[0][inputs["input_ids"].shape[1]:]
        last_raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            return *parse_response(last_raw), total_time
        except FormatError as parse_exc:
            print(f"  [{tag}] Attempt {attempt}/{max_attempts} failed — "
                  f"parse error: {parse_exc}")
            continue
    raise FormatError(last_raw)


def run_condition(df: pd.DataFrame, prompts: list[str], cfg: SLMConfig,
                   slm: str, condition: str, test_path: Path):
    """Run one (SLM, condition) cell; checkpoint into test.csv."""
    prefix = f"{slm}_{condition}_zs"          # e.g. qwen_sentonly_zs

    if slm == "qwen":
        tag = "Qwen"
        from unsloth import FastLanguageModel
        source = resolve_local_snapshot(cfg.qwen_model_name)
        if source != cfg.qwen_model_name:
            print(f"Loading Qwen3.5-4B from local snapshot: {source}")
        else:
            print("Loading Qwen3.5-4B ...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=source, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit)
        FastLanguageModel.for_inference(model)
        sys_msg = "You are a financial sentiment analyst. Follow the output format exactly."
        gen = {"max_new_tokens": cfg.qwen_max_new_tokens,
               "temperature": cfg.qwen_temperature, "top_p": cfg.qwen_top_p}
        mk = lambda p: [{"role": "system", "content": sys_msg},
                        {"role": "user", "content": p}]
    else:
        tag = "Gemma"
        from unsloth import FastModel
        source = resolve_local_snapshot(cfg.gemma_model_name)
        if source != cfg.gemma_model_name:
            print(f"Loading Gemma3-4B from local snapshot: {source}")
        else:
            print("Loading Gemma3-4B ...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=source, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit, load_in_8bit=False,
            full_finetuning=False)
        FastModel.for_inference(model)
        gen = {"max_new_tokens": cfg.gemma_max_new_tokens,
               "temperature": cfg.gemma_temperature, "top_p": cfg.gemma_top_p,
               "top_k": cfg.gemma_top_k, "_gemma": True}
        mk = lambda p: [{"role": "user",
                         "content": [{"type": "text", "text": p}]}]

    for col in (f"{prefix}_label", f"{prefix}_explanation",
                f"{prefix}_recommendation", f"{prefix}_time_sec"):
        if col not in df.columns:
            df[col] = None

    n = len(df)
    n_fb = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.notna(df.at[idx, f"{prefix}_label"]):
            continue
        prompt = prompts[i]
        try:
            label, expl, rec, elapsed = _generate_and_parse(
                model, tokenizer, mk(prompt), gen, cfg, tag)
        except FormatError as exc:
            raw = str(exc)
            print(f"\n[{tag}/{condition}] ALL RETRIES FAILED — row {i} (idx {idx}).")
            if condition == "panelonly":
                label = Counter([str(row["logreg_pred"]), str(row["svm_pred"]),
                                 str(row["xgb_pred"])]).most_common(1)[0][0]
            else:  # sentence-only: no panel to fall back on
                label = "neutral"          # majority class; excluded from metrics
                label = f"[parse_error]{label}"
            expl = f"[parse_error] {raw[:300]}"
            rec = "Unable to generate recommendation due to format error."
            elapsed = 0.0
            n_fb += 1
        df.at[idx, f"{prefix}_label"] = label
        df.at[idx, f"{prefix}_explanation"] = expl
        df.at[idx, f"{prefix}_recommendation"] = rec
        df.at[idx, f"{prefix}_time_sec"] = elapsed
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            df.to_csv(test_path, index=False)
            print(f"  {tag}/{condition}: {i+1}/{n} rows done (checkpoint)")

    del model, tokenizer
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{tag}/{condition} complete. Fallbacks: {n_fb}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Metrics + McNemar
# ──────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: list[str], y_pred: list[str]):
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    f1s, sups = [], []
    for cls in ("negative", "neutral", "positive"):
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t != cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != cls and t == cls)
        pe = tp / (tp + fp) if tp + fp else 0.0
        re = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pe * re / (pe + re) if pe + re else 0.0)
        sups.append(sum(1 for t in y_true if t == cls))
    macro_f1 = sum(f1s) / len(f1s)
    weighted_f1 = sum(f * s for f, s in zip(f1s, sups)) / n
    return acc, macro_f1, weighted_f1


def mcnemar_exact(y_a: list[str], y_b: list[str], truth: list[str]):
    b = sum(1 for a, c, t in zip(y_a, y_b, truth) if a == t and c != t)
    c_ = sum(1 for a, c, t in zip(y_a, y_b, truth) if a != t and c == t)
    nd = b + c_
    if nd == 0:
        return b, c_, 0.0, 1.0
    chi2 = (abs(b - c_) - 1) ** 2 / nd
    k = min(b, c_)
    tail = sum(math.comb(nd, i) for i in range(k + 1)) / 2 ** nd
    return b, c_, chi2, min(1.0, 2 * tail)


def compute_and_save_metrics(df: pd.DataFrame, cells: list[tuple[str, str]],
                             full_run: bool) -> None:
    """cells = [(slm, condition), ...]"""
    truth = df["label"].str.strip().str.lower().tolist()
    rows = []
    mcn = {}
    ref_q = df["qwen_zs_label"].astype(str).str.strip().str.lower().tolist() \
        if "qwen_zs_label" in df.columns else None
    for slm, condition in cells:
        col = f"{slm}_{condition}_zs_label"
        if col not in df.columns or df[col].isna().all():
            continue
        clean = df[col].notna() & ~df[col].astype(str).str.startswith(
            "[parse_error]", na=False)
        sub = df[clean]
        if sub.empty:
            continue
        y_pred = sub[col].astype(str).str.strip().str.lower().tolist()
        y_true = sub["label"].astype(str).str.strip().str.lower().tolist()
        acc, mf1, wf1 = _metrics(y_true, y_pred)
        name = f"{slm}_{condition}"
        rows.append({
            "model": name, "n_evaluated": len(sub),
            "accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "avg_time_sec": df.loc[clean, f"{slm}_{condition}_zs_time_sec"].mean(),
        })
        # paired comparison vs the full-information condition (same rows)
        if ref_q is not None:
            ref = df.loc[clean, "qwen_zs_label"].astype(str).str.strip().str.lower().tolist()
            b, c, chi2, p = mcnemar_exact(y_pred, ref, y_true)
            mcn[f"{name}_vs_{slm}_sentence+panel" if slm == "qwen"
                else f"{name}_vs_qwen_sentence+panel"] = {
                "b": b, "c": c, "chi2_cc": chi2, "p_exact": p}
            print(f"[McNemar] {name} vs full-info condition: "
                  f"b={b} c={c} chi2={chi2:.2f} p={p:.4f}")
        # sentence-only vs panel-only (both SLM cells done?)
    if not full_run:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        md = pd.DataFrame(rows)
        md.to_csv(RESULTS_DIR / "test_metrics_zs_ablation.csv", index=False)
        print(f"Metrics saved to: {RESULTS_DIR}/test_metrics_zs_ablation.csv")
        print(md.to_string(index=False))
    if mcn:
        with open(RESULTS_DIR / "ablation_mcnemar.json", "w") as f:
            json.dump(mcn, f, indent=2)
        print(f"McNemar saved to: {RESULTS_DIR}/ablation_mcnemar.json")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Hub-robust model loading
# ──────────────────────────────────────────────────────────────────────────────

def _hub_reachable() -> bool:
    """Quick reachability probe for huggingface.co."""
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


def resolve_local_snapshot(repo_id: str) -> str:
    """Resolve a HF repo id to its local cache snapshot directory if cached.

    Loading from the local snapshot avoids ANY Hub access (robust to network
    outages on the node) and pins the run to the exact cached revision --- the
    same weights used by the previously completed experiments. This also
    sidesteps unsloth's loader, which queries the Hub unconditionally for
    repo-style names but takes a purely local branch when given a directory
    path. Returns the repo id unchanged if no local snapshot exists (normal
    online behaviour).
    """
    import os, glob
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser(
        os.path.join("~", ".cache", "huggingface"))
    model_dir = os.path.join(hf_home, "hub",
                             "models--" + repo_id.replace("/", "--"))
    ref = os.path.join(model_dir, "refs", "main")
    if os.path.isfile(ref):
        try:
            commit = open(ref).read().strip()
        except OSError:
            commit = ""
        snap = os.path.join(model_dir, "snapshots", commit)
        if os.path.isfile(os.path.join(snap, "config.json")):
            return snap
    hits = sorted(glob.glob(os.path.join(model_dir, "snapshots", "*",
                                         "config.json")))
    if hits:
        return os.path.dirname(hits[-1])
    return repo_id


def main() -> None:
    # Prefer LOCAL cache snapshots: identical weights as previous runs, and no
    # Hub dependency (the node's DNS has proven flaky). Falls back to normal
    # online loading if a model is not cached.
    hub_ok = _hub_reachable()
    if not hub_ok:
        print("WARNING: huggingface.co unreachable from this node.")
    _cfg = SLMConfig()
    missing = [r for r in (_cfg.qwen_model_name, _cfg.gemma_model_name)
               if resolve_local_snapshot(r) == r]
    if missing and not hub_ok:
        raise RuntimeError(
            f"Hub unreachable and no local cache snapshot found for: {missing}. ")
    if missing:
        print(f"NOTE: no local snapshot for {missing}; will load from the Hub.")
    else:
        print("Both SLMs found in local HF cache; loading from snapshots.")

    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["sentence", "panel", "both"],
                    default="both")
    ap.add_argument("--slm", choices=["qwen", "gemma", "both"], default="both")
    ap.add_argument("--smoke", action="store_true",
                    help="20-row environment check (one SLM, one condition)")
    args = ap.parse_args()

    test_path = TEST_CSV
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")
    df = pd.read_csv(test_path)
    required = ["sentence", "label",
                "logreg_pred", "logreg_confidence", "logreg_entropy",
                "logreg_prob_negative", "logreg_prob_neutral", "logreg_prob_positive",
                "logreg_top_features",
                "svm_pred", "svm_confidence", "svm_entropy",
                "svm_prob_negative", "svm_prob_neutral", "svm_prob_positive",
                "svm_top_features",
                "xgb_pred", "xgb_confidence", "xgb_entropy",
                "xgb_prob_negative", "xgb_prob_neutral", "xgb_prob_positive",
                "xgb_top_features_global"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in test CSV: {missing}\n"
                         "Run Classic/predict_classic_models.py first.")

    n_rows = 20 if args.smoke else len(df)
    df_work = df.iloc[:n_rows].copy()

    conditions = ["sentence", "panel"] if args.condition == "both" else [args.condition]
    slms = ["qwen", "gemma"] if args.slm == "both" else [args.slm]
    if args.smoke:
        conditions, slms = conditions[:1], slms[:1]

    builder = {"sentence": build_sentence_only_prompt,
               "panel": build_panel_only_prompt}
    cond_tag = {"sentence": "sentonly", "panel": "panelonly"}

    cells = []
    for condition in conditions:
        prompts = [builder[condition](row) for _, row in df_work.iterrows()]
        print(f"\nBuilt {len(prompts)} {condition}-only prompts.")
        for slm in slms:
            cells.append((slm, cond_tag[condition]))
            df_work = run_condition(df_work, prompts, SLMConfig(), slm,
                                     cond_tag[condition], test_path)

    # merge back into the full test.csv
    new_cols = [c for c in df_work.columns
                if (c.endswith("_zs_label") or c.endswith("_zs_explanation")
                    or c.endswith("_zs_recommendation") or c.endswith("_zs_time_sec"))
                and ("sentonly" in c or "panelonly" in c)]
    for col in new_cols:
        if col not in df.columns:
            df[col] = None
    mask = df.index.isin(df_work.index)
    for col in new_cols:
        df.loc[mask, col] = df_work[col]
    df.to_csv(test_path, index=False)
    print(f"\nUpdated test CSV with new columns: {test_path}")

    compute_and_save_metrics(df_work, cells, full_run=not args.smoke)

    if args.smoke:
        print("\nSMOKE TEST COMPLETE — ablation chain OK.")
        for col in new_cols:
            if col.endswith("_explanation"):
                vals = df_work[col].dropna()
                if len(vals):
                    print(f"--- sample {col} ---")
                    print(str(vals.iloc[0])[:400])
                    break


if __name__ == "__main__":
    main()