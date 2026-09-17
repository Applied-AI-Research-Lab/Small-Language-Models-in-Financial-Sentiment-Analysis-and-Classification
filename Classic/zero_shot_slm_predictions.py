#!/usr/bin/env python3

from __future__ import annotations

import gc
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

ALLOWED_LABELS = {"positive", "negative", "neutral"}

SAVE_EVERY = 10


@dataclass
class ZeroShotConfig:
    test_csv: str = "Datasets/financial_phrasebank/test.csv"
    results_dir: str = "Classic/results"

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


class FormatError(RuntimeError):
    pass


def _majority_vote(preds: list[str]) -> str:
    return Counter(preds).most_common(1)[0][0]


def _agreement_status(preds: list[str]) -> str:
    unique = set(preds)
    if len(unique) == 1:
        return "Full agreement — all 3 models agree"
    if len(unique) == 2:
        counts = Counter(preds)
        top = counts.most_common()
        return (
            f"Majority agreement: {top[0][0]} ({top[0][1]}/3 models) "
            f"vs {top[1][0]} ({top[1][1]}/3)"
        )
    return "Split — all 3 models disagree"


def build_prompt(row: pd.Series) -> str:
    preds = [str(row["logreg_pred"]), str(row["svm_pred"]), str(row["xgb_pred"])]
    agree = _agreement_status(preds)
    sentence = str(row["sentence"]).strip()

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
        "three-model advisory panel that has already analyzed the sentence below.\n\n"
        f'SENTENCE: "{sentence}"\n\n'
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
        "Based on all panel signals, determine the correct final sentiment, explain "
        "your reasoning clearly by referencing the model signals and key words, and "
        "provide an actionable business recommendation for a financial decision-maker.\n\n"
        'Respond with ONLY a JSON object — no other text, no markdown fences:\n'
        '{"label": "<positive|negative|neutral>", '
        '"explanation": "<2-3 sentences referencing panel signals and key words>", '
        '"recommendation": "<1-2 sentences of actionable advice for a financial decision-maker>"}'
    )


def parse_response(raw: str) -> tuple[str, str, str]:
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


def normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    if text in ALLOWED_LABELS:
        return text
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return "neutral"


def _apply_chat_template_compat(tokenizer, messages: list[dict], enable_thinking: bool = False, **kwargs):
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
            block_msgs.append(
                {
                    "role": m["role"],
                    "content": [{"type": "text", "text": content}]
                    if isinstance(content, str)
                    else content,
                }
            )
        try:
            return tokenizer.apply_chat_template(block_msgs, **call_kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(block_msgs, **kwargs)


def load_qwen_zero_shot(cfg: ZeroShotConfig):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.qwen_model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def infer_qwen(model, tokenizer, prompt: str, cfg: ZeroShotConfig) -> tuple[str, str, str, float]:
    messages = [
        {
            "role": "system",
            "content": "You are a financial sentiment analyst. Follow the output format exactly.",
        },
        {"role": "user", "content": prompt},
    ]

    inputs = _apply_chat_template_compat(
        tokenizer,
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    total_time = 0.0
    last_raw = ""
    max_attempts = max(1, cfg.generation_retry_attempts)

    for attempt in range(1, max_attempts + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=cfg.qwen_max_new_tokens,
                do_sample=True,
                temperature=cfg.qwen_temperature,
                top_p=cfg.qwen_top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        generated = output[0][inputs["input_ids"].shape[1]:]
        last_raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            label, explanation, recommendation = parse_response(last_raw)
            return label, explanation, recommendation, total_time
        except FormatError as parse_exc:
            print(
                f"  [Qwen] Attempt {attempt}/{max_attempts} failed — "
                f"parse error: {parse_exc}"
            )
            continue

    raise FormatError(last_raw)


def run_qwen_zero_shot(df: pd.DataFrame, prompts: list[str], cfg: ZeroShotConfig) -> pd.DataFrame:
    print("Loading Qwen3.5-4B for zero-shot inference...")
    model, tokenizer = load_qwen_zero_shot(cfg)

    if "qwen_zs_label" not in df.columns:
        df["qwen_zs_label"] = None
        df["qwen_zs_explanation"] = None
        df["qwen_zs_recommendation"] = None
        df["qwen_zs_time_sec"] = None

    test_path = Path(cfg.test_csv)
    n = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.notna(df.at[idx, "qwen_zs_label"]):
            continue

        prompt = prompts[i]
        try:
            label, expl, rec, elapsed = infer_qwen(model, tokenizer, prompt, cfg)
        except FormatError as exc:
            raw = str(exc)
            print(
                f"\n{'='*70}\n"
                f"[Qwen zero-shot] ALL RETRIES FAILED — row {i} (test index {idx})\n"
                f"{'='*70}\n"
                f"--- PROMPT ---\n{prompt}\n"
                f"--- LAST MODEL RESPONSE ---\n{raw}\n"
                f"{'='*70}\n"
                f"Falling back to classic majority vote.\n"
            )
            label = _majority_vote(
                [str(row["logreg_pred"]), str(row["svm_pred"]), str(row["xgb_pred"])]
            )
            expl = f"[parse_error] {raw[:300]}"
            rec = "Unable to generate recommendation due to format error."
            elapsed = 0.0

        df.at[idx, "qwen_zs_label"] = label
        df.at[idx, "qwen_zs_explanation"] = expl
        df.at[idx, "qwen_zs_recommendation"] = rec
        df.at[idx, "qwen_zs_time_sec"] = elapsed

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            df.to_csv(test_path, index=False)
            print(f"  Qwen zero-shot: {i + 1}/{n} rows done (checkpoint saved)")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Qwen zero-shot inference complete.")
    return df


def load_gemma_zero_shot(cfg: ZeroShotConfig):
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.gemma_model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    FastModel.for_inference(model)
    return model, tokenizer


def infer_gemma(model, tokenizer, prompt: str, cfg: ZeroShotConfig) -> tuple[str, str, str, float]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    total_time = 0.0
    last_raw = ""
    max_attempts = max(1, cfg.generation_retry_attempts)

    for attempt in range(1, max_attempts + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=cfg.gemma_max_new_tokens,
                do_sample=True,
                temperature=cfg.gemma_temperature,
                top_p=cfg.gemma_top_p,
                top_k=cfg.gemma_top_k,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        generated = output[0][inputs["input_ids"].shape[1]:]
        last_raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            label, explanation, recommendation = parse_response(last_raw)
            return label, explanation, recommendation, total_time
        except FormatError as parse_exc:
            print(
                f"  [Gemma] Attempt {attempt}/{max_attempts} failed — "
                f"parse error: {parse_exc}"
            )
            continue

    raise FormatError(last_raw)


def run_gemma_zero_shot(df: pd.DataFrame, prompts: list[str], cfg: ZeroShotConfig) -> pd.DataFrame:
    print("Loading Gemma3-4B for zero-shot inference...")
    model, tokenizer = load_gemma_zero_shot(cfg)

    if "gemma_zs_label" not in df.columns:
        df["gemma_zs_label"] = None
        df["gemma_zs_explanation"] = None
        df["gemma_zs_recommendation"] = None
        df["gemma_zs_time_sec"] = None

    test_path = Path(cfg.test_csv)
    n = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.notna(df.at[idx, "gemma_zs_label"]):
            continue

        prompt = prompts[i]
        try:
            label, expl, rec, elapsed = infer_gemma(model, tokenizer, prompt, cfg)
        except FormatError as exc:
            raw = str(exc)
            print(
                f"\n{'='*70}\n"
                f"[Gemma zero-shot] ALL RETRIES FAILED — row {i} (test index {idx})\n"
                f"{'='*70}\n"
                f"--- PROMPT ---\n{prompt}\n"
                f"--- LAST MODEL RESPONSE ---\n{raw}\n"
                f"{'='*70}\n"
                f"Falling back to classic majority vote.\n"
            )
            label = _majority_vote(
                [str(row["logreg_pred"]), str(row["svm_pred"]), str(row["xgb_pred"])]
            )
            expl = f"[parse_error] {raw[:300]}"
            rec = "Unable to generate recommendation due to format error."
            elapsed = 0.0

        df.at[idx, "gemma_zs_label"] = label
        df.at[idx, "gemma_zs_explanation"] = expl
        df.at[idx, "gemma_zs_recommendation"] = rec
        df.at[idx, "gemma_zs_time_sec"] = elapsed

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            df.to_csv(test_path, index=False)
            print(f"  Gemma zero-shot: {i + 1}/{n} rows done (checkpoint saved)")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Gemma zero-shot inference complete.")
    return df


def compute_and_save_metrics(df: pd.DataFrame, cfg: ZeroShotConfig) -> None:
    from sklearn.metrics import accuracy_score, f1_score

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_prefix, col in [("qwen_zs", "qwen_zs_label"), ("gemma_zs", "gemma_zs_label")]:
        mask = df[col].notna() & ~df[col].str.startswith("[parse_error]", na=False)
        sub = df[mask]
        if sub.empty:
            continue
        y_true = sub["label"].str.strip().str.lower()
        y_pred = sub[col].str.strip().str.lower()
        rows.append(
            {
                "model": model_prefix,
                "n_evaluated": len(sub),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                "avg_time_sec": df[f"{model_prefix}_time_sec"].mean(),
            }
        )

    if rows:
        metrics_df = pd.DataFrame(rows)
        metrics_path = results_dir / "test_metrics_zs.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Zero-shot metrics saved to: {metrics_path}")
        print(metrics_df.to_string(index=False))


def main() -> None:
    cfg = ZeroShotConfig()
    test_path = Path(cfg.test_csv)

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {test_path}\n"
            "Run Classic/predict_classic_models.py first to populate classic signals."
        )

    df = pd.read_csv(test_path)

    required_cols = [
        "sentence", "label",
        "logreg_pred", "logreg_confidence", "logreg_entropy",
        "logreg_prob_negative", "logreg_prob_neutral", "logreg_prob_positive",
        "logreg_top_features",
        "svm_pred", "svm_confidence", "svm_entropy",
        "svm_prob_negative", "svm_prob_neutral", "svm_prob_positive",
        "svm_top_features",
        "xgb_pred", "xgb_confidence", "xgb_entropy",
        "xgb_prob_negative", "xgb_prob_neutral", "xgb_prob_positive",
        "xgb_top_features_global",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing classic model signal columns in test CSV: {missing}\n"
            "Run Classic/predict_classic_models.py first."
        )

    print(f"Loaded {len(df)} test rows with classic model signals.")

    print("Building advisory-panel prompts...")
    prompts = [build_prompt(row) for _, row in df.iterrows()]
    print(f"Built {len(prompts)} prompts.")

    df = run_qwen_zero_shot(df, prompts, cfg)

    df = run_gemma_zero_shot(df, prompts, cfg)

    df.to_csv(test_path, index=False)
    print(f"\nUpdated test CSV: {test_path}")

    compute_and_save_metrics(df, cfg)

    print("\nZero-shot SLM predictions completed.")
    print("New columns added to test CSV:")
    print("  qwen_zs_label, qwen_zs_explanation, qwen_zs_recommendation, qwen_zs_time_sec")
    print("  gemma_zs_label, gemma_zs_explanation, gemma_zs_recommendation, gemma_zs_time_sec")


if __name__ == "__main__":
    main()
