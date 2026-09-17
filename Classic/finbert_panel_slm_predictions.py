#!/usr/bin/env python3
"""
FinBERT-as-Panel Zero-Shot SLM Experiment
==========================================
Same procedure as Classic/zero_shot_slm_predictions.py, but the advisory panel
is a single FinBERT model instead of the three classical classifiers.

Per test row, FinBERT provides:
  * predicted label (argmax over its 3-class financial sentiment head)
  * class probabilities (softmax)          -> as percentages
  * confidence (max probability)           -> as percentage
  * Shannon entropy of the probability vector
  * NO lexical driving words (opaque transformer; the prompt states this
    explicitly, mirroring the framework's transparency argument).

These signals are assembled into a structured advisory prompt (same style as
the classic panel prompt) and passed in zero-shot mode to Qwen3.5-4B and
Gemma3-4B, which produce: final label, explanation, recommendation --- exactly
as in the original framework.

New columns written to test.csv (NOT overwriting any existing columns):
  finbert_qwen_zs_label, finbert_qwen_zs_explanation,
  finbert_qwen_zs_recommendation, finbert_qwen_zs_time_sec
  finbert_gemma_zs_label, ... (same for Gemma)

Also saved: Classic/results/test_metrics_zs_finbert_panel.csv

Usage (GPU server):
    python Classic/finbert_panel_slm_predictions.py --smoke        # 20 rows
    python Classic/finbert_panel_slm_predictions.py --panel-model zero-shot
    python Classic/finbert_panel_slm_predictions.py --panel-model ft
    python Classic/finbert_panel_slm_predictions.py --slm qwen     # only Qwen
    python Classic/finbert_panel_slm_predictions.py --slm gemma    # only Gemma
    python Classic/finbert_panel_slm_predictions.py                # both SLMs

Panel model choice:
    --panel-model zero-shot  (default) -> ProsusAI/finbert as released
    --panel-model ft               -> the checkpoint fine-tuned by
                                         finbert_baseline.py (3 epochs on the
                                         train split, seed 42). If that
                                         checkpoint does not exist yet, the
                                         script errors with instructions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:  # torch only needed on the GPU server at runtime
    torch = None

ALLOWED_LABELS = {"positive", "negative", "neutral"}
OUR_LABELS = ("negative", "neutral", "positive")
SAVE_EVERY = 10

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
RESULTS_DIR = Path("Classic/results")
FINBERT_RELEASED = "ProsusAI/finbert"
FINBERT_FT_CKPT = Path("Classic/results/finbert_ft_checkpoint")
MAX_LEN = 256
EVAL_BATCH = 64


# ──────────────────────────────────────────────────────────────────────────────
# FinBERT panel signal extraction
# ──────────────────────────────────────────────────────────────────────────────

def load_finbert(panel_model: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if panel_model == "ft":
        if not FINBERT_FT_CKPT.exists():
            raise FileNotFoundError(
                f"Fine-tuned FinBERT checkpoint not found at {FINBERT_FT_CKPT}.\n"
                "Run Classic/finbert_baseline.py first (it saves the checkpoint), "
                "or use --panel-model zero-shot."
            )
        name = str(FINBERT_FT_CKPT)
    else:
        name = FINBERT_RELEASED
    print(f"Loading FinBERT panel model: {name} ...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    # label order (ProsusAI/finbert: {0: positive, 1: negative, 2: neutral});
    # id2label keys may be int or str depending on transformers version
    raw_map = model.config.id2label
    id2label = {int(k): str(v).lower() for k, v in raw_map.items()}
    hf_labels = [id2label[i] for i in range(model.config.num_labels)]
    assert set(hf_labels) == set(OUR_LABELS), f"unexpected labels: {hf_labels}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, hf_labels, device


def finbert_probs_for_all(model, tokenizer, hf_labels, device,
                          texts: list[str]) -> list[dict]:
    """Return per-row dict: pred, probs{neg,neu,pos}, confidence, entropy."""
    if torch is None:
        raise RuntimeError("torch is required on the GPU server")
    hf2our = {i: OUR_LABELS.index(hf_labels[i]) for i in range(len(hf_labels))}
    out = []
    for i in range(0, len(texts), EVAL_BATCH):
        batch = texts[i:i + EVAL_BATCH]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=MAX_LEN, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        reord = torch.zeros_like(probs)
        for hf_i in range(len(hf_labels)):
            reord[:, hf2our[hf_i]] = probs[:, hf_i]
        for row_probs in reord.tolist():
            p = dict(zip(OUR_LABELS, row_probs))
            pred = max(p, key=p.get)
            conf = p[pred]
            ent = -sum(q * math.log(q + 1e-12) for q in row_probs)
            out.append({"pred": pred, "probs": p, "confidence": conf,
                        "entropy": ent})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder (FinBERT single-member panel)
# ──────────────────────────────────────────────────────────────────────────────

class FormatError(RuntimeError):
    pass


def build_finbert_prompt(sentence: str, sig: dict) -> str:
    p = sig["probs"]
    return (
        "You are a senior financial sentiment analyst reviewing the output of a "
        "financial-domain transformer advisory model that has already analyzed "
        "the sentence below.\n\n"
        f'SENTENCE: "{sentence.strip()}"\n\n'
        "━━━ ADVISORY MODEL SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "[1] FINBERT  (financial-domain transformer, pre-trained on financial text)\n"
        f"    Prediction   : {sig['pred']}\n"
        f"    Confidence   : {sig['confidence']*100:.1f}%\n"
        f"    Probabilities: negative {p['negative']*100:.1f}% | "
        f"neutral {p['neutral']*100:.1f}% | positive {p['positive']*100:.1f}%\n"
        f"    Uncertainty  : {sig['entropy']:.4f}  (entropy — lower = more certain)\n"
        "    Driving words: not available (opaque transformer; no lexical "
        "feature attribution provided)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Based on the advisory model signals and the sentence itself, determine "
        "the correct final sentiment, explain your reasoning clearly by "
        "referencing the model signals and the sentence content, and provide an "
        "actionable business recommendation for a financial decision-maker.\n\n"
        'Respond with ONLY a JSON object — no other text, no markdown fences:\n'
        '{"label": "<positive|negative|neutral>", '
        '"explanation": "<2-3 sentences referencing the model signals and the sentence>", '
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


# ──────────────────────────────────────────────────────────────────────────────
# SLM inference (same protocol as zero_shot_slm_predictions.py)
# ──────────────────────────────────────────────────────────────────────────────

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
    """Same compatibility shim as zero_shot_slm_predictions.py (handles Qwen
    block-format content and enable_thinking template flag)."""
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


def load_qwen(cfg: SLMConfig):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.qwen_model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_gemma(cfg: SLMConfig):
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


def _generate_and_parse(model, tokenizer, messages, gen_kwargs, cfg, tag):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
    ) if gen_kwargs.get("_gemma") else _apply_chat_template_compat(
        tokenizer, messages, enable_thinking=False,
        add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
    )
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
            label, explanation, recommendation = parse_response(last_raw)
            return label, explanation, recommendation, total_time
        except FormatError as parse_exc:
            print(f"  [{tag}] Attempt {attempt}/{max_attempts} failed — "
                  f"parse error: {parse_exc}")
            continue
    raise FormatError(last_raw)


def run_slm_over_finbert(df: pd.DataFrame, prompts: list[str],
                          cfg: SLMConfig, slm: str,
                          finbert_preds: list[str], test_path: Path):
    """Run one SLM over the FinBERT-panel prompts; checkpoint into test.csv."""
    if slm == "qwen":
        prefix = "finbert_qwen_zs"
        print("Loading Qwen3.5-4B ...")
        model, tokenizer = load_qwen(cfg)
        sys_msg = "You are a financial sentiment analyst. Follow the output format exactly."
        gen_kwargs = {"max_new_tokens": cfg.qwen_max_new_tokens,
                      "temperature": cfg.qwen_temperature,
                      "top_p": cfg.qwen_top_p}
        template_messages = lambda p: [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": p},
        ]
        tag = "Qwen"
    else:
        prefix = "finbert_gemma_zs"
        print("Loading Gemma3-4B ...")
        model, tokenizer = load_gemma(cfg)
        gen_kwargs = {"max_new_tokens": cfg.gemma_max_new_tokens,
                      "temperature": cfg.gemma_temperature,
                      "top_p": cfg.gemma_top_p,
                      "top_k": cfg.gemma_top_k,
                      "_gemma": True}
        template_messages = lambda p: [
            {"role": "user", "content": [{"type": "text", "text": p}]},
        ]
        tag = "Gemma"

    for col in (f"{prefix}_label", f"{prefix}_explanation",
                f"{prefix}_recommendation", f"{prefix}_time_sec"):
        if col not in df.columns:
            df[col] = None

    n = len(df)
    n_fallback = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.notna(df.at[idx, f"{prefix}_label"]):
            continue
        prompt = prompts[i]
        try:
            label, expl, rec, elapsed = _generate_and_parse(
                model, tokenizer, template_messages(prompt), gen_kwargs, cfg, tag)
        except FormatError as exc:
            raw = str(exc)
            print(f"\n[{tag}] ALL RETRIES FAILED — row {i} (idx {idx}). "
                  f"Falling back to FinBERT panel prediction.")
            label = finbert_preds[i]          # single-member panel: its vote
            expl = f"[parse_error] {raw[:300]}"
            rec = "Unable to generate recommendation due to format error."
            elapsed = 0.0
            n_fallback += 1
        df.at[idx, f"{prefix}_label"] = label
        df.at[idx, f"{prefix}_explanation"] = expl
        df.at[idx, f"{prefix}_recommendation"] = rec
        df.at[idx, f"{prefix}_time_sec"] = elapsed
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            df.to_csv(test_path, index=False)
            print(f"  {tag} (FinBERT panel): {i+1}/{n} rows done (checkpoint)")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{tag} over FinBERT panel complete. Fallbacks: {n_fallback}")
    return df


def compute_and_save_metrics(df: pd.DataFrame, which: list[str]) -> None:
    from sklearn.metrics import accuracy_score, f1_score
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    name_map = {"qwen": ("finbert_qwen_zs", "qwen_over_finbert"),
                "gemma": ("finbert_gemma_zs", "gemma_over_finbert")}
    for slm in which:
        prefix, model_name = name_map[slm]
        col = f"{prefix}_label"
        if col not in df.columns or df[col].isna().all():
            continue
        mask = df[col].notna() & ~df[col].astype(str).str.startswith("[parse_error]", na=False)
        sub = df[mask]
        if sub.empty:
            continue
        y_true = sub["label"].str.strip().str.lower()
        y_pred = sub[col].str.strip().str.lower()
        rows.append({
            "model": model_name,
            "n_evaluated": len(sub),
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "avg_time_sec": df[f"{prefix}_time_sec"].mean(),
        })
    if rows:
        m = pd.DataFrame(rows)
        path = RESULTS_DIR / "test_metrics_zs_finbert_panel.csv"
        m.to_csv(path, index=False)
        print(f"Metrics saved to: {path}")
        print(m.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-model", choices=["zero-shot", "ft"],
                    default="zero-shot",
                    help="FinBERT checkpoint to use as the panel")
    ap.add_argument("--slm", choices=["qwen", "gemma", "both"], default="both")
    ap.add_argument("--smoke", action="store_true",
                    help="20-row environment + protocol check")
    args = ap.parse_args()

    cfg = SLMConfig()
    test_path = TEST_CSV
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")
    df = pd.read_csv(test_path)
    required = ["sentence", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    n_rows = 20 if args.smoke else len(df)
    df_work = df.iloc[:n_rows].copy()

    # 1. FinBERT panel signals for all rows
    fb_model, fb_tok, hf_labels, device = load_finbert(args.panel_model)
    texts = df_work["sentence"].astype(str).tolist()
    print(f"Extracting FinBERT panel signals for {len(texts)} rows ...")
    sigs = finbert_probs_for_all(fb_model, fb_tok, hf_labels, device, texts)
    fb_preds = [s["pred"] for s in sigs]
    # quick report: panel-only accuracy on these rows
    acc = sum(1 for s, t in zip(fb_preds, df_work["label"]) if s == t) / len(df_work)
    print(f"FinBERT panel-only accuracy on these {len(df_work)} rows: {acc*100:.2f}%")

    del fb_model, fb_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Build prompts
    prompts = [build_finbert_prompt(s, sig) for s, sig in
               zip(df_work["sentence"].astype(str), sigs)]
    print(f"Built {len(prompts)} FinBERT-panel prompts.")

    # 3. Run SLMs
    which = ["qwen", "gemma"] if args.slm == "both" else [args.slm]
    if args.smoke:
        which = which[:1]  # smoke: one SLM is enough to validate the chain
    for slm in which:
        df_work = run_slm_over_finbert(df_work, prompts, cfg, slm, fb_preds, test_path)

    # 4. Merge back into the full test.csv (only the rows we ran)
    for col in [c for c in df_work.columns if c.startswith("finbert_") and c.endswith(("_label", "_explanation", "_recommendation", "_time_sec"))]:
        if col not in df.columns:
            df[col] = None
    mask = df.index.isin(df_work.index)
    for col in [c for c in df_work.columns if c.startswith("finbert_") and c.endswith(("_label", "_explanation", "_recommendation", "_time_sec"))]:
        df.loc[mask, col] = df_work[col]
    df.to_csv(test_path, index=False)
    print(f"\nUpdated test CSV with new columns: {test_path}")

    # 5. Metrics
    compute_and_save_metrics(df_work, which)

    if args.smoke:
        print("\nSMOKE TEST COMPLETE — FinBERT-panel → SLM chain OK.")
        print("Sample explanation from the smoke run:")
        for c in [col for col in df_work.columns if col.endswith("_explanation") and col.startswith("finbert_")]:
            vals = df_work[c].dropna()
            if len(vals):
                print(f"--- {c} ---")
                print(str(vals.iloc[0])[:400])
                break


if __name__ == "__main__":
    main()