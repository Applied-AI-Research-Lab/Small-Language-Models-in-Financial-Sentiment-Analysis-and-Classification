#!/usr/bin/env python3
"""
Counterfactual panel-signal perturbation test (R1.4/R2.5 — automated half).

For the SAME 60 items sampled for the expert evaluation (manifest-linked), we
build counterfactual prompts in which every panel member's signal is cyclically
permuted (negative -> neutral -> positive -> negative): predictions, class
probabilities, confidence, entropy, and the panel-status line are all
re-rendered consistently, while the SENTENCE is left untouched.

Faithfulness metric: does the system's emitted label/explanation track the
perturbed signals (counterfactual label == cyclic shift of original label) or
the sentence (label unchanged)?  A high tracking rate demonstrates that the
explanations genuinely condition on the panel signals they cite --- the
counterfactual test the reviewers requested. BOTH deployed zero-shot SLMs
(Qwen3.5-4B and Gemma3-4B) are evaluated on the same 60 items.

Outputs (GPU server, from the project root):
  Datasets/financial_phrasebank/test.csv  +  qwen_cf_* columns (60 rows filled)
  Classic/results/counterfactual_metrics.json
  Classic/results/counterfactual_per_row.csv

Usage:
  source ./activate_project.csh
  python Classic/counterfactual_perturbation.py --smoke   # 8-row check
  python Classic/counterfactual_perturbation.py            # full 60 items
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import pandas as pd

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
MANIFEST = Path("Classic/results/expert_evaluation/sample_manifest.json")
RESULTS_DIR = Path("Classic/results")
SAVE_EVERY = 5
CLASSES = ("negative", "neutral", "positive")
CYCLE = {"negative": "neutral", "neutral": "positive", "positive": "negative"}
PREFIXES = ("logreg", "svm", "xgb")


def entropy_of(p):
    return -sum(0 if q <= 0 else q * math.log(q) for q in p)


def perturb_row(row) -> dict:
    """Cyclically permute every panel signal; keep the sentence and features.

    Returns a plain dict with the same keys build_prompt expects, so the
    original prompt builder renders the counterfactual prompt verbatim.
    """
    import numpy as np
    r = row.to_dict()
    for pfx in PREFIXES:
        probs = np.array([r[f"{pfx}_prob_{c}"] for c in CLASSES], dtype=float)
        # cyclic shift of the probability vector: neg<-neu, neu<-pos, pos<-neg
        new_probs = np.array([
            probs[CLASSES.index(CYCLE[c])] for c in CLASSES
        ])
        new_pred = CYCLE[str(r[f"{pfx}_pred"]).lower()]
        for k, cls in enumerate(CLASSES):
            r[f"{pfx}_prob_{cls}"] = float(new_probs[k])
        r[f"{pfx}_pred"] = new_pred
        r[f"{pfx}_confidence"] = float(new_probs.max())
        r[f"{pfx}_entropy"] = entropy_of(new_probs.tolist())
    return r


def resolve_local_snapshot(repo_id: str) -> str:
    import os
    import glob
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="8-item validation mode (columns suffixed _smoke)")
    args = ap.parse_args()

    sys.path.insert(0, "Classic")
    import torch
    from zero_shot_slm_predictions import (
        build_prompt, parse_response, FormatError,
        _apply_chat_template_compat, ZeroShotConfig)
    from unsloth import FastLanguageModel

    manifest = json.load(open(MANIFEST))
    item_ids = manifest["item_ids"]
    df_full = pd.read_csv(TEST_CSV)
    suffix = "_smoke" if args.smoke else ""
    n_items = 8 if args.smoke else len(item_ids)
    ids = item_ids[:n_items]
    print(f"counterfactual perturbation on {len(ids)} items "
          f"({'SMOKE' if args.smoke else 'full'})")

    for col in ("label", "explanation"):
        for sys_name in ("qwen", "gemma"):
            c = f"{sys_name}_cf_{col}{suffix}"
            if c not in df_full.columns:
                df_full[c] = None

    work = df_full.loc[ids]
    prompts = []
    for idx, row in work.iterrows():
        cf = perturb_row(row)
        prompts.append(build_prompt(pd.Series(cf)))

    cfg = ZeroShotConfig()

    # ── Qwen pass ────────────────────────────────────────────────────────
    from unsloth import FastLanguageModel
    source = resolve_local_snapshot(cfg.qwen_model_name)
    print(f"Loading Qwen3.5-4B from: {source}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source, max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit)
    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_done = 0
    for i, (idx, row) in enumerate(work.iterrows()):
        if pd.notna(df_full.at[idx, f"qwen_cf_label{suffix}"]):
            continue  # resume from checkpoint
        messages = [
            {"role": "system",
             "content": "You are a financial sentiment analyst. Follow the output format exactly."},
            {"role": "user", "content": prompts[i]},
        ]
        inputs = _apply_chat_template_compat(
            tokenizer, messages, enable_thinking=False,
            add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label = None
        for _ in range(max(1, cfg.generation_retry_attempts)):
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=cfg.qwen_max_new_tokens,
                    do_sample=True, temperature=cfg.qwen_temperature,
                    top_p=cfg.qwen_top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            try:
                label, expl, _rec = parse_response(text)
                df_full.at[idx, f"qwen_cf_label{suffix}"] = label
                df_full.at[idx, f"qwen_cf_explanation{suffix}"] = expl
                break
            except FormatError:
                continue
        if label is None:
            print(f"  [qwen-cf] row {idx}: all retries failed")
            df_full.at[idx, f"qwen_cf_label{suffix}"] = "[parse_error]"
        n_done += 1
        if n_done % SAVE_EVERY == 0:
            df_full.to_csv(TEST_CSV, index=False)
            print(f"  qwen: {n_done} items done (checkpoint saved)")
    df_full.to_csv(TEST_CSV, index=False)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Gemma pass ───────────────────────────────────────────────────────
    from unsloth import FastModel
    source_g = resolve_local_snapshot(cfg.gemma_model_name)
    print(f"Loading Gemma3-4B from: {source_g}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=source_g, max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit, load_in_8bit=False, full_finetuning=False)
    FastModel.for_inference(model)

    n_done = 0
    for i, (idx, row) in enumerate(work.iterrows()):
        if pd.notna(df_full.at[idx, f"gemma_cf_label{suffix}"]):
            continue
        messages = [{"role": "user", "content": [{"type": "text", "text": prompts[i]}]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label = None
        for _ in range(max(1, cfg.generation_retry_attempts)):
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=cfg.gemma_max_new_tokens,
                    do_sample=True, temperature=cfg.gemma_temperature,
                    top_p=cfg.gemma_top_p, top_k=cfg.gemma_top_k,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True).strip()
            try:
                label, expl, _rec = parse_response(text)
                df_full.at[idx, f"gemma_cf_label{suffix}"] = label
                df_full.at[idx, f"gemma_cf_explanation{suffix}"] = expl
                break
            except FormatError:
                continue
        if label is None:
            print(f"  [gemma-cf] row {idx}: all retries failed")
            df_full.at[idx, f"gemma_cf_label{suffix}"] = "[parse_error]"
        n_done += 1
        if n_done % SAVE_EVERY == 0:
            df_full.to_csv(TEST_CSV, index=False)
            print(f"  gemma: {n_done} items done (checkpoint saved)")
    df_full.to_csv(TEST_CSV, index=False)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── analysis ───────────────────────────────────────────────────────────
    sub = df_full.loc[ids]
    results = {}
    for sys_name, orig_col in (("qwen", "qwen_zs_label"), ("gemma", "gemma_zs_label")):
        orig_label = sub[orig_col].astype(str).str.lower()
        cf_label = sub[f"{sys_name}_cf_label{suffix}"].astype(str).str.lower()
        shifted = orig_label.map(CYCLE)

        valid = ~cf_label.str.startswith("[parse_error]")
        track = (cf_label == shifted) & valid
        stay = (cf_label == orig_label) & valid
        other = valid & ~track & ~stay

        print(f"\n== counterfactual results — {sys_name} (n={len(ids)}) ==")
        print(f"  parse errors          : {(~valid).sum()}")
        print(f"  tracks perturbed panel: {track.sum()}/{valid.sum()} "
              f"({track.sum()/max(valid.sum(),1)*100:.1f}%)")
        print(f"  stays with sentence   : {stay.sum()}/{valid.sum()} "
              f"({stay.sum()/max(valid.sum(),1)*100:.1f}%)")
        print(f"  neither (third class) : {other.sum()}/{valid.sum()} "
              f"({other.sum()/max(valid.sum(),1)*100:.1f}%)")
        results[sys_name] = {
            "n_valid": int(valid.sum()),
            "n_parse_error": int((~valid).sum()),
            "tracking_rate": float(track.sum() / max(valid.sum(), 1)),
            "sentence_anchored_rate": float(stay.sum() / max(valid.sum(), 1)),
            "neither_rate": float(other.sum() / max(valid.sum(), 1)),
        }

    if not args.smoke:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics = {
            "n_items": int(len(ids)),
            "perturbation": "cyclic negative->neutral->positive->negative "
                            "on all panel signals; sentence unchanged",
            "item_ids": [int(i) for i in ids],
            "systems": results,
        }
        with open(RESULTS_DIR / "counterfactual_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        per_rows = []
        for sys_name, orig_col in (("qwen", "qwen_zs_label"), ("gemma", "gemma_zs_label")):
            orig_label = sub[orig_col].astype(str).str.lower()
            cf_label = sub[f"{sys_name}_cf_label{suffix}"].astype(str).str.lower()
            shifted = orig_label.map(CYCLE)
            valid = ~cf_label.str.startswith("[parse_error]")
            per_rows.append(pd.DataFrame({
                "system": sys_name,
                "item_id": [int(i) for i in ids],
                "sentence": sub["sentence"],
                "orig_label": orig_label,
                "cf_label": cf_label,
                "expected_if_tracking": shifted,
                "tracks": (cf_label == shifted) & valid,
                "sentence_anchored": (cf_label == orig_label) & valid,
            }))
        pd.concat(per_rows).to_csv(RESULTS_DIR / "counterfactual_per_row.csv", index=False)
        print(f"written: {RESULTS_DIR}/counterfactual_metrics.json, "
              f"counterfactual_per_row.csv")
    else:
        print("SMOKE MODE: metrics printed only, no files written.")
        print("SMOKE TEST COMPLETE.")


if __name__ == "__main__":
    main()