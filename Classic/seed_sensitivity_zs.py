#!/usr/bin/env python3
"""
C6 (GPU half) — Decode-seed sensitivity of the Qwen3.5-4B zero-shot layer.

Re-runs the full 340-row zero-shot advisory-panel inference with Qwen3.5-4B
under additional decoding seeds (torch.manual_seed varies per run), measuring
label stability: the fraction of rows whose final label is identical across
runs, and the accuracy of each run. The original run (stored in qwen_zs_label)
corresponds to the default RNG state at load time.

The script reuses the EXACT original prompt builder (imported from
zero_shot_slm_predictions.py) and the identical sampling parameters
(temperature 0.3, top_p 0.9, 512 tokens, enable_thinking=False).

New columns written to test.csv per extra run (torch seed S = 1001, 1002, ...):
  qwen_zs_seed{S}_label (+ qwen_zs_seed{S}_explanation with --keep-text)

Usage (GPU server, from the project root):
    source ./activate_project.csh
    python Classic/seed_sensitivity_zs.py --runs 3
    python Classic/seed_sensitivity_zs.py --smoke        # 20-row check
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
RESULTS_DIR = Path("Classic/results")
SAVE_EVERY = 10
SEED_BASE = 1000


def resolve_local_snapshot(repo_id: str) -> str:
    """Resolve a HF repo id to its local cache snapshot directory if cached
    (avoids Hub access; unsloth's loader takes a purely local branch when
    given a directory path)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3,
                    help="number of extra decode-seed runs (default 3)")
    ap.add_argument("--keep-text", action="store_true",
                    help="store explanations for the extra runs too")
    ap.add_argument("--smoke", action="store_true",
                    help="20-row validation mode (columns suffixed _smoke)")
    args = ap.parse_args()

    sys.path.insert(0, "Classic")
    # reuse the original prompt builder + parser + chat-template shim
    from zero_shot_slm_predictions import (build_prompt, parse_response,
                                           FormatError,
                                           _apply_chat_template_compat,
                                           ZeroShotConfig)

    import torch
    from unsloth import FastLanguageModel

    cfg = ZeroShotConfig()
    df = pd.read_csv(TEST_CSV)
    n_rows = 20 if args.smoke else len(df)
    suffix = "_smoke" if args.smoke else ""
    if args.smoke:
        print("SMOKE MODE: first 20 rows, columns suffixed _smoke")

    df_work = df  # the FULL frame — checkpoints must never truncate test.csv
    work = df.iloc[:n_rows]     # only these rows are processed/analysed
    prompts = [build_prompt(row) for _, row in work.iterrows()]
    print(f"Built {len(prompts)} prompts (original advisory-panel format).")

    source = resolve_local_snapshot(cfg.qwen_model_name)
    print(f"Loading Qwen3.5-4B from: {source}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source, max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit)
    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_labels = {}
    truth = df_work["label"].iloc[:n_rows].astype(str).str.lower().tolist()
    for run in range(1, args.runs + 1):
        seed = SEED_BASE + run       # torch seeds 1001, 1002, ... (distinct
        torch.manual_seed(seed)      # from the original default RNG state)
        torch.cuda.manual_seed_all(seed)
        col = f"qwen_zs_seed{seed}_label{suffix}"
        if col not in df_work.columns:
            df_work[col] = None
        expl_col = f"qwen_zs_seed{seed}_explanation{suffix}"
        if args.keep_text and expl_col not in df_work.columns:
            df_work[expl_col] = None
        print(f"\n=== decode-seed run {run}/{args.runs} (torch seed {seed}) ===")
        n_done = 0
        for i, (idx, row) in enumerate(work.iterrows()):
            if pd.notna(df_work.at[idx, col]):
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
            ok = False
            for attempt in range(max(1, cfg.generation_retry_attempts)):
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=cfg.qwen_max_new_tokens,
                        do_sample=True, temperature=cfg.qwen_temperature,
                        top_p=cfg.qwen_top_p,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip()
                try:
                    label, expl, _rec = parse_response(text)
                    if args.keep_text:
                        df_work.at[idx, expl_col] = expl
                    ok = True
                    break
                except FormatError:
                    continue
            if not ok:
                df_work.at[idx, col] = "[parse_error]"
                print(f"  [run {run}] row {idx}: all retries failed")
            else:
                df_work.at[idx, col] = label
            n_done += 1
            if n_done % SAVE_EVERY == 0 or (i + 1) == n_rows:
                df_work.to_csv(TEST_CSV, index=False)
                print(f"  run {run}: {n_done} new rows this session "
                      f"({i+1}/{n_rows} considered, checkpoint saved)")
        df_work.to_csv(TEST_CSV, index=False)
        labels = df_work[col].iloc[:n_rows].astype(str).str.lower().tolist()
        all_labels[seed] = labels
        acc = sum(a == b for a, b in zip(labels, truth)) / n_rows
        n_err = sum(l.startswith("[parse_error]") for l in labels)
        print(f"  run {run} (seed {seed}): acc={acc*100:.2f}%  "
              f"parse_errors={n_err}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── stability analysis (original + extra runs) ─────────────────────────
    orig = df_work["qwen_zs_label"].iloc[:n_rows].astype(str).str.lower().tolist()
    runs_labels = [orig] + [all_labels[s] for s in sorted(all_labels)]
    n = n_rows
    same_all = sum(
        1 for i in range(n)
        if len({row[i] for row in runs_labels}) == 1
    )
    print(f"\nLabel identical across all {len(runs_labels)} runs: "
          f"{same_all}/{n} ({same_all/n*100:.1f}%)")
    per_run_acc = {}
    for j, row in enumerate(runs_labels):
        accj = sum(a == b for a, b in zip(row, truth)) / n
        name = "original" if j == 0 else f"seed{SEED_BASE + j}"
        per_run_acc[name] = accj
        print(f"  {name}: acc={accj*100:.2f}%")
    majority = [Counter(row[i] for row in runs_labels).most_common(1)[0][0]
                 for i in range(n)]
    maj_acc = sum(m == t for m, t in zip(majority, truth)) / n
    print(f"Majority-vote-over-runs accuracy: {maj_acc*100:.2f}%")

    out = {"n_rows": n, "n_runs": len(runs_labels),
           "seeds": [SEED_BASE + r for r in range(1, args.runs + 1)],
           "stability": same_all / n,
           "majority_vote_acc": maj_acc,
           "per_run_acc": per_run_acc}
    if not args.smoke:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "seed_sensitivity_zs.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"Written: {RESULTS_DIR}/seed_sensitivity_zs.json")
    else:
        print("SMOKE MODE: metrics printed only, no results files written.")

    # the full frame was already saved in-place; the merge below is a no-op
    # safety net that re-reads the file and re-assigns the same columns
    df_full = pd.read_csv(TEST_CSV)
    for c in df_work.columns:
        if c.startswith("qwen_zs_seed") and c.endswith(suffix):
            if c not in df_full.columns:
                df_full[c] = None
            df_full.loc[df_work.index, c] = df_work[c].values
    df_full.to_csv(TEST_CSV, index=False)
    print(f"Updated {TEST_CSV} with decode-seed label columns.")
    if args.smoke:
        print("\nSMOKE TEST COMPLETE.")


if __name__ == "__main__":
    main()