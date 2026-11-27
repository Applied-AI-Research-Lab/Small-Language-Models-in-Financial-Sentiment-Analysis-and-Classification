#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

from common import (BERT_CKPT, MODELS, OOF_CSV, PROJECT, TEST_CSV, TRAIN_CSV, generate,
                    load_slm, resolve_snapshot, set_seed)

EXPECTED = {"unsloth": "2026.3.8", "torch": "2.10.0", "transformers": "5.3.0"}
REPRO_ROWS = [0, 12, 1, 2, 3]


def check_versions(problems, warnings):
    for pkg, want in EXPECTED.items():
        try:
            have = importlib.import_module(pkg).__version__
        except Exception as exc:
            problems.append(f"cannot import {pkg}: {exc}")
            continue
        ok = have.split("+")[0] == want
        print(f"  {pkg:13s} {have:18s} expected {want:10s} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(f"{pkg} {have} != {want} (not the Round-1 environment)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true")
    args = ap.parse_args()
    problems, warnings = [], []

    print("== Python / packages")
    print(f"  python        {sys.version.split()[0]}  ({sys.executable})")
    check_versions(problems, warnings)

    print("== Offline mode")
    for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        print(f"  {v}={os.environ.get(v)}")
        if os.environ.get(v) != "1":
            warnings.append(f"{v} is not 1 (the runner sets it; set it if you run jobs by hand)")
    print(f"  HF_HOME={os.environ.get('HF_HOME')}")

    print("== Input files (read-only)")
    for p in (TEST_CSV, TRAIN_CSV, OOF_CSV):
        print(f"  {'OK ' if p.exists() else 'MISSING'} {p}")
        if not p.exists():
            problems.append(f"missing {p}")
    if TEST_CSV.exists():
        import pandas as pd
        cols = set(pd.read_csv(TEST_CSV, nrows=1).columns)
        need = {"qwen_zs_label", "gemma_zs_label", "qwen_sentonly_zs_label",
                "gemma_sentonly_zs_label", "svm_pred", "svm_confidence"}
        if need - cols:
            problems.append(f"test.csv lacks Round-1 columns: {sorted(need - cols)}")
        else:
            print("  test.csv has all Round-1 reference columns")
    print(f"  {'OK ' if BERT_CKPT.exists() else 'MISSING'} {BERT_CKPT}")
    if not BERT_CKPT.exists():
        problems.append(f"BERT plug-in checkpoint missing: {BERT_CKPT}")

    print("== Pinned model snapshots")
    for slm in MODELS:
        try:
            print(f"  {slm:5s} {resolve_snapshot(slm)}")
        except FileNotFoundError as exc:
            problems.append(str(exc))

    print("== GPUs")
    try:
        import torch
        print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}  "
              f"(CUDA {torch.version.cuda}), devices visible: {torch.cuda.device_count()}")
        if not torch.cuda.is_available():
            problems.append("CUDA not available")
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True)
        for line in out.stdout.strip().splitlines():
            print("  GPU " + line)
    except Exception as exc:
        problems.append(f"GPU check failed: {exc}")

    if args.repro and not problems:
        print("== Reproduction check (original framework prompt, Qwen, 5 rows)")
        sys.path.insert(0, str(PROJECT / "Classic"))
        from zero_shot_slm_predictions import build_prompt
        import pandas as pd
        df = pd.read_csv(TEST_CSV)
        set_seed()
        model, tok = load_slm("qwen")
        hits = 0
        for i in REPRO_ROWS:
            res = generate(model, tok, "qwen", build_prompt(df.iloc[i]), MODELS["qwen"]["max_new_tokens"])
            stored = df.at[i, "qwen_zs_label"]
            hits += res["label"] == stored
            print(f"  row {i}: new={res['label']}  stored={stored}")
        print(f"  reproduced {hits}/{len(REPRO_ROWS)}")
        if hits < len(REPRO_ROWS):
            problems.append(f"reproduction check failed ({hits}/{len(REPRO_ROWS)})")

    print("== Summary")
    for w in warnings:
        print(f"  WARNING: {w}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    if problems:
        print("PREFLIGHT FAILED")
        sys.exit(1)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
