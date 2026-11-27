#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
import time

import pandas as pd

from common import (C1_DIR, COT_EXTRA_TOKENS, INPUTS, METHODS, MODELS, SEED, SAVE_EVERY,
                    SLMS, SMOKE_DIR, generate, load_slm, set_seed)

COLS = ["idx", "sentence", "label", "pred", "explanation", "recommendation", "reasoning",
        "time_sec", "attempts", "parse_error", "raw_on_failure"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--slm", required=True, choices=SLMS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    src = INPUTS / f"prompts_{args.method}.csv"
    if not src.exists():
        sys.exit(f"[job] missing {src} — run prep_inputs.py first")
    prompts = pd.read_csv(src)
    if args.smoke:
        prompts = prompts.head(5)

    out_dir = SMOKE_DIR if args.smoke else C1_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method}_{args.slm}.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=COLS)
    done_idx = set(done["idx"].astype(int)) if len(done) else set()
    todo = prompts[~prompts["idx"].isin(done_idx)]
    print(f"[job] {args.method}/{args.slm}: {len(done_idx)} done, {len(todo)} to go "
          f"-> {out_path}", flush=True)
    if todo.empty:
        return

    set_seed(SEED)
    model, tok = load_slm(args.slm)
    max_new = COT_EXTRA_TOKENS[args.slm] if args.method == "cot" else MODELS[args.slm]["max_new_tokens"]
    need_reasoning = args.method == "cot"

    rows = done.to_dict("records")
    t_start = time.time()
    for k, r in enumerate(todo.itertuples(), 1):
        res = generate(model, tok, args.slm, r.prompt, max_new, need_reasoning)
        rows.append({
            "idx": int(r.idx), "sentence": r.sentence, "label": r.label,
            "pred": res["label"] if res["label"] else r.fallback_label,
            "explanation": res["explanation"], "recommendation": res["recommendation"],
            "reasoning": res["reasoning"], "time_sec": round(res["time_sec"], 3),
            "attempts": res["attempts"], "parse_error": res["parse_error"],
            "raw_on_failure": res["raw"],
        })
        if k % SAVE_EVERY == 0 or k == len(todo):
            pd.DataFrame(rows, columns=COLS).sort_values("idx").to_csv(out_path, index=False)
            el = time.time() - t_start
            print(f"[job] {args.method}/{args.slm}: {len(rows)}/{len(prompts)} rows "
                  f"({el/k:.1f}s/row, checkpoint saved)", flush=True)

    del model, tok
    gc.collect()
    print(f"[job] {args.method}/{args.slm}: finished.", flush=True)


if __name__ == "__main__":
    main()
