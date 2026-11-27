#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from common import (C1_DIR, FRAMEWORK_COL, METHODS, RESULTS, SENTONLY_COL, SLMS, SMOKE_DIR,
                    TEST_CSV)


def mcnemar(y, a, b):
    ca, cb = (a == y), (b == y)
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_exact": float(p)}


def metrics(name, y, pred, split, parse_err=None, t=None):
    return {"system": name, "n": len(y),
            "accuracy": round(100 * accuracy_score(y, pred), 2),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y, pred, average="weighted"), 4),
            "acc_unanimous": round(100 * (pred[~split] == y[~split]).mean(), 2) if (~split).any() else None,
            "acc_split": round(100 * (pred[split] == y[split]).mean(), 2) if split.any() else None,
            "parse_failures": parse_err, "mean_time_sec": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    src = SMOKE_DIR if args.smoke else C1_DIR
    out = SMOKE_DIR if args.smoke else RESULTS

    ref = pd.read_csv(TEST_CSV)
    base = ref[["sentence", "label"]].copy()
    base.insert(0, "idx", range(len(base)))
    preds = ref[["logreg_pred", "svm_pred", "xgb_pred"]]
    base["panel_status"] = (preds.nunique(axis=1) > 1).map({True: "split", False: "unanimous"})
    for slm in SLMS:
        base[f"{slm}_framework"] = ref[FRAMEWORK_COL[slm]]
        base[f"{slm}_sentonly"] = ref[SENTONLY_COL[slm]]

    jobs = {}
    for m in METHODS:
        for s in SLMS:
            f = src / f"{m}_{s}.csv"
            if f.exists():
                jobs[(m, s)] = pd.read_csv(f)
    if not jobs:
        raise SystemExit(f"no job outputs in {src}")

    merged = base.copy()
    for (m, s), df in jobs.items():
        df = df.set_index("idx")
        for c in ("pred", "explanation", "recommendation", "reasoning", "parse_error", "time_sec"):
            merged[f"{m}_{s}_{c}"] = merged["idx"].map(df[c])
    merged.to_csv(out / "test_round2.csv", index=False)

    rows, tests = [], {}
    for s in SLMS:
        sub_all = merged
        split_all = sub_all["panel_status"].eq("split").values
        y_all = sub_all["label"].values
        rows.append(metrics(f"{s} framework (Round 1)", y_all, sub_all[f"{s}_framework"].values, split_all))
        rows.append(metrics(f"{s} sentence-only (Round 1)", y_all, sub_all[f"{s}_sentonly"].values, split_all))
        for m in METHODS:
            if (m, s) not in jobs:
                continue
            col = f"{m}_{s}_pred"
            sub = merged[merged[col].notna()]
            y, p = sub["label"].values, sub[col].values
            sp = sub["panel_status"].eq("split").values
            rows.append(metrics(f"{s} {m}", y, p, sp,
                                int(sub[f"{m}_{s}_parse_error"].sum()),
                                round(float(sub[f"{m}_{s}_time_sec"].mean()), 2)))
            tests[f"{s}/{m}"] = {
                "n": len(sub),
                "vs_framework": mcnemar(y, p, sub[f"{s}_framework"].values),
                "vs_sentence_only": mcnemar(y, p, sub[f"{s}_sentonly"].values),
            }
    met = pd.DataFrame(rows)
    met.to_csv(out / "metrics_c1.csv", index=False)
    (out / "mcnemar_c1.json").write_text(json.dumps(tests, indent=2))

    try:
        table = met.to_markdown(index=False)
    except ImportError:
        table = "```\n" + met.to_string(index=False) + "\n```"
    lines = ["# Round-2 C1 results" + (" (SMOKE — 5 rows, not for the paper)" if args.smoke else ""),
             "", table, "", "## Exact McNemar (job = A)", "",
             "| job | n | vs framework (A✓B✗ / A✗B✓, p) | vs sentence-only (A✓B✗ / A✗B✓, p) |",
             "|---|---|---|---|"]
    for k, v in tests.items():
        f, so = v["vs_framework"], v["vs_sentence_only"]
        lines.append(f"| {k} | {v['n']} | {f['a_right_b_wrong']}/{f['a_wrong_b_right']}, "
                     f"{f['p_exact']:.4f} | {so['a_right_b_wrong']}/{so['a_wrong_b_right']}, "
                     f"{so['p_exact']:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
