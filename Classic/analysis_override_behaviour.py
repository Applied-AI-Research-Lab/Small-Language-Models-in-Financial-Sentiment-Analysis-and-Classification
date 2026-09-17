"""B2 — Override behaviour analysis (R2.4 data half).

Quantifies how each zero-shot SLM deviates from the classical panel on the
340-row test set:

- Unanimous-row overrides (SLM disagrees with a unanimous panel verdict):
    correct overrides   (panel wrong -> SLM right)  = "rescues"
    incorrect overrides (panel right -> SLM wrong)  = "damages"
  precision = correct / total unanimous overrides
  recovery  = rescues / unanimous-but-wrong rows (17)
- Split-row deviations from the panel majority, with outcome breakdown
  (SLM right / majority right / both wrong)
- Total label changes vs. the panel majority vote (net label-change rate)
- Net accuracy change vs. the majority vote (recorded, not tabulated)

Outputs (Classic/results/):
  override_behaviour.csv / .json  — all computed metrics
  Console: verification summary + LaTeX table fragment

Stdlib only; fully deterministic.
"""

import csv
import json
from collections import Counter
from pathlib import Path

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
OUT_DIR = Path("Classic/results")

SLMS = {
    "Qwen3.5-4B (ZS)": "qwen_zs_label",
    "Gemma3-4B (ZS)": "gemma_zs_label",
}


def majority(preds):
    """Panel majority vote (no 3-way ties occur in this test set)."""
    return Counter(preds).most_common(1)[0][0]


def main():
    rows = list(csv.DictReader(open(TEST_CSV)))
    n = len(rows)

    panel_preds = [(r["logreg_pred"], r["svm_pred"], r["xgb_pred"]) for r in rows]
    unanimous = [len(set(p)) == 1 for p in panel_preds]
    maj = [majority(p) for p in panel_preds]
    truth = [r["label"] for r in rows]

    n_unan = sum(unanimous)
    unan_wrong = [unanimous[i] and maj[i] != truth[i] for i in range(n)]
    n_unan_wrong = sum(unan_wrong)
    n_split = n - n_unan
    maj_correct = sum(1 for i in range(n) if maj[i] == truth[i])

    results = {
        "n_rows": n,
        "n_unanimous": n_unan,
        "n_split": n_split,
        "n_unanimous_wrong": n_unan_wrong,
        "majority_vote_accuracy": maj_correct / n,
    }

    per_slm = {}
    for name, col in SLMS.items():
        pred = [r[col] for r in rows]

        # --- Unanimous-row overrides ---
        ov_unan = [i for i in range(n) if unanimous[i] and pred[i] != maj[i]]
        ov_correct = [i for i in ov_unan if pred[i] == truth[i]]
        ov_incorrect = [i for i in ov_unan if pred[i] != truth[i]]
        precision = len(ov_correct) / len(ov_unan) if ov_unan else None
        # consistency check: a correct override on a unanimous row rescues an error
        assert all(unan_wrong[i] for i in ov_correct), "correct override on correct panel?"
        recovery = len(ov_correct) / n_unan_wrong

        # --- Split-row deviations from majority ---
        split_dev = [i for i in range(n) if not unanimous[i] and pred[i] != maj[i]]
        sd_slm_right = sum(1 for i in split_dev if pred[i] == truth[i])
        sd_maj_right = sum(1 for i in split_dev if maj[i] == truth[i])
        sd_both_wrong = len(split_dev) - sd_slm_right - sd_maj_right

        # --- Totals vs majority vote ---
        changes = [i for i in range(n) if pred[i] != maj[i]]
        slm_correct = sum(1 for i in range(n) if pred[i] == truth[i])

        # --- Consistency checks ---
        assert len(changes) == len(ov_unan) + len(split_dev)
        assert len(ov_correct) + len(ov_incorrect) == len(ov_unan)

        per_slm[name] = {
            "unanimous_overrides": len(ov_unan),
            "override_correct": len(ov_correct),
            "override_incorrect": len(ov_incorrect),
            "override_precision": precision,
            "unanimous_error_recovery": recovery,
            "split_deviations": len(split_dev),
            "split_dev_slm_right": sd_slm_right,
            "split_dev_majority_right": sd_maj_right,
            "split_dev_both_wrong": sd_both_wrong,
            "total_label_changes": len(changes),
            "label_change_rate": len(changes) / n,
            "slm_accuracy": slm_correct / n,
            "net_accuracy_vs_majority_pp": (slm_correct - maj_correct) / n * 100,
        }

    results["per_slm"] = per_slm

    # ── Persist ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "override_behaviour.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "override_behaviour.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", *SLMS.keys()])
        q = per_slm["Qwen3.5-4B (ZS)"]
        g = per_slm["Gemma3-4B (ZS)"]
        rows_csv = [
            ("unanimous_row_overrides", q["unanimous_overrides"], g["unanimous_overrides"]),
            ("override_correct", q["override_correct"], g["override_correct"]),
            ("override_incorrect", q["override_incorrect"], g["override_incorrect"]),
            ("override_precision", q["override_precision"], g["override_precision"]),
            ("unanimous_error_recovery", q["unanimous_error_recovery"], g["unanimous_error_recovery"]),
            ("split_row_deviations", q["split_deviations"], g["split_deviations"]),
            ("total_label_changes", q["total_label_changes"], g["total_label_changes"]),
            ("label_change_rate", q["label_change_rate"], g["label_change_rate"]),
            ("net_accuracy_vs_majority_pp", q["net_accuracy_vs_majority_pp"],
             g["net_accuracy_vs_majority_pp"]),
        ]
        for m, a, b in rows_csv:
            w.writerow([m, a, b])

    # ── Console summary ──────────────────────────────────────────────────
    print(f"n={n} | unanimous={n_unan} (wrong on {n_unan_wrong}) | split={n_split}")
    print(f"majority-vote accuracy: {maj_correct}/{n} = {maj_correct/n*100:.2f}%\n")
    for name, m in per_slm.items():
        print(f"{name}:")
        print(f"  unanimous overrides: {m['unanimous_overrides']} "
              f"(correct {m['override_correct']}, incorrect {m['override_incorrect']})")
        print(f"  precision: {m['override_precision']*100:.1f}%  "
              f"recovery: {m['unanimous_error_recovery']*100:.1f}%")
        print(f"  split deviations: {m['split_deviations']} "
              f"(SLM right {m['split_dev_slm_right']}, majority right "
              f"{m['split_dev_majority_right']}, both wrong {m['split_dev_both_wrong']})")
        print(f"  total label changes: {m['total_label_changes']} "
              f"({m['label_change_rate']*100:.1f}%)")
        print(f"  accuracy: {m['slm_accuracy']*100:.2f}%  "
              f"net vs majority: {m['net_accuracy_vs_majority_pp']:+.2f} pp\n")

    # ── LaTeX fragment ────────────────────────────────────────────────────
    print("% ---- LaTeX table fragment ----")
    for name, m in per_slm.items():
        prec = f"{m['override_precision']*100:.1f}\\%"
        rec = f"{m['unanimous_error_recovery']*100:.1f}\\%"
        print(f"% {name}: overrides {m['unanimous_overrides']} ({m['override_correct']}/{m['override_incorrect']}), "
              f"precision {prec}, recovery {rec}, split dev {m['split_deviations']}, "
              f"changes {m['total_label_changes']} ({m['label_change_rate']*100:.1f}\\%)")


if __name__ == "__main__":
    main()