"""B1 — Statistical significance analysis for the seven-model comparison.

Reads the per-row test predictions (Datasets/financial_phrasebank/test.csv),
and produces:

1. Paired McNemar tests for all 21 model pairs:
   - chi-squared statistic with continuity correction,
   - exact two-sided binomial p-value (appropriate for small discordant counts).
2. Bootstrap confidence intervals (10,000 resamples, seed 42):
   - 95% CI for each model's accuracy,
   - 95% CI for each pairwise accuracy difference (paired resampling).

Outputs (Classic/results/):
  significance_mcnemar.csv       — all 21 pairs: b, c, chi2, p_chi2, p_exact, significant
  significance_bootstrap.csv     — per-model accuracy 95% CI
  significance_bootstrap_diff.csv — all 21 pairwise difference 95% CIs
  significance_summary.md        — human-readable summary

Stdlib only; deterministic (seed 42).
"""

import csv
import json
import math
import random
from itertools import combinations
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
RESULTS_DIR = Path("Classic/results")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42
ALPHA = 0.05

# Display name -> prediction column (order defines "Model A" in each pair)
MODELS = {
    "LogReg": "logreg_pred",
    "Linear SVM": "svm_pred",
    "XGBoost": "xgb_pred",
    "Qwen3.5-4B (FT)": "qwen_ft_label",
    "Gemma3-4B (FT)": "gemma_ft_label",
    "Qwen3.5-4B (ZS)": "qwen_zs_label",
    "Gemma3-4B (ZS)": "gemma_zs_label",
}

# The six key pairs reported in the paper (A is the higher-accuracy model)
KEY_PAIRS = [
    ("Qwen3.5-4B (ZS)", "Linear SVM"),
    ("Qwen3.5-4B (ZS)", "Gemma3-4B (FT)"),
    ("Qwen3.5-4B (ZS)", "Qwen3.5-4B (FT)"),
    ("Qwen3.5-4B (ZS)", "Gemma3-4B (ZS)"),
    ("Gemma3-4B (FT)", "Linear SVM"),
    ("Qwen3.5-4B (FT)", "Linear SVM"),
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_rows():
    with open(TEST_CSV, newline="") as f:
        return list(csv.DictReader(f))


def correctness_vectors(rows):
    """Map model display name -> list of booleans (correct per row)."""
    n = len(rows)
    vecs = {}
    for name, col in MODELS.items():
        vecs[name] = [rows[i][col] == rows[i]["label"] for i in range(n)]
    return vecs


# ── McNemar ──────────────────────────────────────────────────────────────────
def mcnemar(vec_a, vec_b):
    """Paired McNemar test between models A and B.

    b = #rows A correct & B wrong; c = #rows A wrong & B correct.
    Returns (b, c, chi2_cc, p_chi2, p_exact).
    """
    b = sum(1 for a, w in zip(vec_a, vec_b) if a and not w)
    c = sum(1 for a, w in zip(vec_a, vec_b) if not a and w)
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0, 1.0
    # chi-squared with continuity correction (1 dof)
    chi2 = (abs(b - c) - 1) ** 2 / n
    # exact survival function for chi2(1): erfc(sqrt(x/2))
    p_chi2 = math.erfc(math.sqrt(chi2 / 2.0))
    # exact two-sided binomial p-value
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_exact = min(1.0, 2.0 * tail)
    return b, c, chi2, p_chi2, p_exact


# ── Bootstrap ───────────────────────────────────────────────────────────────
def bootstrap_cis(vecs, n_rows):
    rng = random.Random(BOOTSTRAP_SEED)
    names = list(vecs)
    acc_samples = {m: [] for m in names}
    diff_samples = {pair: [] for pair in combinations(names, 2)}
    for _ in range(N_BOOTSTRAP):
        idx = [rng.randrange(n_rows) for _ in range(n_rows)]
        accs = {}
        for m in names:
            v = vecs[m]
            accs[m] = sum(v[i] for i in idx) / n_rows
            acc_samples[m].append(accs[m])
        for a, b in combinations(names, 2):
            diff_samples[(a, b)].append(accs[a] - accs[b])
    def ci(vals):
        vals = sorted(vals)
        lo = vals[int(0.025 * N_BOOTSTRAP)]
        hi = vals[int(0.975 * N_BOOTSTRAP) - 1]
        return lo, hi
    model_cis = {m: ci(acc_samples[m]) for m in names}
    diff_cis = {}
    for pair, vals in diff_samples.items():
        lo, hi = ci(vals)
        diff_cis[pair] = (lo, hi)
    return model_cis, diff_cis


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    rows = load_rows()
    n = len(rows)
    vecs = correctness_vectors(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Accuracies
    acc = {m: sum(v) / n for m, v in vecs.items()}

    # All 21 pairs
    mcnemar_rows = []
    for a, b in combinations(MODELS, 2):
        va, vb = vecs[a], vecs[b]
        bb, cc, chi2, p_chi2, p_exact = mcnemar(va, vb)
        mcnemar_rows.append({
            "model_a": a, "model_b": b,
            "acc_a": acc[a], "acc_b": acc[b],
            "b": bb, "c": cc, "chi2_cc": chi2,
            "p_chi2": p_chi2, "p_exact": p_exact,
            "significant_005": p_exact < ALPHA,
        })

    # Bootstrap
    model_cis, diff_cis = bootstrap_cis(vecs, n)

    # ── Write CSVs ───────────────────────────────────────────────────────
    with open(RESULTS_DIR / "significance_mcnemar.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mcnemar_rows[0]))
        w.writeheader()
        w.writerows(mcnemar_rows)

    with open(RESULTS_DIR / "significance_bootstrap.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "ci95_lo", "ci95_hi"])
        for m in MODELS:
            lo, hi = model_cis[m]
            w.writerow([m, f"{acc[m]:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    with open(RESULTS_DIR / "significance_bootstrap_diff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_a", "model_b", "diff_pp", "ci95_lo_pp", "ci95_hi_pp"])
        for (a, b), (lo, hi) in diff_cis.items():
            d = (acc[a] - acc[b]) * 100
            w.writerow([a, b, f"{d:.2f}", f"{lo*100:.2f}", f"{hi*100:.2f}"])

    # ── Console summary ──────────────────────────────────────────────────
    print(f"n = {n} test rows | bootstrap: {N_BOOTSTRAP} resamples, seed {BOOTSTRAP_SEED}\n")
    print("Per-model accuracy 95% bootstrap CIs:")
    for m in MODELS:
        lo, hi = model_cis[m]
        print(f"  {m:18s} {acc[m]*100:6.2f}%  [{lo*100:5.2f}, {hi*100:5.2f}]")

    def find_pair(a, b):
        """Order-agnostic pair lookup."""
        for r in mcnemar_rows:
            if {r["model_a"], r["model_b"]} == {a, b}:
                # orient so model_a is the requested `a`
                if r["model_a"] != a:
                    r = dict(r, model_a=r["model_b"], model_b=r["model_a"],
                             b=r["c"], c=r["b"])
                return r
        raise KeyError((a, b))

    def find_diff_ci(a, b):
        if (a, b) in diff_cis:
            return diff_cis[(a, b)]
        lo, hi = diff_cis[(b, a)]
        return -hi, -lo  # negate and swap when reversing the pair

    print("\nKey pairs (McNemar):")
    for a, b in KEY_PAIRS:
        r = find_pair(a, b)
        (lo, hi) = find_diff_ci(a, b)
        sig = "SIG" if r["p_exact"] < ALPHA else "ns"
        print(f"  {a} vs {b}: b={r['b']} c={r['c']} chi2={r['chi2_cc']:.2f} "
              f"p_chi2={r['p_chi2']:.4f} p_exact={r['p_exact']:.4f} [{sig}] "
              f"Δ={(acc[a]-acc[b])*100:+.2f}pp CI[{lo*100:+.2f},{hi*100:+.2f}]")
    print("\nAll significant pairs (p_exact < 0.05):")
    for r in mcnemar_rows:
        if r["significant_005"]:
            print(f"  {r['model_a']} vs {r['model_b']}: p_exact={r['p_exact']:.4f}")

    # Machine-readable summary
    summary = {
        "n_rows": n,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
        "model_accuracy_ci": {m: [acc[m], *model_cis[m]] for m in MODELS},
        "key_pairs": [
            {
                "a": a, "b": b,
                **{k: find_pair(a, b)[k] for k in ("b", "c", "chi2_cc", "p_exact")},
                "diff_pp": (acc[a] - acc[b]) * 100,
                "diff_ci95_pp": [find_diff_ci(a, b)[0] * 100, find_diff_ci(a, b)[1] * 100],
            }
            for a, b in KEY_PAIRS
        ],
    }
    with open(RESULTS_DIR / "significance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWritten: {RESULTS_DIR}/significance_mcnemar.csv, "
          f"significance_bootstrap.csv, significance_bootstrap_diff.csv, "
          f"significance_summary.json")


if __name__ == "__main__":
    main()