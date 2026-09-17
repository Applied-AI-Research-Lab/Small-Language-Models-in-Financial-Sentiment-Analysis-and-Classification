"""B4 — Calibration diagnostics of the panel signals (R1.3 diagnostic half).

From the existing per-row probability columns in
Datasets/financial_phrasebank/test.csv (340 rows), computes for each classical
panel member (LogReg, SVM softmax, XGBoost):

- Multiclass Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
  (0 = perfect; 2.0 = worst possible for 3 classes)
- Expected Calibration Error (ECE), confidence-based, 10 equal-width bins:
  ECE = sum_b (n_b / N) * |acc_b - conf_b|, where conf_i = max_k p_ik
- Mean confidence and accuracy
- Reliability-curve data (per bin: mean confidence, accuracy, count)

Also renders a reliability diagram (Paper/Figures/fig_calibration_reliability)
if matplotlib is available.

Note: LogReg and SVM were trained with balanced class weights, which distorts
posterior probabilities toward minority classes; SVM 'probabilities' are a
softmax over decision scores; XGBoost uses multi:softprob. None of the three
is verified calibrated --- this script quantifies that.

Outputs (Classic/results/):
  calibration_metrics.json / .csv
  calibration_reliability_bins.csv
"""

import csv
import json
from pathlib import Path

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
OUT_DIR = Path("Classic/results")
FIG_DIR = Path("Paper/Figures")

CLASSES = ("negative", "neutral", "positive")
MODELS = {
    "LogReg": "logreg",
    "Linear SVM": "svm",
    "XGBoost": "xgb",
}
N_BINS = 10


def brier_multiclass(probs, labels):
    """probs: list of [p_neg, p_neu, p_pos]; labels: class strings."""
    total = 0.0
    for p, y in zip(probs, labels):
        for k, cls in enumerate(CLASSES):
            yk = 1.0 if cls == y else 0.0
            total += (p[k] - yk) ** 2
    return total / len(probs)


def ece_confidence(probs, labels, n_bins=N_BINS):
    """Confidence-based ECE with equal-width bins over [0, 1]."""
    n = len(probs)
    bins = [[] for _ in range(n_bins)]  # (conf, correct)
    for p, y in zip(probs, labels):
        conf = max(p)
        pred = CLASSES[p.index(conf)]
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, 1.0 if pred == y else 0.0))
    ece = 0.0
    curve = []
    for b, items in enumerate(bins):
        if not items:
            curve.append((b, None, None, 0))
            continue
        confs = [c for c, _ in items]
        accs = [a for _, a in items]
        mc = sum(confs) / len(confs)
        ma = sum(accs) / len(accs)
        ece += len(items) / n * abs(ma - mc)
        curve.append((b, mc, ma, len(items)))
    return ece, curve


def main():
    rows = list(csv.DictReader(open(TEST_CSV)))
    n = len(rows)
    labels = [r["label"] for r in rows]

    results = {}
    for name, prefix in MODELS.items():
        probs = [[float(r[f"{prefix}_prob_{c}"]) for c in CLASSES] for r in rows]
        # sanity: probabilities sum to 1
        for p in probs:
            assert abs(sum(p) - 1.0) < 1e-6, f"{name}: probs do not sum to 1"
        confs = [max(p) for p in probs]
        preds = [CLASSES[p.index(c)] for p, c in zip(probs, confs)]
        acc = sum(1 for p, y in zip(preds, labels) if p == y) / n
        brier = brier_multiclass(probs, labels)
        ece, curve = ece_confidence(probs, labels)
        results[name] = {
            "accuracy": acc,
            "mean_confidence": sum(confs) / n,
            "confidence_gap": sum(confs) / n - acc,
            "brier_multiclass": brier,
            "ece_10bin": ece,
        }
        results[name]["_curve"] = curve

        print(f"== {name} ==")
        print(f"  accuracy          : {acc*100:.2f}%")
        print(f"  mean confidence   : {sum(confs)/n*100:.2f}%")
        print(f"  conf - acc gap    : {(sum(confs)/n - acc)*100:+.2f} pp")
        print(f"  Brier (multiclass): {brier:.4f}")
        print(f"  ECE (10-bin)      : {ece:.4f}")
        print(f"  bins (conf/acc/n) :")
        for b, mc, ma, cnt in curve:
            if cnt:
                print(f"    bin {b}: conf {mc*100:5.1f}%  acc {ma*100:5.1f}%  n={cnt}")
        print()

    # persist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "calibration_metrics.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "_curve"}
                   for k, v in results.items()}, f, indent=2)
    with open(OUT_DIR / "calibration_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "mean_confidence", "conf_minus_acc_pp",
                    "brier_multiclass", "ece_10bin"])
        for k, v in results.items():
            w.writerow([k, f"{v['accuracy']:.4f}", f"{v['mean_confidence']:.4f}",
                        f"{v['confidence_gap']*100:+.2f}",
                        f"{v['brier_multiclass']:.4f}", f"{v['ece_10bin']:.4f}"])
    with open(OUT_DIR / "calibration_reliability_bins.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "bin", "mean_confidence", "accuracy", "n"])
        for k, v in results.items():
            for b, mc, ma, cnt in v["_curve"]:
                w.writerow([k, b,
                            f"{mc:.4f}" if mc is not None else "",
                            f"{ma:.4f}" if ma is not None else "", cnt])

    # reliability figure (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        styles = {"LogReg": ("tab:blue", "o"),
                  "Linear SVM": ("tab:orange", "s"),
                  "XGBoost": ("tab:green", "^")}
        for k, v in results.items():
            xs = [mc for _, mc, ma, c in v["_curve"] if c]
            ys = [ma for _, mc, ma, c in v["_curve"] if c]
            color, marker = styles[k]
            ax.plot(xs, ys, marker=marker, linestyle="-", color=color,
                    label=f"{k} (ECE={v['ece_10bin']:.3f})", markersize=4)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
        ax.set_xlabel("Model confidence (max class probability)")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title("Reliability of panel probability signals (340 test rows)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(0.2, 1.0)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "fig_calibration_reliability.pdf")
        fig.savefig(FIG_DIR / "fig_calibration_reliability.png", dpi=200)
        print(f"Figure written: {FIG_DIR}/fig_calibration_reliability.pdf/.png")
    except Exception as exc:  # matplotlib missing etc.
        print(f"Figure not generated: {exc}")


if __name__ == "__main__":
    main()