"""
generate_figures.py
-------------------
Produces all high-resolution figures (300 DPI) for the Financial Sentiment paper.

Figures saved to: Figures/
  fig1_accuracy_comparison.pdf/.png
  fig2_macro_f1_comparison.pdf/.png
  fig3_per_class_f1_heatmap.pdf/.png
  fig4_confusion_matrices.pdf/.png
  fig5_cv_boxplots.pdf/.png
  fig6_inference_time.pdf/.png
  fig7_per_class_bars.pdf/.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ── paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
FIGS   = os.path.join(BASE, "Figures")
RES    = os.path.join(BASE, "Classic", "results")
DATA   = os.path.join(BASE, "Datasets", "financial_phrasebank", "test.csv")
os.makedirs(FIGS, exist_ok=True)

DPI        = 300
FIG_EXT    = ["pdf", "png"]
CLASSES    = ["negative", "neutral", "positive"]
PALETTE    = "#2b6cb0"          # main blue
PALETTE2   = "#c05621"          # accent orange

# Model display names (ordered for all comparison charts)
MODEL_ORDER = [
    "logreg", "svm", "xgboost",
    "qwen_ft", "gemma_ft",
    "qwen_zs", "gemma_zs",
]
MODEL_LABELS = {
    "logreg":   "LogReg",
    "svm":      "Linear SVM",
    "xgboost":  "XGBoost",
    "qwen_ft":  "Qwen3.5-4B\n(fine-tuned)",
    "gemma_ft": "Gemma3-4B\n(fine-tuned)",
    "qwen_zs":  "Qwen3.5-4B\n(zero-shot)",
    "gemma_zs": "Gemma3-4B\n(zero-shot)",
}
MODEL_COLORS = {
    "logreg":   "#4299e1",
    "svm":      "#2b6cb0",
    "xgboost":  "#2c7a7b",
    "qwen_ft":  "#d69e2e",
    "gemma_ft": "#c05621",
    "qwen_zs":  "#805ad5",
    "gemma_zs": "#b794f4",
}

# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
y_true = df["label"].astype(str)

PRED_COLS = {
    "logreg":   "logreg_pred",
    "svm":      "svm_pred",
    "xgboost":  "xgb_pred",
    "qwen_ft":  "qwen_ft_label",
    "gemma_ft": "gemma_ft_label",
    "qwen_zs":  "qwen_zs_label",
    "gemma_zs": "gemma_zs_label",
}

# Build per-model classification reports
reports = {}
for mid, col in PRED_COLS.items():
    y_pred = df[col].astype(str)
    reports[mid] = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )

cv_summary = pd.read_csv(os.path.join(RES, "cv_summary_metrics.csv"))
cv_folds   = pd.read_csv(os.path.join(RES, "cv_folds_metrics.csv"))
train_time = pd.read_csv(os.path.join(RES, "train_time_per_model.csv"))

# Inference time per sample (seconds)
INFER_TIME_MS = {
    "logreg":   df["logreg_time_sec"].mean() * 1000,
    "svm":      df["svm_time_sec"].mean() * 1000,
    "xgboost":  df["xgb_time_sec"].mean() * 1000,
    "qwen_ft":  df["qwen_ft_time_sec"].mean() * 1000,
    "gemma_ft": df["gemma_ft_time_sec"].mean() * 1000,
    "qwen_zs":  df["qwen_zs_time_sec"].mean() * 1000,
    "gemma_zs": df["gemma_zs_time_sec"].mean() * 1000,
}


def savefig(fig, name):
    for ext in FIG_EXT:
        path = os.path.join(FIGS, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  saved {name}")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 — Overall accuracy comparison (all 7 models)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating fig1_accuracy_comparison …")
acc = {m: reports[m]["accuracy"] for m in MODEL_ORDER}

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    range(len(MODEL_ORDER)),
    [acc[m] for m in MODEL_ORDER],
    color=[MODEL_COLORS[m] for m in MODEL_ORDER],
    width=0.6, edgecolor="white", linewidth=0.8, zorder=3,
)
ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Test Accuracy — All Models", fontsize=13, fontweight="bold")
ax.set_ylim(0.84, 0.97)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))
ax.axhline(0.92, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top","right"]].set_visible(False)

for bar, m in zip(bars, MODEL_ORDER):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{acc[m]*100:.2f}%",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
    )

# Separator between classic / fine-tuned / zero-shot groups
for x in [2.5, 4.5]:
    ax.axvline(x, color="lightgray", linewidth=1.2, linestyle=":", zorder=1)

legend_patches = [
    Patch(color="#4299e1", label="Classic"),
    Patch(color="#d69e2e", label="Fine-tuned LLM"),
    Patch(color="#805ad5", label="Zero-shot LLM"),
]
ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9, loc="lower right")
plt.tight_layout()
savefig(fig, "fig1_accuracy_comparison")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 — Macro F1 comparison (all 7 models)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating fig2_macro_f1_comparison …")
mf1 = {m: reports[m]["macro avg"]["f1-score"] for m in MODEL_ORDER}

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    range(len(MODEL_ORDER)),
    [mf1[m] for m in MODEL_ORDER],
    color=[MODEL_COLORS[m] for m in MODEL_ORDER],
    width=0.6, edgecolor="white", linewidth=0.8, zorder=3,
)
ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
ax.set_ylabel("Macro F1-Score", fontsize=12)
ax.set_title("Test Macro F1-Score — All Models", fontsize=13, fontweight="bold")
ax.set_ylim(0.83, 0.96)
ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
ax.spines[["top","right"]].set_visible(False)

for bar, m in zip(bars, MODEL_ORDER):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{mf1[m]:.4f}",
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
    )

for x in [2.5, 4.5]:
    ax.axvline(x, color="lightgray", linewidth=1.2, linestyle=":", zorder=1)

ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9, loc="lower right")
plt.tight_layout()
savefig(fig, "fig2_macro_f1_comparison")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MERGED FIG — (a) accuracy + (b) macro F1 side-by-side panel
# (referenced by main.tex as \includegraphics{merged-fig})
# ═══════════════════════════════════════════════════════════════════════════
print("Generating merged-fig …")
fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

for ax, metric, label, ylim, fmt, title in (
    (axes[0], acc, "Test Accuracy (%)", (0.84, 0.97), lambda m: f"{acc[m]*100:.2f}%",
     "Test Accuracy — All Models"),
    (axes[1], mf1, "Macro F1-Score", (0.83, 0.96), lambda m: f"{mf1[m]:.4f}",
     "Test Macro F1-Score — All Models"),
):
    bars = ax.bar(
        range(len(MODEL_ORDER)),
        [metric[m] for m in MODEL_ORDER],
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        width=0.6, edgecolor="white", linewidth=0.8, zorder=3,
    )
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, m in zip(bars, MODEL_ORDER):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ylim[1] - ylim[0]) * 0.008,
            fmt(m),
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
        )
    for x in [2.5, 4.5]:
        ax.axvline(x, color="lightgray", linewidth=1.2, linestyle=":", zorder=1)
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9, loc="lower right")

fig.suptitle("Model Family Comparison — Accuracy and Macro F1", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
savefig(fig, "merged-fig")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — Per-class F1 heatmap (models × classes)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating fig3_per_class_f1_heatmap …")
hmap_data = pd.DataFrame(
    {m: {c: reports[m][c]["f1-score"] for c in CLASSES} for m in MODEL_ORDER}
).T
hmap_data.index = [MODEL_LABELS[m].replace("\n", " ") for m in MODEL_ORDER]
hmap_data.columns = ["Negative", "Neutral", "Positive"]

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    hmap_data,
    annot=True, fmt=".4f", cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    vmin=0.84, vmax=1.00,
    annot_kws={"size": 10, "weight": "bold"},
    ax=ax, cbar_kws={"label": "F1-Score"},
)
ax.set_title("Per-Class F1-Score — All Models", fontsize=13, fontweight="bold")
ax.set_xlabel("Sentiment Class", fontsize=11)
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=10, rotation=0)
plt.tight_layout()
savefig(fig, "fig3_per_class_f1_heatmap")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — Confusion matrices (all 7 models in a 2×4 grid)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating fig4_confusion_matrices …")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, m in enumerate(MODEL_ORDER):
    ax = axes[i]
    y_pred = df[PRED_COLS[m]].astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm,
        annot=np.array([[f"{v:.2f}\n({c})" for v, c in zip(row_n, row_c)]
                        for row_n, row_c in zip(cm_norm, cm)]),
        fmt="",
        cmap="Blues",
        linewidths=0.5, linecolor="white",
        vmin=0, vmax=1,
        annot_kws={"size": 9, "weight": "bold"},
        xticklabels=["Neg", "Neu", "Pos"],
        yticklabels=["Neg", "Neu", "Pos"],
        cbar=False,
        ax=ax,
    )
    acc_val = reports[m]["accuracy"]
    ax.set_title(f'{MODEL_LABELS[m].replace(chr(10), " ")}\nAcc={acc_val*100:.2f}%',
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.tick_params(labelsize=9)

# Hide the unused 8th subplot
axes[7].set_visible(False)

fig.suptitle("Confusion Matrices (row-normalised fractions; raw counts in parentheses) — All Models",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "fig4_confusion_matrices")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 — CV box plots (classic models, fold-level accuracy & macro F1)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating fig5_cv_boxplots …")
cv_model_labels = {"logreg": "LogReg", "svm": "Linear SVM", "xgboost": "XGBoost"}

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, metric, ylabel in zip(
    axes,
    ["accuracy", "macro_f1"],
    ["Accuracy", "Macro F1-Score"],
):
    data_by_model = [
        cv_folds[cv_folds["model"] == m][metric].values
        for m in ["logreg", "svm", "xgboost"]
    ]
    # `labels` was renamed `tick_labels` in matplotlib 3.11; keep both working
    box_kwargs = dict(
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.6},
        widths=0.45,
    )
    try:
        bplot = ax.boxplot(data_by_model,
                           tick_labels=[cv_model_labels[m] for m in ["logreg", "svm", "xgboost"]],
                           **box_kwargs)
    except TypeError:
        bplot = ax.boxplot(data_by_model,
                           labels=[cv_model_labels[m] for m in ["logreg", "svm", "xgboost"]],
                           **box_kwargs)
    colors = ["#4299e1", "#2b6cb0", "#2c7a7b"]
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual points
    for i, data in enumerate(data_by_model, start=1):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(data))
        ax.scatter(i + jitter, data, color="black", s=25, zorder=5, alpha=0.8)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"5-Fold CV {ylabel}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    if metric == "accuracy":
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))

fig.suptitle("Cross-Validation Performance (Classic Models)", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig(fig, "fig5_cv_boxplots")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 — Inference time (log scale, seconds per sample)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating fig6_inference_time …")
fig, ax = plt.subplots(figsize=(9, 5))
times_ms = [INFER_TIME_MS[m] for m in MODEL_ORDER]
times_s = [t / 1000.0 for t in times_ms]   # seconds — matches Table values

bars = ax.bar(
    range(len(MODEL_ORDER)),
    times_s,
    color=[MODEL_COLORS[m] for m in MODEL_ORDER],
    width=0.6, edgecolor="white", linewidth=0.8, zorder=3,
)
ax.set_yscale("log")
ax.set_xticks(range(len(MODEL_ORDER)))
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
ax.set_ylabel("Inference time per sample (s, log scale)", fontsize=11)
ax.set_title("Inference Latency — All Models", fontsize=13, fontweight="bold")
ax.grid(axis="y", linewidth=0.5, alpha=0.4, which="both", zorder=0)
ax.spines[["top","right"]].set_visible(False)

for bar, t_s in zip(bars, times_s):
    if t_s < 0.001:
        label = f"{t_s*1000:.3f} ms"
    elif t_s < 1:
        label = f"{t_s*1000:.1f} ms"
    else:
        label = f"{t_s:.2f} s"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.15,
        label,
        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
    )

for x in [2.5, 4.5]:
    ax.axvline(x, color="lightgray", linewidth=1.2, linestyle=":", zorder=1)

ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9, loc="upper left")
plt.tight_layout()
savefig(fig, "fig6_inference_time")
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7 — Per-class precision / recall / F1 grouped bar chart (all models)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating fig7_per_class_bars …")
metrics_list = ["precision", "recall", "f1-score"]
class_colors = {"negative": "#fc8181", "neutral": "#68d391", "positive": "#63b3ed"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, cls in zip(axes, CLASSES):
    x = np.arange(len(MODEL_ORDER))
    width = 0.25
    offsets = [-width, 0, width]

    for offset, metric, hatch in zip(offsets, metrics_list, ["", "//", "xx"]):
        vals = [reports[m][cls][metric] for m in MODEL_ORDER]
        ax.bar(
            x + offset, vals, width=width,
            label=metric.replace("f1-score", "F1"),
            color=class_colors[cls],
            alpha=0.55 + offsets.index(offset) * 0.2,
            hatch=hatch, edgecolor="white", linewidth=0.6, zorder=3,
        )

    ax.set_title(f"{cls.capitalize()} Class", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[m].replace("\n", "\n") for m in MODEL_ORDER],
        fontsize=8.5,
    )
    ax.set_ylim(0.72, 1.02)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    if ax == axes[0]:
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)

fig.suptitle("Per-Class Precision / Recall / F1 — All Models",
             fontsize=13, fontweight="bold")
plt.tight_layout()
savefig(fig, "fig7_per_class_bars")
plt.close()


print(f"\nAll figures saved to: {FIGS}/")
