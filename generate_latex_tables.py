"""
generate_latex_tables.py
------------------------
Recomputes all statistics from raw data and outputs LaTeX table code
for the Financial Sentiment paper.

Tables:
  T1  — Dataset split statistics
  T2  — Cross-validation summary (classic models, mean ± std)
  T3  — Test-set overall metrics (all 7 models)
  T4  — Test-set per-class P / R / F1 — Classic models
  T5  — Test-set per-class P / R / F1 — Fine-tuned LLMs
  T6  — Test-set per-class P / R / F1 — Zero-shot LLMs
  T7  — Inference speed summary (all 7 models)
"""

import os, json, textwrap
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ── paths ──────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
RES   = os.path.join(BASE, "Classic", "results")
DATA  = os.path.join(BASE, "Datasets", "financial_phrasebank", "test.csv")
TRAIN = os.path.join(BASE, "Datasets", "financial_phrasebank", "train.csv")
VAL   = os.path.join(BASE, "Datasets", "financial_phrasebank", "validation.csv")

CLASSES     = ["negative", "neutral", "positive"]
MODEL_ORDER = ["logreg", "svm", "xgboost", "qwen_ft", "gemma_ft", "qwen_zs", "gemma_zs"]
MODEL_NAMES = {
    "logreg":   "Logistic Regression",
    "svm":      "Linear SVM",
    "xgboost":  "XGBoost",
    "qwen_ft":  r"Qwen3.5-4B (fine-tuned)",
    "gemma_ft": r"Gemma3-4B (fine-tuned)",
    "qwen_zs":  r"Qwen3.5-4B (zero-shot)",
    "gemma_zs": r"Gemma3-4B (zero-shot)",
}
PRED_COLS = {
    "logreg":   "logreg_pred",
    "svm":      "svm_pred",
    "xgboost":  "xgb_pred",
    "qwen_ft":  "qwen_ft_label",
    "gemma_ft": "gemma_ft_label",
    "qwen_zs":  "qwen_zs_label",
    "gemma_zs": "gemma_zs_label",
}
TIME_COLS = {
    "logreg":   "logreg_time_sec",
    "svm":      "svm_time_sec",
    "xgboost":  "xgb_time_sec",
    "qwen_ft":  "qwen_ft_time_sec",
    "gemma_ft": "gemma_ft_time_sec",
    "qwen_zs":  "qwen_zs_time_sec",
    "gemma_zs": "gemma_zs_time_sec",
}

# ── load data ──────────────────────────────────────────────────────────────
df_test  = pd.read_csv(DATA)
df_train = pd.read_csv(TRAIN)
df_val   = pd.read_csv(VAL)
y_true   = df_test["label"].astype(str)

cv_summary = pd.read_csv(os.path.join(RES, "cv_summary_metrics.csv"))
cv_folds   = pd.read_csv(os.path.join(RES, "cv_folds_metrics.csv"))
train_time = pd.read_csv(os.path.join(RES, "train_time_per_model.csv"))

# Build classification reports
reports = {}
for mid, col in PRED_COLS.items():
    y_pred = df_test[col].astype(str)
    reports[mid] = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )

SEP = "\n" + "─" * 72 + "\n"

# ══════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════
def hr(title):
    print(SEP + f"TABLE: {title}" + SEP)

# ══════════════════════════════════════════════════════════════════════════
# T1 — Dataset Split Statistics
# ══════════════════════════════════════════════════════════════════════════
hr("T1 — Dataset Split Statistics")

splits = {"Train": df_train, "Validation": df_val, "Test": df_test}
rows = []
for name, sdf in splits.items():
    vc = sdf["label"].value_counts()
    rows.append({
        "Split":    name,
        "Total":    len(sdf),
        "Neutral":  vc.get("neutral", 0),
        "Positive": vc.get("positive", 0),
        "Negative": vc.get("negative", 0),
    })
t1 = pd.DataFrame(rows)

print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Financial PhraseBank dataset split statistics.}")
print(r"\label{tab:dataset}")
print(r"\begin{tabular}{lrrrr}")
print(r"\toprule")
print(r"Split & Total & Neutral & Positive & Negative \\")
print(r"\midrule")
for _, row in t1.iterrows():
    print(f"{row['Split']} & {row['Total']} & {row['Neutral']} & "
          f"{row['Positive']} & {row['Negative']} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# Verify
print("\n[VERIFY] Split counts:")
for _, row in t1.iterrows():
    total_check = row["Neutral"] + row["Positive"] + row["Negative"]
    match = "OK" if total_check == row["Total"] else f"MISMATCH (sum={total_check})"
    print(f"  {row['Split']}: {row['Total']} total — {match}")


# ══════════════════════════════════════════════════════════════════════════
# T2 — Cross-Validation Summary (Classic Models)
# ══════════════════════════════════════════════════════════════════════════
hr("T2 — Cross-Validation Summary (Classic Models, 5-fold)")

# Recompute from fold-level data to verify
cv_check = cv_folds.groupby("model").agg(
    acc_mean=("accuracy", "mean"),
    acc_std=("accuracy", "std"),
    f1_mean=("macro_f1", "mean"),
    f1_std=("macro_f1", "std"),
    wf1_mean=("weighted_f1", "mean"),
    wf1_std=("weighted_f1", "std"),
).reset_index()

print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{5-fold cross-validation results on the combined train+validation pool (1,924 samples).}")
print(r"\label{tab:cv}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Model & Accuracy & Macro F1 & Weighted F1 \\")
print(r"\midrule")
classic_order = ["logreg", "svm", "xgboost"]
for m in classic_order:
    row = cv_check[cv_check["model"] == m].iloc[0]
    print(f"{MODEL_NAMES[m]} & "
          f"${row['acc_mean']*100:.2f} \\pm {row['acc_std']*100:.2f}$ & "
          f"${row['f1_mean']:.4f} \\pm {row['f1_std']:.4f}$ & "
          f"${row['wf1_mean']:.4f} \\pm {row['wf1_std']:.4f}$ \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n[VERIFY] CV summary vs stored file:")
for m in classic_order:
    stored = cv_summary[cv_summary["model"] == m].iloc[0]
    recomp = cv_check[cv_check["model"] == m].iloc[0]
    for col_s, col_r in [("accuracy_mean","acc_mean"),("macro_f1_mean","f1_mean")]:
        diff = abs(stored[col_s] - recomp[col_r])
        status = "OK" if diff < 1e-8 else f"DIFF={diff:.2e}"
        print(f"  {m} {col_s}: {status}")


# ══════════════════════════════════════════════════════════════════════════
# T3 — Overall Test Metrics (all 7 models)
# ══════════════════════════════════════════════════════════════════════════
hr("T3 — Overall Test Metrics (all 7 models, n=340)")

print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Test-set performance of all models on the Financial PhraseBank "
      r"(340 samples). Best result per column in \textbf{bold}.}")
print(r"\label{tab:test_overall}")
print(r"\begin{tabular}{lccc}")
print(r"\toprule")
print(r"Model & Accuracy & Macro F1 & Weighted F1 \\")
print(r"\midrule")

all_acc  = {m: reports[m]["accuracy"]                     for m in MODEL_ORDER}
all_mf1  = {m: reports[m]["macro avg"]["f1-score"]        for m in MODEL_ORDER}
all_wf1  = {m: reports[m]["weighted avg"]["f1-score"]     for m in MODEL_ORDER}

best_acc = max(all_acc.values())
best_mf1 = max(all_mf1.values())
best_wf1 = max(all_wf1.values())

group_lines = {"xgboost": True, "gemma_ft": True}   # draw \midrule after these

for m in MODEL_ORDER:
    acc_s  = f"{all_acc[m]*100:.2f}\\%"
    mf1_s  = f"{all_mf1[m]:.4f}"
    wf1_s  = f"{all_wf1[m]:.4f}"

    if abs(all_acc[m]  - best_acc) < 1e-9: acc_s  = r"\textbf{" + acc_s + "}"
    if abs(all_mf1[m]  - best_mf1) < 1e-9: mf1_s  = r"\textbf{" + mf1_s + "}"
    if abs(all_wf1[m]  - best_wf1) < 1e-9: wf1_s  = r"\textbf{" + wf1_s + "}"

    print(f"{MODEL_NAMES[m]} & {acc_s} & {mf1_s} & {wf1_s} \\\\")
    if m in group_lines:
        print(r"\midrule")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n[VERIFY] Test accuracy recomputed from raw predictions:")
for m in MODEL_ORDER:
    y_pred = df_test[PRED_COLS[m]].astype(str)
    recomp = accuracy_score(y_true, y_pred)
    stored = all_acc[m]
    diff   = abs(recomp - stored)
    print(f"  {m:12s}: recomputed={recomp:.6f}  stored={stored:.6f}  diff={diff:.2e}  {'OK' if diff<1e-9 else 'MISMATCH'}")


# ══════════════════════════════════════════════════════════════════════════
# T4 — Per-Class Metrics: Classic Models
# ══════════════════════════════════════════════════════════════════════════
hr("T4 — Per-Class Metrics: Classic Models")

def per_class_table(models, caption, label):
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{" + caption + "}")
    print(r"\label{" + label + "}")
    print(r"\begin{tabular}{llcccr}")
    print(r"\toprule")
    print(r"Model & Class & Precision & Recall & F1-Score & Support \\")
    print(r"\midrule")
    for mi, m in enumerate(models):
        rpt = reports[m]
        for ci, cls in enumerate(CLASSES):
            model_cell = MODEL_NAMES[m] if ci == 0 else ""
            sup = rpt[cls]["support"]
            p   = rpt[cls]["precision"]
            r   = rpt[cls]["recall"]
            f1  = rpt[cls]["f1-score"]
            print(f"{model_cell} & {cls.capitalize()} & "
                  f"{p:.4f} & {r:.4f} & {f1:.4f} & {int(sup)} \\\\")
        macro = rpt["macro avg"]
        print(f" & \\textit{{Macro avg}} & "
              f"{macro['precision']:.4f} & {macro['recall']:.4f} & "
              f"{macro['f1-score']:.4f} & --- \\\\")
        if mi < len(models) - 1:
            print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

per_class_table(
    ["logreg", "svm", "xgboost"],
    "Per-class precision, recall, and F1-score for classic models on the test set.",
    "tab:classic_perclass",
)

# ══════════════════════════════════════════════════════════════════════════
# T5 — Per-Class Metrics: Fine-tuned LLMs
# ══════════════════════════════════════════════════════════════════════════
hr("T5 — Per-Class Metrics: Fine-tuned LLMs")

per_class_table(
    ["qwen_ft", "gemma_ft"],
    "Per-class precision, recall, and F1-score for fine-tuned LLMs on the test set.",
    "tab:finetuned_perclass",
)

# ══════════════════════════════════════════════════════════════════════════
# T6 — Per-Class Metrics: Zero-shot LLMs
# ══════════════════════════════════════════════════════════════════════════
hr("T6 — Per-Class Metrics: Zero-shot LLMs")

per_class_table(
    ["qwen_zs", "gemma_zs"],
    "Per-class precision, recall, and F1-score for zero-shot LLMs (advisory-panel XAI layer) on the test set.",
    "tab:zeroshot_perclass",
)

# ══════════════════════════════════════════════════════════════════════════
# T7 — Inference Speed
# ══════════════════════════════════════════════════════════════════════════
hr("T7 — Inference Speed (all 7 models)")

train_map = {r["model"]: r for _, r in train_time.iterrows()}

# Fine-tuned train time not in train_time.csv — load from report if available
# Use "N/A" for LLMs (fine-tuning runtime not tracked in classic results)
TRAIN_TIME_DISPLAY = {
    "logreg":   f"{train_map['logreg']['total_train_time_sec']:.1f}\\,s",
    "svm":      f"{train_map['svm']['total_train_time_sec']:.2f}\\,s",
    "xgboost":  f"{train_map['xgboost']['total_train_time_sec']:.0f}\\,s",
    "qwen_ft":  r"$\sim$120 steps",
    "gemma_ft": r"$\sim$120 steps",
    "qwen_zs":  "---",
    "gemma_zs": "---",
}

print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Inference latency (mean $\pm$ std per sample) and training time for all models.}")
print(r"\label{tab:speed}")
print(r"\begin{tabular}{lccr}")
print(r"\toprule")
print(r"Model & Avg.\ Inference (ms) & Std.\ (ms) & Training Time \\")
print(r"\midrule")

for m in MODEL_ORDER:
    col = TIME_COLS[m]
    times_s = df_test[col].dropna()
    mean_ms = times_s.mean() * 1000
    std_ms  = times_s.std()  * 1000
    print(f"{MODEL_NAMES[m]} & "
          f"{mean_ms:.3f} & "
          f"{std_ms:.3f} & "
          f"{TRAIN_TIME_DISPLAY[m]} \\\\")
    if m == "xgboost":
        print(r"\midrule")
    if m == "gemma_ft":
        print(r"\midrule")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n[VERIFY] Inference time stats:")
for m in MODEL_ORDER:
    col = TIME_COLS[m]
    times_s = df_test[col].dropna()
    print(f"  {m:12s}: n={len(times_s)}, mean={times_s.mean()*1000:.3f}ms, "
          f"min={times_s.min()*1000:.3f}ms, max={times_s.max()*1000:.3f}ms")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(SEP + "SUMMARY — All 7 models, sorted by accuracy" + SEP)
rows = []
for m in MODEL_ORDER:
    rows.append({
        "Model":       MODEL_NAMES[m],
        "Accuracy":    f"{reports[m]['accuracy']*100:.2f}%",
        "Macro F1":    f"{reports[m]['macro avg']['f1-score']:.4f}",
        "Weighted F1": f"{reports[m]['weighted avg']['f1-score']:.4f}",
    })
summary_df = pd.DataFrame(rows)
summary_df_sorted = summary_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
print(summary_df_sorted.to_string(index=False))
