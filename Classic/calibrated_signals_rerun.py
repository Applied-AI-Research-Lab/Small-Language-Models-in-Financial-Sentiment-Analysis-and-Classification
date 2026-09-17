#!/usr/bin/env python3
"""
C5 — Calibrated panel signals + Qwen3.5-4B zero-shot re-run (GPU server).

Motivation (Section sec:calibration): all three panel members are
under-confident; the SVM softmax scores are not calibrated probabilities.
This experiment applies leakage-free sigmoid calibration to the panel
members and re-runs the zero-shot advisory-panel evaluation with the
calibrated signals.

Pipeline
--------
1. Fit sigmoid-calibrated versions of LogReg (control), Linear SVM, and
   XGBoost on the 1,924-sentence train+validation pool using 5-fold
   cross-validated CalibratedClassifierCV (leakage-free: calibration is
   fitted on held-out folds).
2. Report test-set Brier / ECE of the calibrated probabilities vs the
   original uncalibrated signals (B4 values: LR .143/.099, SVM .204/.230,
   XGB .147/.041).
3. Write calibrated per-row panel signals to test.csv as NEW columns
   (prefix *_cal_*): pred, confidence, entropy, prob_{negative,neutral,
   positive}. Original columns are left untouched.
4. Build the advisory prompt from the CALIBRATED signals (identical prompt
   structure, "calibrated" wording preserved from the original) and re-run
   Qwen3.5-4B zero-shot over the 340 test rows with column prefix
   qwen_calzs_* (checkpointed every 10 rows, resumable).
5. McNemar (exact) of the calibrated-signal run vs the original
   qwen_zs_label run.

New test.csv columns:
  logreg_cal_pred/_confidence/_entropy/_prob_{negative,neutral,positive}
  svm_cal_*, xgb_cal_*
  qwen_calzs_label/_explanation/_recommendation/_time_sec

Usage (GPU server, from the project root):
    source ./activate_project.csh
    python Classic/calibrated_signals_rerun.py --slm      # full pipeline
    python Classic/calibrated_signals_rerun.py            # CPU part only
    python Classic/calibrated_signals_rerun.py --smoke     # 20-row SLM check
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POOL_CSVS = [Path("Datasets/financial_phrasebank/train.csv"),
             Path("Datasets/financial_phrasebank/validation.csv")]
TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
RESULTS_DIR = Path("Classic/results")
CLASSES = ("negative", "neutral", "positive")
SAVE_EVERY = 10
CAL_SEED = 42


# ── CPU: models + calibration ──────────────────────────────────────────────

def make_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                           min_df=2, max_df=0.98, sublinear_tf=True)


def make_base_models(seed=CAL_SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from xgboost import XGBClassifier
    return {
        "logreg": LogisticRegression(C=4.0, class_weight="balanced",
                                     max_iter=4000, random_state=seed,
                                     solver="lbfgs"),
        "svm": LinearSVC(C=1.0, class_weight="balanced", random_state=seed),
        "xgboost": XGBClassifier(objective="multi:softprob", n_estimators=350,
                                 max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9,
                                 n_jobs=-1, random_state=seed,
                                 eval_metric="mlogloss", verbosity=0),
    }


def entropy_of(p):
    return -sum(0 if q <= 0 else q * math.log(q) for q in p)


def fit_calibrated_models():
    """Fit per-fold-calibrated models on the pool; return fitted pipelines
    (vectorizer + calibrated classifier) that can transform test rows.

    Labels are integer-encoded (negative=0, neutral=1, positive=2) because
    XGBoost requires numeric classes; predictions are decoded back to
    strings by write_calibrated_columns.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    pool = pd.concat([pd.read_csv(p) for p in POOL_CSVS], ignore_index=True)
    pool["sentence"] = pool["sentence"].astype(str).str.strip()
    pool["label"] = pool["label"].astype(str).str.strip().str.lower()
    X_text = pool["sentence"]
    y = np.array([CLASSES.index(t) for t in pool["label"]])

    pipelines = {}
    for name, base in make_base_models().items():
        # per-fold TF-IDF to keep calibration leakage-free:
        # CalibratedClassifierCV(cv=5) internally cross-fits the base model
        # and fits the sigmoid on held-out predictions; the vectorizer is
        # rebuilt inside each fold via the Pipeline.
        pipe = Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", CalibratedClassifierCV(estimator=base, method="sigmoid",
                                           cv=StratifiedKFold(
                                               n_splits=5, shuffle=True,
                                               random_state=CAL_SEED))),
        ])
        pipe.fit(X_text, y)
        pipelines[name] = pipe
        print(f"  fitted calibrated {name} on {len(pool)} pool rows")
    return pipelines


def brier_multiclass(probs, labels):
    total = 0.0
    for p, y in zip(probs, labels):
        for k, cls in enumerate(CLASSES):
            yk = 1.0 if cls == y else 0.0
            total += (p[k] - yk) ** 2
    return total / len(probs)


def ece_confidence(probs, labels, n_bins=10):
    n = len(probs)
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        conf = max(p)
        pred = CLASSES[p.index(conf)]
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, 1.0 if pred == y else 0.0))
    ece = 0.0
    for items in bins:
        if not items:
            continue
        mc = sum(c for c, _ in items) / len(items)
        ma = sum(a for _, a in items) / len(items)
        ece += len(items) / n * abs(ma - mc)
    return ece


def write_calibrated_columns(pipelines, df_test):
    """Predict calibrated probabilities for test rows; add *_cal_* columns.

    Integer class codes are decoded back to labels via CLASSES (the
    predict_proba column order matches the encoded classes 0/1/2).
    Column prefixes follow the existing signal columns: xgboost -> xgb.
    """
    prefix_map = {"logreg": "logreg", "svm": "svm", "xgboost": "xgb"}
    for name, pipe in pipelines.items():
        prefix = prefix_map[name]
        probs = pipe.predict_proba(df_test["sentence"].astype(str))
        # sklearn orders predict_proba columns by the encoded classes [0,1,2],
        # which is exactly CLASSES order (negative, neutral, positive)
        order = np.argsort(pipe.classes_)
        probs = probs[:, order]
        preds_idx = probs.argmax(axis=1)
        # normalise away any tiny numeric drift
        probs = probs / probs.sum(axis=1, keepdims=True)
        for k, cls in enumerate(CLASSES):
            df_test[f"{prefix}_cal_prob_{cls}"] = probs[:, k]
        df_test[f"{prefix}_cal_pred"] = [CLASSES[int(i)] for i in preds_idx]
        df_test[f"{prefix}_cal_confidence"] = probs.max(axis=1)
        df_test[f"{prefix}_cal_entropy"] = [entropy_of(list(p)) for p in probs]
        tf_col = "xgb_top_features_global" if prefix == "xgb" else f"{prefix}_top_features"
        df_test[f"{prefix}_cal_top_features"] = df_test[tf_col] if tf_col in df_test.columns else "N/A"
    return df_test


def calibration_report(df_test):
    """Brier/ECE of calibrated signals vs the original uncalibrated ones."""
    labels = df_test["label"].astype(str).str.lower().tolist()
    csv_prefix = {"logreg": "logreg", "svm": "svm", "xgboost": "xgb"}
    report = {}
    for name in ("logreg", "svm", "xgboost"):
        row = {}
        for tag, prefix in (("original", csv_prefix[name]),
                            ("calibrated", f"{csv_prefix[name]}_cal")):
            probs = [[float(r[f"{prefix}_prob_{c}"]) for c in CLASSES]
                     for r in df_test.to_dict("records")]
            acc = sum(
                1 for p, y in zip(probs, labels)
                if CLASSES[int(np.argmax(p))] == y) / len(labels)
            row[tag] = {
                "accuracy": acc,
                "brier_multiclass": brier_multiclass(probs, labels),
                "ece_10bin": ece_confidence(probs, labels),
            }
        report[name] = row
    return report


# ── Prompt (identical structure; calibrated signals substituted) ───────────

def build_calibrated_prompt(row) -> str:
    """Advisory-panel prompt built from the *_cal_* columns; the structure
    and wording are identical to the original build_prompt so that the only
    difference is the numerical scale of the panel signals."""
    preds = [str(row["logreg_cal_pred"]), str(row["svm_cal_pred"]),
             str(row["xgb_cal_pred"])]
    from collections import Counter
    unique = set(preds)
    if len(unique) == 1:
        agree = "Full agreement — all 3 models agree"
    elif len(unique) == 2:
        counts = Counter(preds)
        top = counts.most_common()
        agree = (f"Majority agreement: {top[0][0]} ({top[0][1]}/3 models) "
                 f"vs {top[1][0]} ({top[1][1]}/3)")
    else:
        agree = "Split — all 3 models disagree"

    sentence = str(row["sentence"]).strip()
    lr_neg = float(row.get("logreg_cal_prob_negative", 0)) * 100
    lr_neu = float(row.get("logreg_cal_prob_neutral", 0)) * 100
    lr_pos = float(row.get("logreg_cal_prob_positive", 0)) * 100
    sv_neg = float(row.get("svm_cal_prob_negative", 0)) * 100
    sv_neu = float(row.get("svm_cal_prob_neutral", 0)) * 100
    sv_pos = float(row.get("svm_cal_prob_positive", 0)) * 100
    xg_neg = float(row.get("xgb_cal_prob_negative", 0)) * 100
    xg_neu = float(row.get("xgb_cal_prob_neutral", 0)) * 100
    xg_pos = float(row.get("xgb_cal_prob_positive", 0)) * 100
    lr_conf = float(row.get("logreg_cal_confidence", 0)) * 100
    sv_conf = float(row.get("svm_cal_confidence", 0)) * 100
    xg_conf = float(row.get("xgb_cal_confidence", 0)) * 100
    lr_ent = float(row.get("logreg_cal_entropy", 0))
    sv_ent = float(row.get("svm_cal_entropy", 0))
    xg_ent = float(row.get("xgb_cal_entropy", 0))
    lr_feats = str(row.get("logreg_cal_top_features", "N/A"))
    sv_feats = str(row.get("svm_cal_top_features", "N/A"))
    xg_feats = str(row.get("xgb_cal_top_features", "N/A"))

    return (
        "You are a senior financial sentiment analyst reviewing the output of a "
        "three-model advisory panel that has already analyzed the sentence below.\n\n"
        f'SENTENCE: "{sentence}"\n\n'
        "━━━ ADVISORY PANEL SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "[1] LOGISTIC REGRESSION  (calibrated probability expert)\n"
        f"    Prediction   : {row['logreg_cal_pred']}\n"
        f"    Confidence   : {lr_conf:.1f}%\n"
        f"    Probabilities: negative {lr_neg:.1f}% | neutral {lr_neu:.1f}% | positive {lr_pos:.1f}%\n"
        f"    Uncertainty  : {lr_ent:.4f}  (entropy — lower = more certain)\n"
        f"    Driving words: {lr_feats}\n\n"
        "[2] LINEAR SVM  (maximum-margin boundary expert)\n"
        f"    Prediction   : {row['svm_cal_pred']}\n"
        f"    Confidence   : {sv_conf:.1f}%\n"
        f"    Probabilities: negative {sv_neg:.1f}% | neutral {sv_neu:.1f}% | positive {sv_pos:.1f}%\n"
        f"    Uncertainty  : {sv_ent:.4f}\n"
        f"    Driving words: {sv_feats}\n\n"
        "[3] XGBOOST  (non-linear keyword-interaction expert)\n"
        f"    Prediction   : {row['xgb_cal_pred']}\n"
        f"    Confidence   : {xg_conf:.1f}%\n"
        f"    Probabilities: negative {xg_neg:.1f}% | neutral {xg_neu:.1f}% | positive {xg_pos:.1f}%\n"
        f"    Uncertainty  : {xg_ent:.4f}\n"
        f"    Key features : {xg_feats}\n\n"
        f"Panel status : {agree}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Based on all panel signals, determine the correct final sentiment, explain "
        "your reasoning clearly by referencing the model signals and key words, and "
        "provide an actionable business recommendation for a financial decision-maker.\n\n"
        'Respond with ONLY a JSON object — no other text, no markdown fences:\n'
        '{"label": "<positive|negative|neutral>", '
        '"explanation": "<2-3 sentences referencing panel signals and key words>", '
        '"recommendation": "<1-2 sentences of actionable advice for a financial decision-maker>"}'
    )


# ── SLM part (GPU) ─────────────────────────────────────────────────────────

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


def run_qwen_calibrated(df, cfg, n_rows, smoke=False):
    """Run Qwen ZS over the calibrated signals for the first n_rows rows.

    `df` is the FULL test frame — only the first n_rows rows are processed,
    and every checkpoint saves the full frame (never a truncated slice).
    """
    import torch
    from unsloth import FastLanguageModel
    from zero_shot_slm_predictions import (
        parse_response, FormatError, _apply_chat_template_compat)

    prefix = "qwen_calzs_smoke" if smoke else "qwen_calzs"
    for suffix in ("label", "explanation", "recommendation", "time_sec"):
        col = f"{prefix}_{suffix}"
        if col not in df.columns:
            df[col] = None

    # completeness guard: prompts must never be built from incomplete
    # calibrated signals (a missing neutral/positive probability would be
    # rendered as 0.0% and corrupt the evaluation)
    work_check = df.iloc[:n_rows]
    for pfx in ("logreg", "svm", "xgb"):
        for cls in CLASSES:
            col = f"{pfx}_cal_prob_{cls}"
            if col not in work_check.columns or work_check[col].isna().any():
                raise RuntimeError(
                    f"'{col}' is missing or incomplete for the rows to be "
                    f"processed. The calibrated signals must contain the "
                    f"complete probability vector — re-run the CPU part "
                    f"with the fixed script.")

    source = resolve_local_snapshot(cfg.qwen_model_name)
    print(f"Loading Qwen3.5-4B from: {source}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source, max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit)
    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    work = df.iloc[:n_rows]
    prompts = [build_calibrated_prompt(r) for _, r in work.iterrows()]
    import time as _time
    for i, (idx, row) in enumerate(work.iterrows()):
        if pd.notna(df.at[idx, f"{prefix}_label"]):
            continue  # resume from checkpoint
        t0 = _time.perf_counter()
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
                label, expl, rec = parse_response(text)
                df.at[idx, f"{prefix}_label"] = label
                df.at[idx, f"{prefix}_explanation"] = expl
                df.at[idx, f"{prefix}_recommendation"] = rec
                df.at[idx, f"{prefix}_time_sec"] = _time.perf_counter() - t0
                break
            except FormatError:
                continue
        if label is None:
            print(f"  [calzs] row {idx}: all retries failed — parse_error")
            df.at[idx, f"{prefix}_label"] = "[parse_error]"
            df.at[idx, f"{prefix}_time_sec"] = _time.perf_counter() - t0
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n_rows:
            df.to_csv(TEST_CSV, index=False)   # saves the FULL frame
            print(f"  calibrated Qwen ZS: {i+1}/{n_rows} rows (checkpoint saved)")

    df.to_csv(TEST_CSV, index=False)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def mcnemar_exact(pred_a, pred_b, truth):
    from math import comb
    b = sum(1 for a, p, t in zip(pred_a, pred_b, truth)
            if a == t and p != t)
    c = sum(1 for a, p, t in zip(pred_a, pred_b, truth)
            if a != t and p == t)
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    p = min(1.0, p)
    return b, c, float(p), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slm", action="store_true",
                    help="run the GPU Qwen re-evaluation after the CPU part")
    ap.add_argument("--smoke", action="store_true",
                    help="20-row SLM validation mode (columns suffixed _smoke)")
    args = ap.parse_args()

    sys.path.insert(0, "Classic")
    from zero_shot_slm_predictions import ZeroShotConfig

    df = pd.read_csv(TEST_CSV)
    n_rows = 20 if args.smoke else len(df)

    # ── guard: a previous qwen_calzs run may have been produced with
    # incomplete calibrated signals (the pre-fix merge bug wrote only
    # *_cal_prob_negative to the CSV, so prompts were rendered with
    # neutral/positive probabilities of 0.0%). Such a run is invalid and
    # is cleared automatically for a clean re-run. ─────────────────────
    if "qwen_calzs_label" in df.columns and df["qwen_calzs_label"].notna().any():
        cal_complete = all(
            f"{p}_cal_prob_{cls}" in df.columns
            and df[f"{p}_cal_prob_{cls}"].notna().all()
            for p in ("logreg", "svm", "xgb")
            for cls in CLASSES)
        if not cal_complete:
            drop = [c for c in df.columns if c.startswith("qwen_calzs_")]
            print(f"WARNING: an existing qwen_calzs_* run was found, but the "
                  f"calibrated probability columns are incomplete — that run "
                  f"was produced with degenerate prompts (neutral/positive = "
                  f"0.0%) and is INVALID. Dropping {len(drop)} columns and "
                  f"re-running the evaluation from scratch.")
            df = df.drop(columns=drop)
            df.to_csv(TEST_CSV, index=False)

    # ── CPU part: calibration + signals + report ────────────────────────────
    print(f"Fitting leakage-free sigmoid calibration on the pool "
          f"(CalibratedClassifierCV, cv=5, seed {CAL_SEED}) ...")
    pipelines = fit_calibrated_models()

    df_work = df.iloc[:n_rows].copy()
    df_work = write_calibrated_columns(pipelines, df_work)
    report = calibration_report(df_work)

    # write the *_cal_* columns — COMPLETE (all three class probabilities per
    # panel member; the pre-fix version merged only *_cal_prob_negative, so
    # prompts were later rendered with neutral/positive probabilities of 0.0%)
    if args.smoke:
        df_full = pd.read_csv(TEST_CSV)
        for c in df_work.columns:
            if ("_cal_prob_" in c) or c.endswith("_cal_pred") or \
               c.endswith("_cal_confidence") or c.endswith("_cal_entropy") or \
               c.endswith("_cal_top_features"):
                if c not in df_full.columns:
                    df_full[c] = None
                df_full.loc[df_work.index, c] = df_work[c]
        df_full.to_csv(TEST_CSV, index=False)
    else:
        df_work.to_csv(TEST_CSV, index=False)   # df_work IS the full frame
    print(f"Written calibrated panel-signal columns to {TEST_CSV}")

    if args.smoke:
        # a smoke run covers 20 rows only — never overwrite the genuine
        # 340-row report; print the 20-row numbers for inspection instead
        print("\n== 20-row smoke report (NOT saved) ==")
        for name in report:
            o, c = report[name]["original"], report[name]["calibrated"]
            print(f"  {name}: Brier {o['brier_multiclass']:.4f} -> "
                  f"{c['brier_multiclass']:.4f} | ECE {o['ece_10bin']:.4f} -> "
                  f"{c['ece_10bin']:.4f}")
        print("SMOKE: calibration report not written; full run will save it.")
        if not args.slm:
            return
        df_full = pd.read_csv(TEST_CSV)
        n_run = 20
        cfg = ZeroShotConfig()
        df_run = run_qwen_calibrated(df_full, cfg, n_run, smoke=args.smoke)
        print("\nSMOKE TEST COMPLETE (calibrated CPU part + 20-row SLM).")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "calibrated_signals_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Written: {RESULTS_DIR}/calibrated_signals_report.json")
    print("\n== Calibration improvement (test set) ==")
    for name in report:
        o, c = report[name]["original"], report[name]["calibrated"]
        print(f"  {name}: Brier {o['brier_multiclass']:.4f} -> "
              f"{c['brier_multiclass']:.4f} | ECE {o['ece_10bin']:.4f} -> "
              f"{c['ece_10bin']:.4f} | acc {o['accuracy']*100:.2f}% -> "
              f"{c['accuracy']*100:.2f}%")

    if not args.slm:
        print("\nCPU part complete. Re-run with --slm to add the GPU "
              "Qwen re-evaluation.")
        return

    # ── GPU part: Qwen ZS over calibrated signals ──────────────────────────
    df_full = pd.read_csv(TEST_CSV)
    have = [c for c in df_full.columns if c.endswith("_cal_prob_negative")]
    if len(have) != 3:
        raise RuntimeError("Calibrated columns missing — rerun CPU part.")
    n_run = len(df_full)
    cfg = ZeroShotConfig()
    df_run = run_qwen_calibrated(df_full, cfg, n_run, smoke=False)

    # ── metrics + McNemar vs original ──────────────────────────────────────
    labels = df_run["label"].astype(str).str.lower().tolist()
    cal = df_run["qwen_calzs_label"].astype(str).str.lower().tolist()
    orig = df_run["qwen_zs_label"].astype(str).str.lower().tolist()
    n_eval = sum(1 for l in cal if not l.startswith("[parse_error]"))
    acc_cal = sum(a == t for a, t in zip(cal, labels)) / len(labels)
    acc_orig = sum(a == t for a, t in zip(orig, labels)) / len(labels)
    b, c_, p = mcnemar_exact(cal, orig, labels)[:3]
    print(f"\nQwen over calibrated signals : acc={acc_cal*100:.2f}% "
          f"(n_evaluated={n_eval})")
    print(f"Qwen over original signals   : acc={acc_orig*100:.2f}%")
    print(f"McNemar calibrated vs original: b={b} c={c_} exact p={p:.4f}")

    out = {
        "qwen_calzs_accuracy": acc_cal,
        "qwen_zs_accuracy_original": acc_orig,
        "n_evaluated": n_eval,
        "mcnemar_calzs_vs_orig": {"b": b, "c": c_, "p_exact": p},
    }
    with open(RESULTS_DIR / "calibrated_signals_srun.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {RESULTS_DIR}/calibrated_signals_srun.json")


if __name__ == "__main__":
    main()