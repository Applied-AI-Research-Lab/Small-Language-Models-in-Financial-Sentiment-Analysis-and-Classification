#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


@dataclass
class TrainConfig:
    train_csv: str = "Datasets/financial_phrasebank/train.csv"
    val_csv: str = "Datasets/financial_phrasebank/validation.csv"

    artifacts_dir: str = "Classic/artifacts"
    results_dir: str = "Classic/results"

    random_state: int = 42
    n_splits: int = 5

    tfidf_max_features: int = 20000
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.98

    logreg_c: float = 4.0
    svm_c: float = 1.0

    xgb_n_estimators: int = 350
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9


def ensure_dirs(cfg: TrainConfig) -> tuple[Path, Path]:
    artifacts_dir = Path(cfg.artifacts_dir)
    results_dir = Path(cfg.results_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir, results_dir


def load_train_val(cfg: TrainConfig) -> pd.DataFrame:
    train_df = pd.read_csv(cfg.train_csv)
    val_df = pd.read_csv(cfg.val_csv)

    for col in ["sentence", "label"]:
        if col not in train_df.columns or col not in val_df.columns:
            raise ValueError("Train and validation CSVs must include sentence,label columns.")

    df = pd.concat([train_df[["sentence", "label"]], val_df[["sentence", "label"]]], ignore_index=True)
    df = df.dropna().copy()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def make_vectorizer(cfg: TrainConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        ngram_range=(cfg.tfidf_ngram_min, cfg.tfidf_ngram_max),
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        sublinear_tf=True,
    )


def cv_logreg(X_text: pd.Series, y: np.ndarray, cfg: TrainConfig) -> list[dict]:
    rows = []
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y), start=1):
        vec = make_vectorizer(cfg)
        X_tr = vec.fit_transform(X_text.iloc[tr_idx])
        X_va = vec.transform(X_text.iloc[va_idx])

        model = LogisticRegression(
            C=cfg.logreg_c,
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=cfg.random_state,
        )
        model.fit(X_tr, y[tr_idx])

        pred = model.predict(X_va)
        rows.append(
            {
                "model": "logreg",
                "fold": fold,
                "accuracy": accuracy_score(y[va_idx], pred),
                "macro_f1": f1_score(y[va_idx], pred, average="macro"),
                "weighted_f1": f1_score(y[va_idx], pred, average="weighted"),
            }
        )
    return rows


def cv_svm(X_text: pd.Series, y: np.ndarray, cfg: TrainConfig) -> list[dict]:
    rows = []
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y), start=1):
        vec = make_vectorizer(cfg)
        X_tr = vec.fit_transform(X_text.iloc[tr_idx])
        X_va = vec.transform(X_text.iloc[va_idx])

        model = LinearSVC(C=cfg.svm_c, class_weight="balanced", random_state=cfg.random_state)
        model.fit(X_tr, y[tr_idx])

        pred = model.predict(X_va)
        rows.append(
            {
                "model": "svm",
                "fold": fold,
                "accuracy": accuracy_score(y[va_idx], pred),
                "macro_f1": f1_score(y[va_idx], pred, average="macro"),
                "weighted_f1": f1_score(y[va_idx], pred, average="weighted"),
            }
        )
    return rows


def cv_xgb(X_text: pd.Series, y_label: np.ndarray, cfg: TrainConfig) -> tuple[list[dict], LabelEncoder]:
    rows = []
    le = LabelEncoder()
    y = le.fit_transform(y_label)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y), start=1):
        vec = make_vectorizer(cfg)
        X_tr = vec.fit_transform(X_text.iloc[tr_idx])
        X_va = vec.transform(X_text.iloc[va_idx])

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(le.classes_),
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            random_state=cfg.random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        model.fit(X_tr, y[tr_idx])

        pred = model.predict(X_va)
        rows.append(
            {
                "model": "xgboost",
                "fold": fold,
                "accuracy": accuracy_score(y[va_idx], pred),
                "macro_f1": f1_score(y[va_idx], pred, average="macro"),
                "weighted_f1": f1_score(y[va_idx], pred, average="weighted"),
            }
        )
    return rows, le


def train_full_models(
    X_text: pd.Series,
    y_label: np.ndarray,
    cfg: TrainConfig,
    artifacts_dir: Path,
) -> list[dict]:
    timing_rows = []

    t0 = time.perf_counter()
    vec_lr = make_vectorizer(cfg)
    X_lr = vec_lr.fit_transform(X_text)
    vectorize_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    mdl_lr = LogisticRegression(
        C=cfg.logreg_c,
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=cfg.random_state,
    )
    mdl_lr.fit(X_lr, y_label)
    fit_sec = time.perf_counter() - t1

    joblib.dump({"vectorizer": vec_lr, "model": mdl_lr}, artifacts_dir / "logreg_tfidf.joblib")
    timing_rows.append(
        {
            "model": "logreg",
            "vectorize_time_sec": vectorize_sec,
            "fit_time_sec": fit_sec,
            "total_train_time_sec": vectorize_sec + fit_sec,
        }
    )

    t0 = time.perf_counter()
    vec_svm = make_vectorizer(cfg)
    X_svm = vec_svm.fit_transform(X_text)
    vectorize_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    mdl_svm = LinearSVC(C=cfg.svm_c, class_weight="balanced", random_state=cfg.random_state)
    mdl_svm.fit(X_svm, y_label)
    fit_sec = time.perf_counter() - t1

    joblib.dump({"vectorizer": vec_svm, "model": mdl_svm}, artifacts_dir / "svm_tfidf.joblib")
    timing_rows.append(
        {
            "model": "svm",
            "vectorize_time_sec": vectorize_sec,
            "fit_time_sec": fit_sec,
            "total_train_time_sec": vectorize_sec + fit_sec,
        }
    )

    t0 = time.perf_counter()
    vec_xgb = make_vectorizer(cfg)
    X_xgb = vec_xgb.fit_transform(X_text)
    vectorize_sec = time.perf_counter() - t0

    le = LabelEncoder()
    y_xgb = le.fit_transform(y_label)

    t1 = time.perf_counter()
    mdl_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        random_state=cfg.random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    mdl_xgb.fit(X_xgb, y_xgb)
    fit_sec = time.perf_counter() - t1

    feature_names = np.array(vec_xgb.get_feature_names_out())
    importances = mdl_xgb.feature_importances_
    top_idx = np.argsort(importances)[-30:][::-1]
    top_features = [{"feature": feature_names[i], "importance": float(importances[i])} for i in top_idx]

    joblib.dump(
        {
            "vectorizer": vec_xgb,
            "model": mdl_xgb,
            "label_encoder": le,
            "top_features": top_features,
        },
        artifacts_dir / "xgboost_tfidf.joblib",
    )

    timing_rows.append(
        {
            "model": "xgboost",
            "vectorize_time_sec": vectorize_sec,
            "fit_time_sec": fit_sec,
            "total_train_time_sec": vectorize_sec + fit_sec,
        }
    )

    return timing_rows


def summarize_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cv_df.groupby("model", as_index=False)[["accuracy", "macro_f1", "weighted_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary = summary.rename(columns={"model_": "model"})
    return summary


def main() -> None:
    cfg = TrainConfig()
    artifacts_dir, results_dir = ensure_dirs(cfg)

    df = load_train_val(cfg)
    X_text = df["sentence"]
    y_label = df["label"].to_numpy()

    cv_rows = []
    cv_rows.extend(cv_logreg(X_text, y_label, cfg))
    cv_rows.extend(cv_svm(X_text, y_label, cfg))

    xgb_rows, xgb_le = cv_xgb(X_text, y_label, cfg)
    cv_rows.extend(xgb_rows)

    cv_df = pd.DataFrame(cv_rows)
    cv_summary_df = summarize_cv(cv_df)

    cv_df.to_csv(results_dir / "cv_folds_metrics.csv", index=False)
    cv_summary_df.to_csv(results_dir / "cv_summary_metrics.csv", index=False)

    train_timing_rows = train_full_models(X_text, y_label, cfg, artifacts_dir)
    train_timing_df = pd.DataFrame(train_timing_rows)
    train_timing_df.to_csv(results_dir / "train_time_per_model.csv", index=False)

    run_summary = {
        "config": asdict(cfg),
        "train_val_rows": int(len(df)),
        "label_distribution": df["label"].value_counts().to_dict(),
        "xgb_label_classes": list(map(str, xgb_le.classes_)),
        "artifacts_dir": str(artifacts_dir),
        "results_dir": str(results_dir),
    }
    (results_dir / "train_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Classic model training completed.")
    print(f"CV fold metrics: {results_dir / 'cv_folds_metrics.csv'}")
    print(f"CV summary metrics: {results_dir / 'cv_summary_metrics.csv'}")
    print(f"Training time per model: {results_dir / 'train_time_per_model.csv'}")
    print(f"Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
