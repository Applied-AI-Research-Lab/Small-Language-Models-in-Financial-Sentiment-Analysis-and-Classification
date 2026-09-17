#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class PredictConfig:
    test_csv: str = "Datasets/financial_phrasebank/test.csv"
    artifacts_dir: str = "Classic/artifacts"
    results_dir: str = "Classic/results"
    top_k_features: int = 5


def entropy_of_probs(prob_row: np.ndarray) -> float:
    p = np.clip(prob_row.astype(float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def margin_from_probs(prob_row: np.ndarray) -> float:
    s = np.sort(prob_row)
    if len(s) < 2:
        return float(s[-1])
    return float(s[-1] - s[-2])


def top_tokens_from_linear_row(x_row, coef_row: np.ndarray, feat_names: np.ndarray, top_k: int) -> str:
    contrib = x_row.multiply(coef_row)
    arr = np.asarray(contrib.toarray()).ravel()
    nz = np.flatnonzero(arr)
    if nz.size == 0:
        return ""
    best = nz[np.argsort(arr[nz])[-top_k:]][::-1]
    tokens = [f"{feat_names[i]}:{arr[i]:.4f}" for i in best]
    return " | ".join(tokens)


def add_logreg_signals(df: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    bundle = joblib.load(Path(cfg.artifacts_dir) / "logreg_tfidf.joblib")
    vec = bundle["vectorizer"]
    model = bundle["model"]

    X = vec.transform(df["sentence"].astype(str).tolist())
    class_labels = list(map(str, model.classes_))

    probs = []
    pred = []
    row_times = []
    pred_idx = []
    for i in range(X.shape[0]):
        x_row = X[i]
        start = time.perf_counter()
        row_prob = model.predict_proba(x_row)[0]
        row_time = time.perf_counter() - start

        row_idx = int(np.argmax(row_prob))
        probs.append(row_prob)
        pred.append(class_labels[row_idx])
        pred_idx.append(row_idx)
        row_times.append(row_time)

    probs = np.asarray(probs)
    feat_names = np.array(vec.get_feature_names_out())

    df["logreg_pred"] = pred
    df["logreg_confidence"] = probs.max(axis=1)
    df["logreg_margin"] = [margin_from_probs(row) for row in probs]
    df["logreg_entropy"] = [entropy_of_probs(row) for row in probs]
    df["logreg_time_sec"] = row_times

    for cls in class_labels:
        cls_idx = class_labels.index(cls)
        df[f"logreg_prob_{cls}"] = probs[:, cls_idx]

    top_feats = []
    for i in range(X.shape[0]):
        cls_i = int(pred_idx[i])
        top_feats.append(top_tokens_from_linear_row(X[i], model.coef_[cls_i], feat_names, cfg.top_k_features))
    df["logreg_top_features"] = top_feats

    return df


def add_svm_signals(df: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    bundle = joblib.load(Path(cfg.artifacts_dir) / "svm_tfidf.joblib")
    vec = bundle["vectorizer"]
    model = bundle["model"]

    X = vec.transform(df["sentence"].astype(str).tolist())
    class_labels = list(map(str, model.classes_))

    scores = []
    pred = []
    row_times = []
    pred_idx = []
    for i in range(X.shape[0]):
        x_row = X[i]
        start = time.perf_counter()
        row_score = model.decision_function(x_row)
        row_time = time.perf_counter() - start

        row_score = np.asarray(row_score).ravel()
        if row_score.ndim == 1 and row_score.shape[0] == 1:
            row_score = np.array([-row_score[0], row_score[0]])

        row_probs = softmax(row_score)
        row_idx = int(np.argmax(row_probs))

        scores.append(row_score)
        pred.append(class_labels[row_idx])
        pred_idx.append(row_idx)
        row_times.append(row_time)

    pseudo_probs = np.asarray([softmax(np.asarray(s).ravel()) for s in scores])

    feat_names = np.array(vec.get_feature_names_out())

    df["svm_pred"] = pred
    df["svm_confidence"] = pseudo_probs.max(axis=1)
    df["svm_margin"] = [margin_from_probs(row) for row in pseudo_probs]
    df["svm_entropy"] = [entropy_of_probs(row) for row in pseudo_probs]
    df["svm_time_sec"] = row_times

    for cls in class_labels:
        cls_idx = class_labels.index(cls)
        df[f"svm_prob_{cls}"] = pseudo_probs[:, cls_idx]

    top_feats = []
    for i in range(X.shape[0]):
        cls_i = int(pred_idx[i])
        top_feats.append(top_tokens_from_linear_row(X[i], model.coef_[cls_i], feat_names, cfg.top_k_features))
    df["svm_top_features"] = top_feats

    return df


def add_xgb_signals(df: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    bundle = joblib.load(Path(cfg.artifacts_dir) / "xgboost_tfidf.joblib")
    vec = bundle["vectorizer"]
    model = bundle["model"]
    le = bundle["label_encoder"]
    top_features = bundle.get("top_features", [])

    X = vec.transform(df["sentence"].astype(str).tolist())

    class_labels = list(map(str, le.classes_))

    probs = []
    pred = []
    row_times = []
    for i in range(X.shape[0]):
        x_row = X[i]
        start = time.perf_counter()
        row_prob = model.predict_proba(x_row)[0]
        row_time = time.perf_counter() - start

        row_idx = int(np.argmax(row_prob))
        probs.append(row_prob)
        pred.append(class_labels[row_idx])
        row_times.append(row_time)

    probs = np.asarray(probs)

    df["xgb_pred"] = pred
    df["xgb_confidence"] = probs.max(axis=1)
    df["xgb_margin"] = [margin_from_probs(row) for row in probs]
    df["xgb_entropy"] = [entropy_of_probs(row) for row in probs]
    df["xgb_time_sec"] = row_times

    for cls in class_labels:
        cls_idx = class_labels.index(cls)
        df[f"xgb_prob_{cls}"] = probs[:, cls_idx]

    global_feats = " | ".join([f"{d['feature']}:{d['importance']:.4f}" for d in top_features[: cfg.top_k_features]])
    df["xgb_top_features_global"] = global_feats

    return df


def main() -> None:
    cfg = PredictConfig()
    test_path = Path(cfg.test_csv)
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test CSV: {test_path}")

    df = pd.read_csv(test_path)
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError("Test CSV must include sentence,label columns.")

    artifacts_dir = Path(cfg.artifacts_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    required = [
        artifacts_dir / "logreg_tfidf.joblib",
        artifacts_dir / "svm_tfidf.joblib",
        artifacts_dir / "xgboost_tfidf.joblib",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained artifacts. Run Classic/train_classic_models.py first. Missing: " + ", ".join(missing)
        )

    df = add_logreg_signals(df, cfg)
    df = add_svm_signals(df, cfg)
    df = add_xgb_signals(df, cfg)

    df.to_csv(test_path, index=False)

    y_true = df["label"].astype(str).str.lower().to_numpy()
    metrics_rows = []
    for model_col, model_name in [
        ("logreg_pred", "logreg"),
        ("xgb_pred", "xgboost"),
        ("svm_pred", "svm"),
    ]:
        y_pred = df[model_col].astype(str).str.lower().to_numpy()
        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            }
        )
    pd.DataFrame(metrics_rows).to_csv(results_dir / "test_metrics_classic.csv", index=False)

    print("Classic model predictions completed.")
    print(f"Updated test CSV: {test_path}")
    print(f"Test metrics: {results_dir / 'test_metrics_classic.csv'}")


if __name__ == "__main__":
    main()
