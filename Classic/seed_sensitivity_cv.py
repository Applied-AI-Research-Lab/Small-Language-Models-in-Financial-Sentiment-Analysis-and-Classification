
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

POOL_CSVS = [Path("Datasets/financial_phrasebank/train.csv"),
             Path("Datasets/financial_phrasebank/validation.csv")]
TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
OUT_DIR = Path("Classic/results")
SEEDS = [42, 7, 123, 2024, 555]
CLASSES = ("negative", "neutral", "positive")


def make_vectorizer():
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                           min_df=2, max_df=0.98, sublinear_tf=True)


def make_models(seed):
    return {
        "logreg": LogisticRegression(C=4.0, class_weight="balanced",
                                     max_iter=4000, random_state=seed),
        "svm": LinearSVC(C=1.0, class_weight="balanced", random_state=seed),
        "xgboost": XGBClassifier(objective="multi:softprob", n_estimators=350,
                                 max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9,
                                 n_jobs=-1, random_state=seed,
                                 eval_metric="mlogloss", verbosity=0),
    }


def main():
    pool = pd.concat([pd.read_csv(p) for p in POOL_CSVS], ignore_index=True)
    test = pd.read_csv(TEST_CSV)
    y_pool = np.array([CLASSES.index(t) for t in pool["label"]])
    y_test = np.array([CLASSES.index(t) for t in test["label"]])
    print(f"Pool: {len(pool)} | Test: {len(test)}")

    cv_acc = {m: [] for m in ("logreg", "svm", "xgboost")}
    test_acc = {m: [] for m in ("logreg", "svm", "xgboost")}

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_acc = {m: [] for m in cv_acc}
        for tr, te in skf.split(pool["sentence"], y_pool):
            vec = make_vectorizer()
            Xtr = vec.fit_transform(pool["sentence"].iloc[tr])
            Xte = vec.transform(pool["sentence"].iloc[te])
            models = make_models(seed)
            for name, mdl in models.items():
                mdl.fit(Xtr, y_pool[tr])
                pred = mdl.predict(Xte)
                fold_acc[name].append(float((pred == y_pool[te]).mean()))
        for name in cv_acc:
            cv_acc[name].append(float(np.mean(fold_acc[name])))

        vec = make_vectorizer()
        X_pool = vec.fit_transform(pool["sentence"])
        X_test = vec.transform(test["sentence"])
        models = make_models(seed)
        for name, mdl in models.items():
            mdl.fit(X_pool, y_pool)
            pred = mdl.predict(X_test)
            test_acc[name].append(float((pred == y_test).mean()))
        print(f"  seed {seed}: CV acc "
              + " ".join(f"{n}={cv_acc[n][-1]*100:.2f}%" for n in cv_acc)
              + " | test acc "
              + " ".join(f"{n}={test_acc[n][-1]*100:.2f}%" for n in test_acc))

    results = {}
    print("\n== Across seeds (mean ± std) ==")
    for name in cv_acc:
        cv_a = np.array(cv_acc[name])
        t_a = np.array(test_acc[name])
        results[name] = {
            "cv_mean": float(cv_a.mean()), "cv_std": float(cv_a.std(ddof=1)),
            "test_mean": float(t_a.mean()), "test_std": float(t_a.std(ddof=1)),
            "per_seed_cv": cv_acc[name], "per_seed_test": test_acc[name],
        }
        print(f"  {name:8s} CV: {cv_a.mean()*100:.2f}% ± {cv_a.std(ddof=1)*100:.2f}"
              f"  | test: {t_a.mean()*100:.2f}% ± {t_a.std(ddof=1)*100:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "seed_sensitivity_cv.json", "w") as f:
        json.dump({"seeds": SEEDS, "results": results}, f, indent=2)
    with open(OUT_DIR / "seed_sensitivity_cv.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "cv_mean", "cv_std", "test_mean", "test_std"])
        for name, r in results.items():
            w.writerow([name, f"{r['cv_mean']:.4f}", f"{r['cv_std']:.4f}",
                         f"{r['test_mean']:.4f}", f"{r['test_std']:.4f}"])
    print(f"\nWritten: {OUT_DIR}/seed_sensitivity_cv.json, seed_sensitivity_cv.csv")


if __name__ == "__main__":
    main()