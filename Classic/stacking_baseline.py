
import csv
import json
import math
from pathlib import Path

POOL_CSVS = [Path("Datasets/financial_phrasebank/train.csv"),
             Path("Datasets/financial_phrasebank/validation.csv")]
TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
OUT_DIR = Path("Classic/results")
SEED = 42
CLASSES = ("negative", "neutral", "positive")


def mcnemar_exact(y_a, y_b, truth):
    b = sum(1 for a, c, t in zip(y_a, y_b, truth) if a == t and c != t)
    c_ = sum(1 for a, c, t in zip(y_a, y_b, truth) if a != t and c == t)
    nd = b + c_
    if nd == 0:
        return b, c_, 0.0, 1.0
    chi2 = (abs(b - c_) - 1) ** 2 / nd
    k = min(b, c_)
    tail = sum(math.comb(nd, i) for i in range(k + 1)) / 2 ** nd
    return b, c_, chi2, min(1.0, 2 * tail)


def macro_metrics(y_true, y_pred):
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    f1s, sups = [], []
    for cls in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t != cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != cls and t == cls)
        pe = tp / (tp + fp) if tp + fp else 0.0
        re = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pe * re / (pe + re) if pe + re else 0.0)
        sups.append(sum(1 for t in y_true if t == cls))
    macro = sum(f1s) / len(f1s)
    weighted = sum(f * s for f, s in zip(f1s, sups)) / n
    return acc, macro, weighted


def main():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC
    from sklearn.utils.extmath import softmax as sklearn_softmax
    from xgboost import XGBClassifier

    pool = pd.concat([pd.read_csv(p) for p in POOL_CSVS], ignore_index=True)
    X_text = pool["sentence"].astype(str).tolist()
    y_pool = pool["label"].tolist()
    y_enc = np.array([CLASSES.index(t) for t in y_pool])
    print(f"Pool: {len(pool)} sentences.")

    from sklearn.feature_extraction.text import TfidfVectorizer

    def make_vectorizer():
        return TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                               min_df=2, max_df=0.98, sublinear_tf=True)

    def fit_models(vec, texts, labels):
        X = vec.fit_transform(texts)
        lr = LogisticRegression(C=4.0, class_weight="balanced",
                                max_iter=4000, random_state=SEED)
        svm = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
        xgb = XGBClassifier(objective="multi:softprob", n_estimators=350,
                            max_depth=6, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, n_jobs=-1, random_state=SEED,
                            eval_metric="mlogloss", verbosity=0)
        lr.fit(X, labels)
        svm.fit(X, labels)
        xgb.fit(X, labels)
        return lr, svm, xgb

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(pool), 9))
    for fold, (tr, te) in enumerate(skf.split(X_text, y_enc)):
        vec = make_vectorizer()
        Xtr = vec.fit_transform([X_text[i] for i in tr])
        Xte = vec.transform([X_text[i] for i in te])
        ytr = y_enc[tr]
        lr = LogisticRegression(C=4.0, class_weight="balanced",
                                max_iter=4000, random_state=SEED).fit(Xtr, ytr)
        svm = LinearSVC(C=1.0, class_weight="balanced",
                        random_state=SEED).fit(Xtr, ytr)
        xgb = XGBClassifier(objective="multi:softprob", n_estimators=350,
                            max_depth=6, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, n_jobs=-1, random_state=SEED,
                            eval_metric="mlogloss", verbosity=0).fit(Xtr, ytr)
        oof[np.ix_(te, [0, 1, 2])] = lr.predict_proba(Xte)
        scores = svm.decision_function(Xte)
        oof[np.ix_(te, [3, 4, 5])] = sklearn_softmax(scores)
        oof[np.ix_(te, [6, 7, 8])] = xgb.predict_proba(Xte)
        print(f"  fold {fold+1}/5 done")
    print("OOF probability features complete.")

    stacker = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    stacker.fit(oof, y_enc)

    test = pd.read_csv(TEST_CSV)
    feats = []
    for _, r in test.iterrows():
        feats.append([
            float(r["logreg_prob_negative"]), float(r["logreg_prob_neutral"]), float(r["logreg_prob_positive"]),
            float(r["svm_prob_negative"]), float(r["svm_prob_neutral"]), float(r["svm_prob_positive"]),
            float(r["xgb_prob_negative"]), float(r["xgb_prob_neutral"]), float(r["xgb_prob_positive"]),
        ])
    X_test = np.array(feats)
    pred_enc = stacker.predict(X_test)
    preds = [CLASSES[i] for i in pred_enc]
    y_true = test["label"].tolist()

    acc, mf1, wf1 = macro_metrics(y_true, preds)
    print(f"\nStacking (LR over 9 OOF probabilities): "
          f"acc={acc*100:.2f}%  macroF1={mf1:.4f}  wF1={wf1:.4f}")

    from collections import Counter
    maj = [Counter([r["logreg_pred"], r["svm_pred"], r["xgb_pred"]]).most_common(1)[0][0]
           for _, r in test.iterrows()]
    maj_acc, maj_f1, maj_wf1 = macro_metrics(y_true, maj)
    print(f"Majority vote: acc={maj_acc*100:.2f}%")
    qz = test["qwen_zs_label"].astype(str).str.lower().tolist()
    qz_acc = sum(a == t for a, t in zip(qz, y_true)) / len(y_true)
    print(f"Qwen ZS (framework): acc={qz_acc*100:.2f}%")

    mcn = {}
    b, c, chi2, p = mcnemar_exact(preds, qz, y_true)
    mcn["stacking_vs_qwen_zs"] = {"b": b, "c": c, "chi2_cc": chi2, "p_exact": p}
    print(f"McNemar stacking vs Qwen ZS: b={b} c={c} p={p:.4f}")
    b, c, chi2, p = mcnemar_exact(preds, maj, y_true)
    mcn["stacking_vs_majority"] = {"b": b, "c": c, "chi2_cc": chi2, "p_exact": p}
    print(f"McNemar stacking vs majority: b={b} c={c} p={p:.4f}")

    results = {
        "stacking_lr": {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
                        "n": len(y_true), "features": "9 OOF probabilities",
                        "cv": "5-fold stratified, seed 42"},
        "majority_vote": {"accuracy": maj_acc, "macro_f1": maj_f1,
                          "weighted_f1": maj_wf1},
        "mcnemar": mcn,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stacking_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "stacking_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "accuracy", "macro_f1", "weighted_f1"])
        w.writerow(["stacking_lr", f"{acc:.4f}", f"{mf1:.4f}", f"{wf1:.4f}"])
        w.writerow(["majority_vote", f"{maj_acc:.4f}", f"{maj_f1:.4f}", f"{maj_wf1:.4f}"])
    print(f"\nWritten: {OUT_DIR}/stacking_metrics.json, stacking_metrics.csv")


if __name__ == "__main__":
    main()