import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "Classic" / "results"
D1 = RES / "round2" / "d1_expert"
C1 = ROOT / "round2_experiments" / "results"
SEED = 42


def norm(s):
    return s.astype(str).str.strip().str.lower()


def mcnemar(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = float(binomtest(b, b + c).pvalue) if b + c else 1.0
    return {"b": b, "c": c, "p_exact": p}


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    out, run, k = {}, 0.0, len(names)
    for i, n in enumerate(names):
        run = max(run, min(1.0, (k - i) * pvals[n]))
        out[n] = run
    return out


def main():
    t = pd.read_csv(ROOT / "Datasets" / "financial_phrasebank" / "test.csv")
    fb = pd.read_csv(RES / "finbert_per_row.csv")
    bert = pd.read_csv(RES / "bert_per_row.csv")
    assert (t.sentence.values == fb.sentence.values).all()
    assert (t.sentence.values == bert.sentence.values).all()
    y = norm(t.label)
    ok = {c: (norm(t[c]) == y).values for c in [
        "qwen_zs_label", "gemma_zs_label", "qwen_ft_tuned_label", "gemma_ft_tuned_label",
        "qwen_sentonly_zs_label", "gemma_sentonly_zs_label"]}
    ok["finbert_zs"] = (norm(fb.finbert_zs_pred) == y).values
    ok["finbert_ft"] = (norm(fb.finbert_ft_pred) == y).values
    ok["bert_ft"] = (norm(bert.bert_ft_pred) == y).values
    res = {}

    res["tuned_vs_finbert"] = {
        "qwen": mcnemar(ok["qwen_ft_tuned_label"], ok["finbert_zs"]),
        "gemma": mcnemar(ok["gemma_ft_tuned_label"], ok["finbert_zs"]),
    }
    posthoc = {
        "tuned_qwen_vs_zs_qwen": mcnemar(ok["qwen_ft_tuned_label"], ok["qwen_zs_label"]),
        "tuned_gemma_vs_zs_gemma": mcnemar(ok["gemma_ft_tuned_label"], ok["gemma_zs_label"]),
        "finbert_zs_vs_zs_qwen": mcnemar(ok["finbert_zs"], ok["qwen_zs_label"]),
        "finbert_ft_vs_zs_qwen": mcnemar(ok["finbert_ft"], ok["qwen_zs_label"]),
    }
    adj = holm({k: v["p_exact"] for k, v in posthoc.items()})
    for k in posthoc:
        posthoc[k]["p_holm"] = adj[k]
    res["posthoc_vs_zero_shot"] = posthoc
    res["bert_vs_finbert"] = mcnemar(ok["bert_ft"], ok["finbert_zs"])

    q, g = norm(t.qwen_zs_label), norm(t.gemma_zs_label)
    res["qwen_vs_gemma_zs"] = {
        "labels_differ": int((q != g).sum()),
        "correctness_differs": int(((q == y) != (g == y)).sum()),
        "both_wrong_different_label": int(((q != y) & (g != y) & (q != g)).sum()),
    }

    panel = pd.DataFrame({c: norm(t[c]) for c in ["logreg_pred", "svm_pred", "xgb_pred"]})
    majority = panel.mode(axis=1)[0]
    unanimous = panel.nunique(axis=1) == 1
    runs = {}
    for c in ["qwen_zs_label", "qwen_zs_seed1001_label", "qwen_zs_seed1002_label", "qwen_zs_seed1003_label"]:
        lab = norm(t[c])
        over = unanimous & (lab != majority)
        runs[c] = {"accuracy": float((lab == y).mean()), "unanimous_overrides": int(over.sum()),
                   "correct": int((over & (lab == y)).sum()), "damages": int((over & (majority == y)).sum())}
    res["qwen_override_by_decoding_run"] = runs

    mc = json.loads((C1 / "mcnemar_c1.json").read_text())
    tests = {k: v["vs_framework"]["p_exact"] for k, v in mc.items()}
    tests["qwen/sentence_only"] = mcnemar(ok["qwen_zs_label"], ok["qwen_sentonly_zs_label"])["p_exact"]
    tests["gemma/sentence_only"] = mcnemar(ok["gemma_zs_label"], ok["gemma_sentonly_zs_label"])["p_exact"]
    res["llm_baselines_holm"] = {k: {"p_exact": tests[k], "p_holm": v} for k, v in holm(tests).items()}

    knn = pd.read_csv(C1 / "c1" / "knn_qwen.csv").sort_values("idx")
    assert (knn.sentence.values == t.sentence.values).all()
    d = (norm(knn.pred) == norm(knn.label)).values.astype(int) - ok["qwen_zs_label"].astype(int)
    rng = np.random.default_rng(SEED)
    boots = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(10000)]
    res["retrieval_minus_framework_qwen_pp"] = {
        "delta": float(d.mean() * 100), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])]}

    key = pd.read_csv(D1 / "d1_blinding_key_PRIVATE.csv")
    experts = {}
    for e in (1, 2):
        w = pd.read_excel(D1 / f"D1_ExpertEvaluation_Expert{e}.xlsx", sheet_name="Ratings").merge(key, on="row")
        for q_ in ("Q1", "Q2"):
            name = {"Q1": "factual_accuracy", "Q2": "faithfulness"}[q_]
            a_col = pd.to_numeric(w[f"A_{q_}_{name}_1to5"], errors="coerce")
            b_col = pd.to_numeric(w[f"B_{q_}_{name}_1to5"], errors="coerce")
            w[f"fw_{q_}"] = np.where(w.A == "framework", a_col, b_col)
            w[f"so_{q_}"] = np.where(w.A == "framework", b_col, a_col)
        pref = w.PREFERENCE_A_B_Equal.astype(str).str.strip().str.upper()
        fw_pref = ((pref == "A") & (w.A == "framework")) | ((pref == "B") & (w.B == "framework"))
        so_pref = pref.isin(["A", "B"]) & ~fw_pref
        comments = w.comments.dropna().astype(str).str.strip()
        comments = comments[comments != ""]
        out = {"votes": {"framework": int(fw_pref.sum()), "sentence_only": int(so_pref.sum()),
                         "equal": int((pref == "EQUAL").sum())},
               "sign_test_p": float(binomtest(int(so_pref.sum()), int(so_pref.sum() + fw_pref.sum())).pvalue),
               "comments": int(len(comments)),
               "comments_mentioning_xgboost": int(comments.str.contains("xgb", case=False).sum())}
        for q_ in ("Q1", "Q2"):
            diff = w[f"so_{q_}"] - w[f"fw_{q_}"]
            out[q_] = {"framework": float(w[f"fw_{q_}"].mean()), "sentence_only": float(w[f"so_{q_}"].mean()),
                       "wilcoxon_p": float(wilcoxon(diff, zero_method="pratt").pvalue)}
        experts[f"expert{e}"] = out
    res["d1_per_expert"] = experts

    (RES / "round2" / "supplementary_tests.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
