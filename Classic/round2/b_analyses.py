#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "Datasets" / "financial_phrasebank"
RES = ROOT / "Classic" / "results"
OUT = RES / "round2"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
NB = 10000

d = pd.read_csv(D / "test.csv")
L = lambda s: s.astype(str).str.strip().str.lower()
g = L(d.label)
votes = d[["logreg_pred", "svm_pred", "xgb_pred"]].apply(L)
unan = votes.nunique(axis=1) == 1
mv = votes.mode(axis=1)[0]
assert (votes.nunique(axis=1) <= 2).all()
N, NU, NS = len(d), int(unan.sum()), int((~unan).sum())
R: dict = {}


def mc(a, b, mask=None):
    m = slice(None) if mask is None else mask
    A, B, G = a[m] == g[m], b[m] == g[m], None
    x, y = int((A & ~B).sum()), int((~A & B).sum())
    return {"b": x, "c": y, "p": float(binomtest(x, x + y).pvalue) if x + y else 1.0}


def acc(p, mask=None):
    m = slice(None) if mask is None else mask
    return round(100 * float((p[m] == g[m]).mean()), 2)


q = L(d.qwen_zs_label)
fam = {k: mc(q, L(d[c])) for k, c in [("LR", "logreg_pred"), ("SVM", "svm_pred"), ("XGB", "xgb_pred")]}
ps = sorted(fam.items(), key=lambda kv: kv[1]["p"])
adj, run = {}, 0.0
for i, (k, v) in enumerate(ps):
    run = max(run, min(1.0, (len(ps) - i) * v["p"]))
    adj[k] = run
for k in fam:
    fam[k]["p_holm"] = adj[k]
R["H_holm_primary"] = fam

sj = json.load(open(RES / "seed_sensitivity_cv.json"))["results"]
R["B2_seeds_cv"] = {m: {"test_range_pp": round(100 * (max(v["per_seed_test"]) - min(v["per_seed_test"])), 2),
                        "test_min": round(100 * min(v["per_seed_test"]), 2),
                        "test_max": round(100 * max(v["per_seed_test"]), 2),
                        "test_sd_pp": round(100 * v["test_std"], 2),
                        "cv_seed42": round(100 * v["per_seed_cv"][0], 2),
                        "cv_min": round(100 * min(v["per_seed_cv"]), 2),
                        "cv_max": round(100 * max(v["per_seed_cv"]), 2)} for m, v in sj.items()}
S = pd.concat([q] + [L(d[f"qwen_zs_seed100{i}_label"]) for i in (1, 2, 3)], axis=1)
vc = S.apply(lambda r: r.value_counts().tolist(), axis=1)
ties = int(sum(1 for v in vc if len(v) > 1 and v[0] == v[1]))
R["B2_seeds_decoding_qwen"] = {"identical_rows": int((S.nunique(axis=1) == 1).sum()),
                               "per_run_acc": [acc(S.iloc[:, i]) for i in range(4)],
                               "ties_2_2": ties}

abl = {}
for m in ("qwen", "gemma"):
    fw, so, po = L(d[f"{m}_zs_label"]), L(d[f"{m}_sentonly_zs_label"]), L(d[f"{m}_panelonly_zs_label"])
    abl[m] = {
        "acc": {"framework": acc(fw), "sentence_only": acc(so), "panel_only": acc(po), "majority": acc(mv)},
        "fw_vs_sentonly": mc(fw, so), "fw_vs_panelonly": mc(fw, po),
        "unanimous": {"framework": acc(fw, unan), "sentence_only": acc(so, unan), "panel_only": acc(po, unan),
                      "majority": acc(mv, unan), "fw_vs_sentonly": mc(fw, so, unan)},
        "split": {"framework": acc(fw, ~unan), "sentence_only": acc(so, ~unan), "panel_only": acc(po, ~unan),
                  "majority": acc(mv, ~unan), "fw_vs_sentonly": mc(fw, so, ~unan), "fw_vs_majority": mc(fw, mv, ~unan)},
        "panelonly_follows_majority": int((po == mv).sum()),
    }
R["B3_ablation"] = abl

fb = pd.read_csv(RES / "finbert_per_row.csv")
bb = pd.read_csv(RES / "bert_per_row.csv")
assert (fb.sentence == d.sentence).all() and (bb.sentence == d.sentence).all()
pan = {"classical": mv, "bert": L(bb.bert_ft_pred), "finbert": L(fb.finbert_zs_pred)}
slm = {("qwen", "classical"): q, ("qwen", "bert"): L(d.bert_qwen_zs_label), ("qwen", "finbert"): L(d.finbert_qwen_zs_label),
       ("gemma", "classical"): L(d.gemma_zs_label), ("gemma", "bert"): L(d.bert_gemma_zs_label),
       ("gemma", "finbert"): L(d.finbert_gemma_zs_label)}
b4 = {"panel_acc": {k: acc(v) for k, v in pan.items()},
      "panel_vs_panel": {"bert_vs_classical": mc(pan["bert"], pan["classical"]),
                         "finbert_vs_classical": mc(pan["finbert"], pan["classical"]),
                         "finbert_vs_bert": mc(pan["finbert"], pan["bert"])}}
for m in ("qwen", "gemma"):
    b4[m] = {}
    for p in ("classical", "bert", "finbert"):
        s, P = slm[(m, p)], pan[p]
        b4[m][p] = {"acc": acc(s), "follow_pct": round(100 * float((s == P).mean()), 1),
                    "follow_if_panel_right": round(100 * float((s == P)[P == g].mean()), 1),
                    "follow_if_panel_wrong": round(100 * float((s == P)[P != g].mean()), 1),
                    "panel_wrong_n": int((P != g).sum()), "vs_panel": mc(s, P)}
    b4[m]["slm_pairwise"] = {f"{a}_vs_{b}": mc(slm[(m, a)], slm[(m, b)])
                             for a, b in [("classical", "bert"), ("classical", "finbert"), ("bert", "finbert")]}
R["B4_panel_substitution"] = b4

E = RES / "expert_evaluation"
key = pd.read_csv(E / "blinding_key_PRIVATE.csv")
rows = []
for e in (1, 2):
    r = pd.read_excel(E / f"ExpertEvaluation_Expert{e}.xlsx", sheet_name="Ratings")
    r["expert"] = e
    rows.append(r)
X = pd.concat(rows).merge(key, on=["item_id", "condition_order"])
X["trust"] = X.Q4_trust_YES_or_NO.astype(str).str.strip().str.upper().eq("YES").astype(int)
X["unan"] = X.item_id.map(unan)
X["sys_label"] = [L(d[f"{c.split('_')[0]}_zs_label"]).iloc[i] for i, c in zip(X.item_id, X.condition)]
own_col = [c for c in X.columns if c.upper().startswith("Q5")]
X["own"] = X[own_col[0]].astype(str).str.strip().str.lower() if own_col else np.nan
Q = {"Q1": "Q1_factual_accuracy_1to5", "Q2": "Q2_faithfulness_1to5", "Q3": "Q3_actionability_1to5", "Q4": "trust"}
w_u, w_s = NU / N, NS / N
b1 = {"strata_population": {"unanimous": NU, "split": NS}, "systems": {}}
for c in ("qwen_zs", "gemma_zs"):
    s = X[X.condition == c]
    it = s.groupby("item_id").agg(**{k: (v, "mean") for k, v in Q.items()}, u=("unan", "first"))
    U, Sp = it[it.u], it[~it.u]
    out = {}
    for k in Q:
        pooled = float(it[k].mean())
        wtd = w_u * float(U[k].mean()) + w_s * float(Sp[k].mean())
        bs_w, bs_p = [], []
        for _ in range(NB):
            bu = U[k].values[rng.integers(0, len(U), len(U))]
            bsp = Sp[k].values[rng.integers(0, len(Sp), len(Sp))]
            bs_w.append(w_u * bu.mean() + w_s * bsp.mean())
            bs_p.append(np.r_[bu, bsp].mean())
        f = 100 if k == "Q4" else 1
        out[k] = {"pooled": round(f * pooled, 2), "unanimous": round(f * float(U[k].mean()), 2),
                  "split": round(f * float(Sp[k].mean()), 2), "reweighted": round(f * wtd, 2),
                  "pooled_ci": [round(f * x, 2) for x in np.percentile(bs_p, [2.5, 97.5])],
                  "reweighted_ci": [round(f * x, 2) for x in np.percentile(bs_w, [2.5, 97.5])]}
    out["trust_counts"] = {"unanimous": f"{int(s[s.unan].trust.sum())}/{int(s.unan.sum())}",
                           "split": f"{int(s[~s.unan].trust.sum())}/{int((~s.unan).sum())}",
                           "pooled": f"{int(s.trust.sum())}/{len(s)}"}
    if own_col:
        agree = s.own == s.sys_label
        out["by_expert_agreement"] = {k: {"agree": round(float(s[agree][v].mean()) * (100 if k == "Q4" else 1), 2),
                                          "disagree": round(float(s[~agree][v].mean()) * (100 if k == "Q4" else 1), 2),
                                          "n_agree": int(agree.sum()), "n_disagree": int((~agree).sum())}
                                      for k, v in Q.items()}
    b1["systems"][c] = out
kap = {}
wide = X.pivot_table(index=["item_id", "condition"], columns="expert", values=list(Q.values()), aggfunc="first")
for k, v in Q.items():
    kap[k] = {}
    for scope in ("pooled", "qwen_zs", "gemma_zs"):
        ww = wide if scope == "pooled" else wide.xs(scope, level="condition")
        a, b = ww[(v, 1)].astype(int), ww[(v, 2)].astype(int)
        kap[k][scope] = {"cohen": round(cohen_kappa_score(a, b), 3),
                         "quadratic": round(cohen_kappa_score(a, b, weights="quadratic"), 3) if k != "Q4" else None,
                         "exact_agreement": round(float((a == b).mean()), 3)}
b1["kappa"] = kap
im = X.groupby(["item_id", "condition"])[Q["Q2"]].mean().unstack()
b1["wilcoxon_Q2_qwen_vs_gemma"] = float(wilcoxon(im.qwen_zs, im.gemma_zs).pvalue)
R["B1_expert"] = b1

NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(%|percent)?")


def numbers(text):
    out = []
    for m in NUM.finditer(str(text)):
        v = float(m.group(1).replace(",", "."))
        out.append((v, bool(m.group(2))))
    return out


def supported(v, is_pct, pool):
    cands = [v] + ([v / 100] if is_pct else []) + ([v * 100] if (not is_pct and v <= 1) else [])
    return any(abs(c - p) <= max(0.051, 0.006 * abs(p)) for c in cands for p in pool)


def panel_pool(i):
    pool = []
    for m in ("logreg", "svm", "xgb"):
        for k in ("confidence", "entropy", "prob_negative", "prob_neutral", "prob_positive"):
            x = float(d.at[i, f"{m}_{k}"])
            pool += [x, round(100 * x, 1)] if k != "entropy" else [x, round(x, 2), round(x, 3), round(x, 4)]
        feats = str(d.at[i, f"{m}_top_features" if m != "xgb" else "xgb_top_features_global"])
        pool += [float(z) for z in re.findall(r"\d+(?:\.\d+)?", feats)]
    pool += [1, 2, 3]
    return pool


def sent_pool(i):
    return [float(z.replace(",", ".")) for z in re.findall(r"\d+(?:[.,]\d+)?", str(d.at[i, "sentence"]))]


PANEL_NAMES = re.compile(r"logistic|logreg|\bsvm\b|support vector|xgboost|\bxgb\b|panel|classifier|model", re.I)
b5 = {}
for m in ("qwen", "gemma"):
    for cond, col, has_panel in (("framework", f"{m}_zs_explanation", True), ("sentence_only", f"{m}_sentonly_zs_explanation", False)):
        n_claims = n_unsup = rows_unsup = rows_with_num = 0
        lens = []
        for i in range(N):
            txt = str(d.at[i, col])
            lens.append(len(txt))
            pool = sent_pool(i) + (panel_pool(i) if has_panel else [])
            nums = numbers(txt)
            if nums:
                rows_with_num += 1
            uns = [v for v, p in nums if not supported(v, p, pool)]
            n_claims += len(nums)
            n_unsup += len(uns)
            rows_unsup += bool(uns)
        b5[f"{m}_{cond}"] = {"mean_len_chars": round(float(np.mean(lens)), 1),
                             "rows_with_numeric_claim_pct": round(100 * rows_with_num / N, 1),
                             "numeric_claims": n_claims,
                             "unsupported_numeric_claims": n_unsup,
                             "unsupported_claim_rate_pct": round(100 * n_unsup / max(n_claims, 1), 1),
                             "rows_with_unsupported_number_pct": round(100 * rows_unsup / N, 1),
                             "cites_panel_or_model_pct": round(100 * float(d[col].astype(str).str.contains(PANEL_NAMES).mean()), 1),
                             "label_accuracy": acc(L(d[f"{m}_zs_label" if has_panel else f"{m}_sentonly_zs_label"]))}
R["B5_explanations_with_vs_without"] = b5

items = sorted(X.item_id.unique())
R["D1_items_label_acc"] = {c: {"all": acc(L(d[c]).iloc[items].reset_index(drop=True).set_axis(items)),
                               } for c in ()}
sub = {}
for c in ("qwen_zs_label", "qwen_sentonly_zs_label", "gemma_zs_label", "gemma_sentonly_zs_label"):
    p = L(d[c])
    sub[c] = {"all": round(100 * float((p[items] == g[items]).mean()), 1),
              "unanimous_correct": int(((p == g) & unan)[items].sum()),
              "split_correct": int(((p == g) & ~unan)[items].sum())}
R["D1_items_label_acc"] = sub

(OUT / "b_results.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
