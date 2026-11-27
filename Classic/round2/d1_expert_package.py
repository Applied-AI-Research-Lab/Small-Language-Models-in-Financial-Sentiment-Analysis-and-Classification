#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEST_CSV = ROOT / "Datasets" / "financial_phrasebank" / "test.csv"
R1_MANIFEST = ROOT / "Classic" / "results" / "expert_evaluation" / "sample_manifest.json"
OUT = ROOT / "Classic" / "results" / "round2" / "d1_expert"
KEY = OUT / "d1_blinding_key_PRIVATE.csv"
PLAN = OUT / "d1_analysis_plan.md"
SEED = 2026
N_BOOT = 10000
POP = {"unanimous": 310, "split": 30}

sys.path.insert(0, str(ROOT))
from Classic.expert_evaluation_package import render_signals

INSTRUCTIONS = """\
EXPERT EVALUATION: TWO EXPLANATIONS OF THE SAME SENTIMENT DECISION

Each row is one sentence from a financial news corpus. You see:
  - the sentence,
  - the signals of three statistical models (Logistic Regression, Linear SVM, XGBoost),
  - two explanations, A and B, each with the sentiment label it argues for.
A and B were written by two variants of the same automated system. Their order is random.

For each row, please fill in five dropdowns:
  A_Q1 / B_Q1  FACTUAL ACCURACY - Are ALL factual claims in the explanation correct?
               This covers claims about the sentence and, where the explanation cites
               them, claims about the model signals (check them against the signal block).
               1 = mostly wrong ... 5 = all correct.
  A_Q2 / B_Q2  FAITHFULNESS - Does the reasoning actually support the label shown?
               1 = contradicts it ... 5 = fully supports it.
  PREFERENCE   Which explanation better justifies its label for a financial analyst?
               A, B, or Equal.
  COMMENTS     Optional. Leave blank unless something is clearly wrong.

Judge the explanations, not whether you agree with the label.
There are 60 rows. Expect about 1 to 1.5 hours in total; you may stop and continue later.
"""

COLS = ["row", "sentence", "panel_signals", "A_label", "A_explanation", "B_label", "B_explanation",
        "A_Q1_factual_accuracy_1to5", "A_Q2_faithfulness_1to5",
        "B_Q1_factual_accuracy_1to5", "B_Q2_faithfulness_1to5",
        "PREFERENCE_A_B_Equal", "comments"]
WIDTH = {"row": 6, "sentence": 45, "panel_signals": 50, "A_label": 10, "A_explanation": 55,
         "B_label": 10, "B_explanation": 55, "A_Q1_factual_accuracy_1to5": 11,
         "A_Q2_faithfulness_1to5": 11, "B_Q1_factual_accuracy_1to5": 11,
         "B_Q2_faithfulness_1to5": 11, "PREFERENCE_A_B_Equal": 12, "comments": 30}


def load():
    d = pd.read_csv(TEST_CSV)
    ids = json.load(open(R1_MANIFEST))["item_ids"]
    assert len(ids) == 60
    votes = d[["logreg_pred", "svm_pred", "xgb_pred"]].apply(lambda s: s.str.lower())
    d["stratum"] = np.where(votes.nunique(axis=1) == 1, "unanimous", "split")
    return d, ids


def build():
    d, ids = load()
    rng = random.Random(SEED)
    order = ids[:]
    rng.shuffle(order)
    a_is_fw = [True] * 30 + [False] * 30
    rng.shuffle(a_is_fw)
    rows, key = [], []
    for k, (i, fw_first) in enumerate(zip(order, a_is_fw), start=1):
        r = d.loc[i]
        fw = (str(r.qwen_zs_label).lower(), str(r.qwen_zs_explanation))
        so = (str(r.qwen_sentonly_zs_label).lower(), str(r.qwen_sentonly_zs_explanation))
        A, B = (fw, so) if fw_first else (so, fw)
        rows.append({"row": k, "sentence": str(r.sentence), "panel_signals": render_signals(r),
                     "A_label": A[0], "A_explanation": A[1], "B_label": B[0], "B_explanation": B[1],
                     **{c: "" for c in COLS[7:]}})
        key.append({"row": k, "item_id": int(i), "A": "framework" if fw_first else "sentence_only",
                    "B": "sentence_only" if fw_first else "framework", "stratum": r.stratum,
                    "gold": str(r.label).lower(), "fw_label": fw[0], "so_label": so[0]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(key).to_csv(KEY, index=False)
    master = OUT / "D1_ExpertEvaluation.xlsx"
    write_workbook(master, rows)
    for e in (1, 2):
        shutil.copyfile(master, OUT / f"D1_ExpertEvaluation_Expert{e}.xlsx")
    write_plan(master, pd.DataFrame(key))
    print(f"written to {OUT}: workbook (60 rows), 2 expert copies, private key, analysis plan")


def write_workbook(path, rows):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "Instructions"
    ws.column_dimensions["A"].width = 100
    for i, line in enumerate(INSTRUCTIONS.strip().splitlines(), start=1):
        c = ws.cell(row=i, column=1, value=line)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if i == 1:
            c.font = Font(bold=True, size=13)
    ws = wb.create_sheet("Ratings")
    thin = Side(style="thin", color="BBBBBB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for j, col in enumerate(COLS, start=1):
        c = ws.cell(row=1, column=j, value=col)
        c.fill = PatternFill("solid", fgColor="1F4E78")
        c.font = Font(bold=True, color="FFFFFF")
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = WIDTH[col]
    for i, row in enumerate(rows, start=2):
        for j, col in enumerate(COLS, start=1):
            c = ws.cell(row=i, column=j, value=row[col])
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = border
        ws.row_dimensions[i].height = 170
    ws.freeze_panes = "B2"
    n = len(rows) + 1
    dv15 = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    dvp = DataValidation(type="list", formula1='"A,B,Equal"', allow_blank=True)
    ws.add_data_validation(dv15)
    ws.add_data_validation(dvp)
    for col in COLS[7:11]:
        L = get_column_letter(COLS.index(col) + 1)
        dv15.add(f"{L}2:{L}{n}")
    L = get_column_letter(COLS.index("PREFERENCE_A_B_Equal") + 1)
    dvp.add(f"{L}2:{L}{n}")
    wb.properties.creator = "anonymous"
    wb.save(path)


def write_plan(master, key):
    sha = hashlib.sha256(master.read_bytes()).hexdigest()
    same = int((key.fw_label == key.so_label).sum())
    PLAN.write_text(f"""# D1 analysis plan (fixed before any rating is returned)

Frozen: {dt.datetime.now().isoformat(timespec='seconds')}
Workbook SHA-256: `{sha}`
Analysis code: `Classic/round2/d1_expert_package.py --analyze` (this file's version at the
time above). Nothing below is changed after ratings are returned.

**Question (Reviewer 2, 2.2):** for the same SLM (Qwen3.5-4B, zero-shot), does giving it the
classical panel signals change the quality of its explanation?

**Units:** 60 items, 2 raters; per item, the framework explanation versus the sentence-only
explanation, side by side, with random A/B order (30/30).

**Primary endpoint: preference.**
- Per item, s_i = mean over the two raters of +1 (framework preferred), -1 (sentence-only
  preferred), 0 (Equal).
- Test: two-sided exact sign test on the items with s_i != 0.
- Reported: the share of votes per option; the item-level mean of s pooled over the
  sample; and the same mean reweighted to the test set (310 unanimous / 30 split). The
  reweighted value has a 95% CI from an item-clustered bootstrap resampled within strata
  ({N_BOOT} resamples).

**Secondary endpoints:** Q1 factual accuracy and Q2 faithfulness.
- Per item: framework minus sentence-only, averaged over the two raters.
- Test: two-sided paired Wilcoxon signed-rank (zero differences handled by the Pratt
  method). Holm adjustment across the two secondary tests.
- Means are reported pooled and reweighted.

**Label-correctness control:** the same analyses restricted to the {same} items on which
both conditions emit the same label. On the other items the label itself differs, and Q2
judges support for different labels.

**Agreement:** Cohen's kappa between the raters, for Q1, Q2 and preference
(quadratic-weighted for Q1/Q2).

**Disclosures for the paper:**
- The raters saw the panel block for both conditions.
- Framework explanations cite panel models, so condition blinding cannot be complete.
- The raters were not told the hypothesis.
- The same raters rated the framework explanations of these 60 items in an earlier study.
- Mean explanation length: framework 365 characters, sentence-only 259.

**Reporting rule:** the results are reported whatever their direction.
""")


def analyze():
    from scipy.stats import binomtest, wilcoxon
    from sklearn.metrics import cohen_kappa_score
    key = pd.read_csv(KEY)
    R = []
    for e in (1, 2):
        w = pd.read_excel(OUT / f"D1_ExpertEvaluation_Expert{e}.xlsx", sheet_name="Ratings")
        w["expert"] = e
        R.append(w)
    X = pd.concat(R).merge(key, on="row")
    for side in ("A", "B"):
        for q, col in (("Q1", f"{side}_Q1_factual_accuracy_1to5"), ("Q2", f"{side}_Q2_faithfulness_1to5")):
            X[f"{side}_{q}"] = pd.to_numeric(X[col], errors="coerce")
    for q in ("Q1", "Q2"):
        X[f"fw_{q}"] = np.where(X.A == "framework", X[f"A_{q}"], X[f"B_{q}"])
        X[f"so_{q}"] = np.where(X.A == "framework", X[f"B_{q}"], X[f"A_{q}"])
    pref = X.PREFERENCE_A_B_Equal.astype(str).str.strip().str.upper()
    X["pref"] = np.select([pref == "EQUAL", ((pref == "A") & (X.A == "framework")) | ((pref == "B") & (X.B == "framework"))],
                          [0, 1], default=-1)
    missing = X[["fw_Q1", "fw_Q2", "so_Q1", "so_Q2"]].isna().sum().sum() + (~pref.isin(["A", "B", "EQUAL"])).sum()
    rng = np.random.default_rng(SEED)

    def summarize(sub):
        it = sub.groupby("item_id").agg(s=("pref", "mean"), stratum=("stratum", "first"),
                                        d1=("fw_Q1", "mean"), so1=("so_Q1", "mean"),
                                        d2=("fw_Q2", "mean"), so2=("so_Q2", "mean"))
        it["d1"] -= it.so1
        it["d2"] -= it.so2
        nz = it.s[it.s != 0]
        out = {"n_items": len(it),
               "votes": {"framework": int((sub.pref == 1).sum()), "sentence_only": int((sub.pref == -1).sum()),
                         "equal": int((sub.pref == 0).sum())},
               "sign_test": {"framework_items": int((nz > 0).sum()), "sentence_only_items": int((nz < 0).sum()),
                             "p": float(binomtest(int((nz > 0).sum()), len(nz)).pvalue) if len(nz) else 1.0},
               "pref_mean_pooled": round(float(it.s.mean()), 3)}
        groups = {k: g for k, g in it.groupby("stratum")}
        if set(groups) == {"unanimous", "split"}:
            wsum = sum(POP.values())
            rw = lambda f: sum(POP[k] / wsum * f(groups[k]) for k in groups)
            out["pref_mean_reweighted"] = round(rw(lambda g: g.s.mean()), 3)
            bs = [sum(POP[k] / wsum * g.s.values[rng.integers(0, len(g), len(g))].mean() for k, g in groups.items())
                  for _ in range(N_BOOT)]
            out["pref_mean_reweighted_ci"] = [round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]
            out["by_stratum_pref_mean"] = {k: round(float(g.s.mean()), 3) for k, g in groups.items()}
        ps = []
        for q, dcol in (("Q1", "d1"), ("Q2", "d2")):
            fwm = round(float(sub[f"fw_{q}"].mean()), 2)
            som = round(float(sub[f"so_{q}"].mean()), 2)
            p = float(wilcoxon(it[dcol], zero_method="pratt").pvalue) if (it[dcol] != 0).any() else 1.0
            ps.append(p)
            out[q] = {"framework": fwm, "sentence_only": som, "p_wilcoxon": p}
        order = np.argsort(ps)
        adj, run = [0, 0], 0
        for rank, j in enumerate(order):
            run = max(run, min(1, (2 - rank) * ps[j]))
            adj[j] = run
        out["Q1"]["p_holm"], out["Q2"]["p_holm"] = adj
        return out

    res = {"missing_cells": int(missing), "all_items": summarize(X),
           "same_label_items": summarize(X[X.fw_label == X.so_label])}
    w = X.pivot_table(index="row", columns="expert", values=["fw_Q1", "so_Q1", "fw_Q2", "so_Q2", "pref"], aggfunc="first")
    kap = {}
    for q in ("Q1", "Q2"):
        a = pd.concat([w[(f"fw_{q}", 1)], w[(f"so_{q}", 1)]]).astype(int)
        b = pd.concat([w[(f"fw_{q}", 2)], w[(f"so_{q}", 2)]]).astype(int)
        kap[q] = {"cohen": round(cohen_kappa_score(a, b), 3),
                  "quadratic": round(cohen_kappa_score(a, b, weights="quadratic"), 3)}
    kap["preference"] = {"cohen": round(cohen_kappa_score(w[("pref", 1)], w[("pref", 2)]), 3)}
    res["kappa"] = kap
    (OUT / "d1_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else build()
