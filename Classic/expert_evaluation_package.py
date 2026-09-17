#!/usr/bin/env python3
"""
Expert evaluation package — D2a sampling + blinded workbooks (R1.4/R2.5).

Design (statistically justified, see Reviewer replies):
  N = 60 items = ALL 30 split-panel rows (census) + 30 unanimous rows sampled
  proportionally by true class (19 neutral / 7 positive / 4 negative of 310).
  Two blinded domain experts (business-department professors) rate each of
  the 120 (item x system) units on three quality dimensions plus trust and
  their own sentiment label.

The evaluation instrument is ONE merged Excel workbook containing all 60
items (120 rating rows: every item shows both SLM explanations as blinded
"Explanation 1"/"Explanation 2"). Every evaluator rates the SAME single
workbook, so all items receive identical 2-rater coverage for BOTH systems.

  ExpertEvaluation.xlsx           — the master template (all 60 items)
  ExpertEvaluation_Expert1.xlsx    — two identical named copies, one per
  ExpertEvaluation_Expert2.xlsx      evaluator; each expert fills in and
                                     returns their own copy so ratings can
                                     be attributed to a rater (required for
                                     inter-rater reliability)

Raters: two professors from business departments (domain experts for the
financial decision-support context). With two raters the inter-rater
statistic is Cohen's/Fleiss' kappa over the 120 rating units (both
equivalent for two raters with pooled marginals); disagreements cannot be
resolved by majority vote, so per-rater means are also reported.

Blinding rules:
  - The identity of the SLM that produced each explanation is hidden:
    the two systems (Qwen3.5-4B zero-shot and Gemma3-4B zero-shot) appear
    only as neutral "Explanation 1" / "Explanation 2", randomly ordered
    per item.
  - The three classical panel members (Logistic Regression, Linear SVM,
    XGBoost) are shown with their real names: they are the shared framework
    input visible in every explanation, not the system under test.
  - No model names ("Qwen", "Gemma"), no condition labels ("calibrated",
    "zero-shot"), no ground-truth label anywhere in the workbook.

Each rating row contains:
  item_id, sentence, panel signals block (predictions/confidence/features),
  explanation text, final label, and four rating fields:
    Q1 factual accuracy of cited numbers/features (1-5)
    Q2 faithfulness: explanation supports the emitted label (1-5)
    Q3 actionability of the recommendation (1-5)
    Q4 overall judgment: would you trust this in a BI report? (yes/no)
  Plus free-text comments.

Usage:
  python3 Classic/expert_evaluation_package.py                 # generate workbooks
  python3 Classic/expert_evaluation_package.py --analyze <dir-or-file>
      # <dir> = directory with the returned ExpertEvaluation_Expert{1,2,3}.xlsx
      # <file> = single CSV/XLSX with an 'expert' column
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
OUT_DIR = Path("Classic/results/expert_evaluation")
SEED = 42
N_UNANIMOUS = 30          # proportional draw from the 310 unanimous rows
N_TOTAL = 60
CLASSES = ("negative", "neutral", "positive")

# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def draw_sample(df: pd.DataFrame) -> pd.DataFrame:
    """All split rows (census) + proportional unanimous draw, seeded."""
    preds = df[["logreg_pred", "svm_pred", "xgb_pred"]].astype(str)
    is_unanimous = preds.nunique(axis=1) == 1

    split = df[~is_unanimous]
    unan = df[is_unanimous]
    assert len(split) == 30, f"expected 30 split rows, got {len(split)}"

    rng = random.Random(SEED)
    # proportional allocation over true classes in the unanimous stratum
    counts = unan["label"].str.lower().value_counts()
    alloc = {c: round(n / len(unan) * N_UNANIMOUS) for c, n in counts.items()}
    # fix rounding drift toward the largest strata
    while sum(alloc.values()) != N_UNANIMOUS:
        if sum(alloc.values()) < N_UNANIMOUS:
            alloc[max(alloc, key=alloc.get)] += 1
        else:
            alloc[max(alloc, key=alloc.get)] -= 1

    picked = []
    for cls, n in alloc.items():
        pool = unan[unan["label"].str.lower() == cls].index.tolist()
        picked.extend(rng.sample(pool, n))
    unan_sample = unan.loc[sorted(picked)]

    sample = pd.concat([split, unan_sample])
    sample = sample.sort_index()
    print(f"sample: {len(split)} split (census) + {len(unan_sample)} unanimous "
          f"(alloc={alloc}) = {len(sample)} items")
    return sample


# ──────────────────────────────────────────────────────────────────────────────
# Signal block rendering (genericized, blinded)
# ──────────────────────────────────────────────────────────────────────────────

def render_signals(row) -> str:
    """Panel signals block with real member names (shared framework input;
    the system under test — the SLM that wrote the explanation — is blinded)."""
    def pct(x): return f"{float(x) * 100:.1f}%"
    def ent(x): return f"{float(x):.4f}"
    lines = []
    for name, pfx in (("LOGISTIC REGRESSION", "logreg"), ("LINEAR SVM", "svm"), ("XGBOOST", "xgb")):
        feat_col = "xgb_top_features_global" if pfx == "xgb" else f"{pfx}_top_features"
        lines.append(
            f"{name}  prediction: {row[f'{pfx}_pred']}   confidence: {pct(row[f'{pfx}_confidence'])}   "
            f"entropy: {ent(row[f'{pfx}_entropy'])}\n"
            f"        probabilities: negative {pct(row[f'{pfx}_prob_negative'])} | "
            f"neutral {pct(row[f'{pfx}_prob_neutral'])} | "
            f"positive {pct(row[f'{pfx}_prob_positive'])}\n"
            f"        cited features: {row[feat_col]}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Workbook generation
# ──────────────────────────────────────────────────────────────────────────────

INSTRUCTIONS = """\
HUMAN EVALUATION OF AUTOMATED FINANCIAL-SENTIMENT EXPLANATIONS

You are reviewing outputs of an automated financial sentiment-analysis system.
For each item you are given:
  - the original sentence from a financial news corpus,
  - the advisory-panel signals computed by three statistical models
    (Logistic Regression, Linear SVM, XGBoost),
  - TWO different system variants, each consisting of: a final sentiment
    label, a natural-language explanation of the verdict, and a business
    recommendation ("Explanation 1" and "Explanation 2" — their identity
    is hidden, and their order is randomized per item).

Judge each explanation independently on its own merits. The two variants
at the same item come from different (blinded) systems; do not try to
guess which is which, and do not let one variant anchor your rating of
the other.

Rate each explanation on four questions (scale 1-5 unless stated):
  Q1  FACTUAL ACCURACY - Are the numbers, feature words, and panel signals
      cited in the explanation correct for this item (matches the panel
      block shown)? 1 = mostly wrong ... 5 = all correct.
  Q2  FAITHFULNESS - Does the reasoning in the explanation actually support
      the final label? 1 = contradicts ... 5 = fully supports.
  Q3  ACTIONABILITY - Is the business recommendation concrete and usable by
      a financial decision-maker? 1 = generic/no value ... 5 = directly usable.
  Q4  TRUST (yes/no) - Would you accept this output in an internal BI report
      without re-checking? Enter YES or NO.
  Q5  YOUR OWN LABEL (negative/neutral/positive) - What is YOUR sentiment
      label for the SENTENCE ITSELF, judged from the sentence alone and
      BEFORE being influenced by either system variant? Answer this
      independently of the two explanations shown: if both variants say
      "positive" but you read the sentence as neutral, write neutral.
  COMMENTS - FULLY OPTIONAL: leave blank. Only use it if something is
      notably wrong (e.g., a cited number that does not match the panel
      block); never write anything for a typical item. All four ratings
      (Q1-Q5) are quick dropdown selections - the whole item takes
      about 1-2 minutes.

Notes:
  - Some sentences are ambiguous; judge the explanation's quality, not
    whether you agree with the label itself.
  - Expect roughly 2 minutes per explanation.
  - The workbook contains 60 items; you may complete it in one or more
    sittings. Each item needs only five dropdown selections.
"""

RATING_COLUMNS = [
    "item_id", "condition_order", "explanation_shown",
    "sentence", "panel_signals", "final_label", "explanation_text",
    "Q1_factual_accuracy_1to5", "Q2_faithfulness_1to5",
    "Q3_actionability_1to5", "Q4_trust_YES_or_NO",
    "Q5_your_own_label_for_the_sentence", "comments",
]


def build_rating_rows(df_sample: pd.DataFrame) -> list[dict]:
    """One rating row per (item x SLM condition), both deployed SLMs per item.

    Conditions: the two deployed zero-shot systems — Qwen3.5-4B (qwen_zs_*)
    and Gemma3-4B (gemma_zs_*) — shown to experts with neutral, per-item
    randomized labels "Explanation 1" / "Explanation 2".
    """
    rng = random.Random(SEED)
    rows = []
    for idx, r in df_sample.iterrows():
        conds = []
        if pd.notna(r.get("qwen_zs_explanation")):
            conds.append(("qwen_zs", str(r["qwen_zs_label"]), r["qwen_zs_explanation"],
                          r.get("qwen_zs_recommendation")))
        if pd.notna(r.get("gemma_zs_explanation")):
            conds.append(("gemma_zs", str(r["gemma_zs_label"]), r["gemma_zs_explanation"],
                          r.get("gemma_zs_recommendation")))
        assert len(conds) == 2, f"item {idx}: both SLM explanations required"
        rng.shuffle(conds)
        for order, (system, label, expl, rec) in enumerate(conds, start=1):
            text = f"{expl}\n\nRECOMMENDATION: {rec}" if pd.notna(rec) else str(expl)
            rows.append({
                "item_id": int(idx),
                "condition_order": order,
                "explanation_shown": f"Explanation {order}",
                "sentence": str(r["sentence"]),
                "panel_signals": render_signals(r),
                "final_label": str(label).lower(),
                "explanation_text": text,
                "Q1_factual_accuracy_1to5": "",
                "Q2_faithfulness_1to5": "",
                "Q3_actionability_1to5": "",
                "Q4_trust_YES_or_NO": "",
                "Q5_your_own_label_for_the_sentence": "",
                "comments": "",
            })
    # global shuffle of presentation order (decoupled from item_id)
    rng.shuffle(rows)
    return rows


def write_workbook(path: Path, rows: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    # ── Sheet 1: instructions ──
    ws_info = wb.active
    ws_info.title = "Instructions"
    ws_info.column_dimensions["A"].width = 100
    ws_info["A1"] = "HUMAN EVALUATION — FINANCIAL SENTIMENT EXPLANATIONS"
    ws_info["A1"].font = Font(bold=True, size=14)
    for i, line in enumerate(INSTRUCTIONS.strip().splitlines(), start=3):
        c = ws_info.cell(row=i, column=1, value=line)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws_info.row_dimensions[i].height = 14

    # ── Sheet 2: ratings ──
    ws = wb.create_sheet("Ratings")
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF")
    thin = Side(style="thin", color="BBBBBB")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for j, col in enumerate(RATING_COLUMNS, start=1):
        c = ws.cell(row=1, column=j, value=col)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = {
            "item_id": 10, "condition_order": 10, "explanation_shown": 14,
            "sentence": 60, "panel_signals": 55, "final_label": 12,
            "explanation_text": 70,
            "Q1_factual_accuracy_1to5": 13, "Q2_faithfulness_1to5": 13,
            "Q3_actionability_1to5": 13, "Q4_trust_YES_or_NO": 13,
            "Q5_your_own_label_for_the_sentence": 15, "comments": 40,
        }[col]

    for i, row in enumerate(rows, start=2):
        for j, col in enumerate(RATING_COLUMNS, start=1):
            c = ws.cell(row=i, column=j, value=row[col])
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = border
        ws.row_dimensions[i].height = 150

    # data validation for the rating columns
    from openpyxl.worksheet.datavalidation import DataValidation
    dv_15 = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    dv_yn = DataValidation(type="list", formula1='"YES,NO"', allow_blank=True)
    dv_lab = DataValidation(type="list", formula1='"negative,neutral,positive"', allow_blank=True)
    ws.add_data_validation(dv_15)
    ws.add_data_validation(dv_yn)
    ws.add_data_validation(dv_lab)
    for col_letter in ("G", "H", "I"):
        dv_15.add(f"{col_letter}2:{col_letter}{len(rows) + 1}")
    dv_yn.add(f"K2:K{len(rows) + 1}")
    dv_lab.add(f"L2:L{len(rows) + 1}")

    wb.save(path)
    print(f"written: {path} ({len(rows)} rating rows)")


def main_generate() -> None:
    df = pd.read_csv(TEST_CSV)
    sample = draw_sample(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rating_rows(sample)

    # one merged master workbook (all 60 items, 120 rating rows) + two
    # identical named copies so each evaluator returns an attributable file
    write_workbook(OUT_DIR / "ExpertEvaluation.xlsx", rows)
    import shutil
    for i in (1, 2):
        dst = OUT_DIR / f"ExpertEvaluation_Expert{i}.xlsx"
        shutil.copyfile(OUT_DIR / "ExpertEvaluation.xlsx", dst)
        print(f"written: {dst} (identical copy for expert {i})")

    # provenance manifest (NOT sent to experts; used for analysis + blinding key)
    manifest = {
        "seed": SEED,
        "n_items": int(len(sample)),
        "n_split_census": 30,
        "n_unanimous": int(len(sample) - 30),
        "conditions_blinded": {
            "Explanation 1": "per-item randomized assignment — see blinding key",
            "Explanation 2": "per-item randomized assignment — see blinding key",
        },
        "systems_under_test": {
            "qwen_zs": "Qwen3.5-4B zero-shot (original signals)",
            "gemma_zs": "Gemma3-4B zero-shot (original signals)",
        },
        "item_ids": [int(i) for i in sample.index],
        "rating_row_count": len(rows),
    }
    with open(OUT_DIR / "sample_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"written: {OUT_DIR}/sample_manifest.json")

    # per-item key mapping (item_id, condition_order) -> system, kept private.
    # Rebuilt with the same seeded per-item shuffle as build_rating_rows, so
    # the (item, order) -> system assignment is exact.
    rng_key = random.Random(SEED)
    key_rows = []
    for idx, r in sample.iterrows():
        conds = [("qwen_zs",), ("gemma_zs",)]
        rng_key.shuffle(conds)
        for order, (system,) in enumerate(conds, start=1):
            key_rows.append({"item_id": int(idx), "condition_order": order,
                             "condition": system})
    key = pd.DataFrame(key_rows)
    key.to_csv(OUT_DIR / "blinding_key_PRIVATE.csv", index=False)
    print(f"written: {OUT_DIR}/blinding_key_PRIVATE.csv (do NOT send to experts)")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def analyze(results_csv: Path) -> None:
    """Analyze the completed expert workbooks (two raters).

    `results_csv` may be a directory containing ExpertEvaluation_Expert{i}.xlsx
    / .csv files — the two returned per-expert workbooks are merged
    automatically with rater attribution — or a single file with an
    `expert` column.
    """
    key = pd.read_csv(OUT_DIR / "blinding_key_PRIVATE.csv")

    frames = []
    p = Path(results_csv)
    if p.is_dir():
        files = sorted(p.glob("ExpertEvaluation_Expert*.*"))
        assert files, f"no ExpertEvaluation_Expert* files in {p}"
        for f in files:
            d = (pd.read_excel(f, sheet_name="Ratings")
                 if f.suffix.lower() == ".xlsx" else pd.read_csv(f))
            stem = f.stem  # e.g. ExpertEvaluation_Expert1
            d["expert"] = stem.split("Expert")[-1] if "Expert" in stem else stem
            frames.append(d)
    else:
        d = (pd.read_excel(p, sheet_name="Ratings")
             if p.suffix.lower() == ".xlsx" else pd.read_csv(p))
        if "expert" not in d.columns:
            raise ValueError("single-file analysis requires an 'expert' column; "
                             "pass the directory of the three Expert*.xlsx files instead")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # merge blinding key to recover conditions
    df = df.merge(key, on=["item_id", "condition_order"], how="left")
    assert df["condition"].notna().all(), "blinding key missing rows"

    experts = sorted(df["expert"].unique())
    print(f"rows: {len(df)} | unique items: {df['item_id'].nunique()} | experts: {experts}")
    for col in ("Q1_factual_accuracy_1to5", "Q2_faithfulness_1to5", "Q3_actionability_1to5"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trust"] = df["Q4_trust_YES_or_NO"].astype(str).str.upper().eq("YES")

    out = {"n_experts": len(experts), "experts": experts}

    # ── inter-rater reliability: Fleiss' kappa over (item, condition) units ──
    def fleiss_kappa(units: list[list[int]], k_categories: int) -> float:
        # units: list of rating vectors (one per rater), integer categories 0..k-1
        n = len(units[0])          # raters per unit
        N = len(units)
        if n < 2:
            return float("nan")
        p_j = [0.0] * k_categories
        for u in units:
            for v in u:
                p_j[v] += 1
        total = sum(p_j)
        p_j = [x / (N * n) for x in p_j]
        P_i = []
        for u in units:
            counts = [0] * k_categories
            for v in u:
                counts[v] += 1
            P_i.append((sum(c * c for c in counts) - n) / (n * (n - 1)))
        P_bar = sum(P_i) / N
        P_e = sum(p * p for p in p_j)
        return (P_bar - P_e) / (1 - P_e) if P_e < 1 else float("nan")

    for metric_col, name, kcat in (
        ("Q1_factual_accuracy_1to5", "Q1", 5),
        ("Q2_faithfulness_1to5", "Q2", 5),
        ("Q3_actionability_1to5", "Q3", 5),
    ):
        piv = df.pivot_table(index=["item_id", "condition"], columns="expert",
                             values=metric_col)
        piv = piv.dropna()
        if piv.shape[1] >= 2 and len(piv) > 0:
            units = [[int(v) - 1 for v in row] for row in piv.values]
            kap = fleiss_kappa(units, kcat)
            out[f"fleiss_kappa_{name}"] = kap
            print(f"Fleiss kappa ({name}, {piv.shape[1]} raters, {len(piv)} units): {kap:.3f}")

    # binary trust kappa
    piv = df.pivot_table(index=["item_id", "condition"], columns="expert", values="trust")
    piv = piv.dropna()
    if piv.shape[1] >= 2 and len(piv) > 0:
        units = [[int(v) for v in row] for row in piv.values]
        kap = fleiss_kappa(units, 2)
        out["fleiss_kappa_Q4_trust"] = kap
        print(f"Fleiss kappa (Q4 trust): {kap:.3f}")

    # ── per-system summaries (all experts pooled) ──
    for cond, g in df.groupby("condition"):
        out[cond] = {
            "n_ratings": len(g),
            "Q1_mean": float(g["Q1_factual_accuracy_1to5"].mean()),
            "Q2_mean": float(g["Q2_faithfulness_1to5"].mean()),
            "Q3_mean": float(g["Q3_actionability_1to5"].mean()),
            "trust_rate": float(g["trust"].mean()),
            "trust_ci95": wilson(int(g["trust"].sum()), len(g)),
        }
        print(f"\n== {cond} (pooled over experts) ==")
        for m in ("Q1_mean", "Q2_mean", "Q3_mean"):
            print(f"  {m}: {out[cond][m]:.2f}")
        lo, hi = out[cond]["trust_ci95"]
        print(f"  trust: {out[cond]['trust_rate']*100:.1f}% (95% CI {lo*100:.1f}-{hi*100:.1f}%)")

    # per-expert means (screening for divergent raters)
    out["per_expert"] = {}
    for exp, g in df.groupby("expert"):
        out["per_expert"][exp] = {
            "Q1_mean": float(g["Q1_factual_accuracy_1to5"].mean()),
            "Q2_mean": float(g["Q2_faithfulness_1to5"].mean()),
            "Q3_mean": float(g["Q3_actionability_1to5"].mean()),
            "trust_rate": float(g["trust"].mean()),
            "n": len(g),
        }
        print(f"expert {exp}: Q1={out['per_expert'][exp]['Q1_mean']:.2f} "
              f"Q2={out['per_expert'][exp]['Q2_mean']:.2f} "
              f"Q3={out['per_expert'][exp]['Q3_mean']:.2f} "
              f"trust={out['per_expert'][exp]['trust_rate']*100:.0f}% (n={len(g)})")

    # ── paired Qwen-vs-Gemma comparison (item-mean over experts, Wilcoxon) ──
    item_mean = df.pivot_table(index="item_id", columns="condition",
                               values="Q2_faithfulness_1to5")
    if item_mean.shape[1] == 2 and item_mean.notna().all(axis=1).any():
        pair = item_mean.dropna()
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1])
            out["wilcoxon_Q2_systems"] = {"stat": float(stat), "p": float(p)}
            print(f"\nWilcoxon (Q2, {pair.columns[0]} vs {pair.columns[1]}): p={p:.4f}")
        except ImportError:
            print("\n(scipy unavailable — Wilcoxon skipped; report means with CIs)")

    # ── expert's own labels (Q5): agreement with gold and with the systems ──
    # (item_id is the positional row index of test.csv). With two raters, a
    # per-item mode tie is arbitrary, so only items where BOTH raters give
    # the same own-label are used; the raw rater-agreement rate is reported.
    gold = pd.read_csv(TEST_CSV)["label"].astype(str).str.lower()
    q5_one_row_per_item = df.drop_duplicates(subset=["expert", "item_id"])
    n_raters = q5_one_row_per_item["expert"].nunique()
    per_rater = {exp: g.set_index("item_id")["Q5_your_own_label_for_the_sentence"].astype(str).str.lower()
                 for exp, g in q5_one_row_per_item.groupby("expert")}
    item_ids_q5 = sorted(set.intersection(*[set(s.dropna().index) for s in per_rater.values()])) \
        if len(per_rater) > 1 else sorted(per_rater[list(per_rater)[0]].dropna().index)
    if len(per_rater) > 1:
        agree_items = [i for i in item_ids_q5
                       if all(per_rater[e][i] == per_rater[list(per_rater)[0]][i] for e in per_rater)]
        out["q5_rater_agreement"] = {
            "n_items": len(item_ids_q5),
            "n_agree": len(agree_items),
            "agreement_rate": len(agree_items) / len(item_ids_q5) if item_ids_q5 else None,
        }
        print(f"Q5 rater agreement (own label): {len(agree_items)}/{len(item_ids_q5)} "
              f"({out['q5_rater_agreement']['agreement_rate']*100:.1f}%)")
    else:
        agree_items = item_ids_q5
    valid_q5 = pd.Series({i: per_rater[list(per_rater)[0]][i] for i in agree_items})
    n_q5_items = len(valid_q5)
    if n_q5_items:
        agree_gold = sum(valid_q5[i] == gold.loc[i] for i in valid_q5.index)
        out["expert_vs_gold"] = {
            "n_items": int(n_q5_items),
            "accuracy": float(agree_gold / n_q5_items),
        }
        print(f"Expert own-label vs gold (both agree): {agree_gold}/{n_q5_items} "
              f"({agree_gold/n_q5_items*100:.1f}%)")
        for cond, g in df.groupby("condition"):
            # consensus own-label (both raters agree) vs this system's label
            sys_label = g.groupby("item_id")["final_label"].agg(
                lambda s: s.astype(str).str.lower().mode().iat[0])
            agree_sys = sum(valid_q5[i] == sys_label.loc[i]
                            for i in valid_q5.index if i in sys_label.index)
            n_own = len(valid_q5)
            out.setdefault("expert_vs_system", {})[cond] = {
                "n_items": int(n_own),
                "agreement": float(agree_sys / n_own) if n_own else None,
            }
            print(f"Expert own-label vs {cond}: {agree_sys}/{n_own}")

        # explanation-quality split by expert-consensus agreement with the
        # system label (uses the row-level Q5 of the rater who rated it)
        for cond, g in df.groupby("condition"):
            g = g.copy()
            own = g["Q5_your_own_label_for_the_sentence"].astype(str).str.lower()
            same = own == g["final_label"].astype(str).str.lower()
            split = {}
            for metric_col, name in (("Q1_factual_accuracy_1to5", "Q1"),
                                     ("Q2_faithfulness_1to5", "Q2"),
                                     ("Q3_actionability_1to5", "Q3")):
                vals = pd.to_numeric(g[metric_col], errors="coerce")
                m_same = vals[same].mean()
                m_diff = vals[~same].mean()
                split[name] = {
                    "when_expert_agrees_with_system_label": float(m_same) if not pd.isna(m_same) else None,
                    "when_expert_disagrees": float(m_diff) if not pd.isna(m_diff) else None,
                }
            tr = g["Q4_trust_YES_or_NO"].astype(str).str.upper().eq("YES")
            split["Q4_trust_rate_agree"] = float(tr[same].mean()) if same.any() else None
            split["Q4_trust_rate_disagree"] = float(tr[~same].mean()) if (~same).any() else None
            out.setdefault("quality_split_by_expert_agreement", {})[cond] = split

    with open(OUT_DIR / "expert_evaluation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten: {OUT_DIR}/expert_evaluation_results.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=str, default=None,
                    help="path to a filled ratings CSV exported from the workbooks")
    args = ap.parse_args()
    if args.analyze:
        analyze(Path(args.analyze))
    else:
        main_generate()


if __name__ == "__main__":
    main()