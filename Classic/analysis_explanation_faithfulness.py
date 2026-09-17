
import csv
import json
import re
from pathlib import Path

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
OUT_DIR = Path("Classic/results")
TOL_PP = 0.5

SLMS = {
    "Qwen3.5-4B (ZS)": "qwen_zs_explanation",
    "Gemma3-4B (ZS)": "gemma_zs_explanation",
}

QUOTED = re.compile(r"(?<![A-Za-z])['‘]([^'’]{1,40})['’]")
PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")

CLASS_LABELS = {"positive", "negative", "neutral"}

ABSENCE_MARKERS = re.compile(
    r"(absen[ct]e?|lack(?:ing| of)?|without|no strong|no explicit|not .*?(?:present|"
    r"disclose|appear|found)|rather than|instead of|despite the (?:absence|lack)|"
    r"neither|nor\b)", re.IGNORECASE)

THRESHOLD_CTX = re.compile(r"(exceed(?:ing|s)?|above|over|at least|more than|"
                          r"up to|below|under|less than|ranging|hovering|"
                          r"consistently above|approximately|around|about)\s*$",
                          re.IGNORECASE)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def quoted_terms(expl: str) -> list[str]:
    out = []
    for t in QUOTED.findall(expl):
        t = t.strip().strip(",.;:").strip()
        if re.search(r"[A-Za-z]", t) and len(t.split()) <= 5:
            out.append(t)
    return out


def percentages_with_context(expl: str):
    out = []
    for m in PCT.finditer(expl):
        ctx = expl[max(0, m.start() - 45):m.start()]
        out.append((float(m.group(1)), ctx))
    return out


def sentence_pcts(sentence: str) -> list[float]:
    return [float(x) for x in PCT.findall(sentence)]


def feature_terms(feat_str: str) -> list[str]:
    terms = []
    for part in feat_str.split("|"):
        if ":" in part:
            terms.append(part.split(":")[0].strip().lower())
    return terms


def main():
    rows = list(csv.DictReader(open(TEST_CSV)))
    n = len(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xgb_lists = {r["xgb_top_features_global"] for r in rows}
    assert len(xgb_lists) == 1, "XGBoost list is not global/constant!"

    per_model = {}
    per_row_records = []

    for name, col in SLMS.items():
        stats = {
            "n": n,
            "rows_with_quoted_terms": 0,
            "rows_with_pct": 0,
            "total_quoted_terms": 0,
            "terms_in_panel_features": 0,
            "terms_in_linear_features": 0,
            "terms_in_xgb_global_list": 0,
            "terms_only_in_sentence": 0,
            "terms_class_labels": 0,
            "terms_cited_as_absent": 0,
            "terms_ungrounded": 0,
            "total_pct_citations": 0,
            "pct_match_panel": 0,
            "pct_match_sentence_only": 0,
            "pct_threshold_claims": 0,
            "pct_unmatched": 0,
        }
        for r in rows:
            expl = r[col]
            sent_n = norm(r["sentence"])
            lr_terms = feature_terms(r["logreg_top_features"])
            svm_terms = feature_terms(r["svm_top_features"])
            xgb_terms = feature_terms(r["xgb_top_features_global"])
            lin_terms = lr_terms + svm_terms
            all_feat_terms = lin_terms + xgb_terms

            terms = quoted_terms(expl)
            pct_ctxs = percentages_with_context(expl)
            pcts = [v for v, _ in pct_ctxs]
            if terms:
                stats["rows_with_quoted_terms"] += 1
            if pcts:
                stats["rows_with_pct"] += 1

            for t in terms:
                tn = norm(t)
                stats["total_quoted_terms"] += 1
                in_lin = any(tn == f or (len(tn) > 3 and tn in f) or (len(f) > 3 and f in tn)
                             for f in lin_terms)
                in_xgb = any(tn == f or (len(tn) > 3 and tn in f) or (len(f) > 3 and f in tn)
                             for f in xgb_terms)
                in_sent = tn in sent_n
                is_label = tn in CLASS_LABELS
                pos = expl.find(t)
                ctx = expl[max(0, pos - 80):pos + len(t) + 5]
                cited_absent = bool(ABSENCE_MARKERS.search(ctx))
                if in_lin:
                    stats["terms_in_linear_features"] += 1
                if in_xgb:
                    stats["terms_in_xgb_global_list"] += 1
                if in_lin or in_xgb:
                    stats["terms_in_panel_features"] += 1
                    continue
                if in_sent:
                    stats["terms_only_in_sentence"] += 1
                elif is_label:
                    stats["terms_class_labels"] += 1
                elif cited_absent and not in_sent:
                    stats["terms_cited_as_absent"] += 1
                else:
                    stats["terms_ungrounded"] += 1

            panel_pcts = []
            for m in ("logreg", "svm", "xgb"):
                for cls in ("negative", "neutral", "positive"):
                    panel_pcts.append(float(r[f"{m}_prob_{cls}"]) * 100)
                panel_pcts.append(float(r[f"{m}_confidence"]) * 100)
            sent_p = sentence_pcts(r["sentence"])
            for p, ctx in pct_ctxs:
                stats["total_pct_citations"] += 1
                if any(abs(p - q) <= TOL_PP for q in panel_pcts):
                    stats["pct_match_panel"] += 1
                elif any(abs(p - q) <= TOL_PP for q in sent_p):
                    stats["pct_match_sentence_only"] += 1
                else:
                    thr = THRESHOLD_CTX.search(ctx)
                    if thr:
                        key = thr.group(1).lower()
                        confs = [float(r[f"{m}_confidence"]) * 100
                                 for m in ("logreg", "svm", "xgb")]
                        ok = False
                        if key.startswith(("exceed", "above", "over", "at least",
                                           "more than", "consistently above")):
                            ok = any(c > p for c in confs)
                        elif key.startswith(("up to", "below", "under", "less than")):
                            ok = any(c < p for c in confs)
                        elif key.startswith(("ranging", "hovering", "approximately",
                                             "around", "about")):
                            ok = min(confs) - 1.5 <= p <= max(confs) + 1.5
                        if ok:
                            stats["pct_threshold_claims"] += 1
                            continue
                    stats["pct_unmatched"] += 1

            per_row_records.append({
                "model": name,
                "sentence": r["sentence"][:60],
                "terms": " | ".join(terms),
                "n_pct": len(pcts),
            })

        tt = stats["total_quoted_terms"]
        tp = stats["total_pct_citations"]
        grounded_terms = (stats["terms_in_panel_features"]
                          + stats["terms_only_in_sentence"]
                          + stats["terms_class_labels"]
                          + stats["terms_cited_as_absent"])
        per_model[name] = {
            **stats,
            "quoted_term_coverage": stats["rows_with_quoted_terms"] / n,
            "term_validity_rate": stats["terms_in_panel_features"] / tt if tt else None,
            "term_grounded_rate": grounded_terms / tt if tt else None,
            "term_hallucination_rate": stats["terms_ungrounded"] / tt if tt else None,
            "pct_faithful_rate": ((stats["pct_match_panel"]
                                   + stats["pct_match_sentence_only"]) / tp) if tp else None,
            "pct_match_panel_rate": stats["pct_match_panel"] / tp if tp else None,
            "pct_unmatched_rate": stats["pct_unmatched"] / tp if tp else None,
        }

    with open(OUT_DIR / "explanation_faithfulness.json", "w") as f:
        json.dump(per_model, f, indent=2)
    with open(OUT_DIR / "explanation_faithfulness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", *SLMS.keys()])
        keys = [k for k in next(iter(per_model.values()))]
        for k in keys:
            w.writerow([k, *[per_model[m][k] for m in SLMS]])
    with open(OUT_DIR / "explanation_faithfulness_per_row.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "sentence", "terms", "n_pct"])
        w.writeheader()
        w.writerows(per_row_records)

    for name, m in per_model.items():
        print(f"== {name} ==")
        print(f"  rows w/ quoted terms: {m['rows_with_quoted_terms']}/{n} "
              f"({m['quoted_term_coverage']*100:.1f}%)  | total terms: {m['total_quoted_terms']}")
        print(f"  terms in panel features: {m['terms_in_panel_features']} "
              f"({m['term_validity_rate']*100:.1f}%)  "
              f"[linear/row-specific: {m['terms_in_linear_features']}, "
              f"xgb-global: {m['terms_in_xgb_global_list']}]")
        print(f"  terms only in sentence: {m['terms_only_in_sentence']}  | "
              f"class labels: {m['terms_class_labels']}  | "
              f"cited-as-absent: {m['terms_cited_as_absent']}  | "
              f"ungrounded: {m['terms_ungrounded']} ({m['term_hallucination_rate']*100:.1f}%)")
        print(f"  pct citations: {m['total_pct_citations']} -> panel-match "
              f"{m['pct_match_panel']} ({m['pct_match_panel_rate']*100:.1f}%), "
              f"sentence-match {m['pct_match_sentence_only']}, "
              f"verified-threshold {m['pct_threshold_claims']}, "
              f"unmatched {m['pct_unmatched']} ({m['pct_unmatched_rate']*100:.1f}%)\n")


if __name__ == "__main__":
    main()