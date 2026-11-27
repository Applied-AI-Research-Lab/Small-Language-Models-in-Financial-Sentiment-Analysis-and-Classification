# D1 analysis plan (fixed before any rating is returned)

Frozen: 2026-11-18T19:06:04
Workbook SHA-256: `bab93b7d20fdf7e35763b6e2a0137bf9e22259531d84f95634a361c337b5bda3`
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
  (10000 resamples).

**Secondary endpoints:** Q1 factual accuracy and Q2 faithfulness.
- Per item: framework minus sentence-only, averaged over the two raters.
- Test: two-sided paired Wilcoxon signed-rank (zero differences handled by the Pratt
  method). Holm adjustment across the two secondary tests.
- Means are reported pooled and reweighted.

**Label-correctness control:** the same analyses restricted to the 50 items on which
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
