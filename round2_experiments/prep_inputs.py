#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from common import (ALLOWED, BERT_CKPT, BERT_PER_ROW, COT_EXTRA_TOKENS, INPUTS, KNN_DUP_COS,
                    KNN_K, MAX_SEQ_LEN, MODELS, OOF_CSV, SEED, SUPERICL_K, TEST_CSV, TRAIN_CSV,
                    norm_text, resolve_snapshot)

JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object — no other text, no markdown fences:\n'
    '{"label": "<positive|negative|neutral>", '
    '"explanation": "<2-3 sentences>", '
    '"recommendation": "<1-2 sentences of actionable advice for a financial decision-maker>"}'
)
TASK_TAIL = ("Determine the correct sentiment (positive, negative, or neutral), explain your "
             "reasoning clearly by referencing the sentence content, and provide an actionable "
             "business recommendation for a financial decision-maker.\n\n")


def prompt_cot(sentence: str) -> str:
    return (
        "You are a senior financial sentiment analyst. Classify the sentiment of the "
        "following sentence from a financial news item.\n\n"
        f'SENTENCE: "{sentence.strip()}"\n\n'
        "Think step by step before answering: first reason about what the sentence reports, "
        "the direction of the reported change, and its likely implication for the company or "
        "its investors; then decide the sentiment (positive, negative, or neutral), explain "
        "your reasoning, and provide an actionable business recommendation for a financial "
        "decision-maker.\n\n"
        "Respond with ONLY a JSON object — no other text, no markdown fences — with the "
        "fields in this order:\n"
        '{"reasoning": "<your step-by-step reasoning>", '
        '"label": "<positive|negative|neutral>", '
        '"explanation": "<2-3 sentences>", '
        '"recommendation": "<1-2 sentences of actionable advice for a financial decision-maker>"}'
    )


def prompt_knn(sentence: str, demos: list[tuple[str, str]]) -> str:
    ex = "".join(f'Example {i}:\nSENTENCE: "{s.strip()}"\nSENTIMENT: {l}\n\n'
                 for i, (s, l) in enumerate(demos, 1))
    return (
        "You are a senior financial sentiment analyst. Classify the sentiment of a sentence "
        "from a financial news item. Labelled examples of similar sentences are given first.\n\n"
        f"{ex}"
        "Now classify the following sentence.\n\n"
        f'SENTENCE: "{sentence.strip()}"\n\n'
        + TASK_TAIL + JSON_INSTRUCTION
    )


def prompt_supericl(sentence: str, plugin_name: str, demos: list[dict], test_sig: dict) -> str:
    ex = "".join(
        f'Sentence: "{d["sentence"].strip()}"\n'
        f'{plugin_name} prediction: {d["pred"]} (Confidence: {d["conf"]:.2f})\n'
        f'Label: {d["label"]}\n\n' for d in demos)
    return (
        "You are a senior financial sentiment analyst. For each sentence from a financial news "
        f"item, a smaller classifier trained on this task ({plugin_name}) provides a predicted "
        "sentiment and its confidence, followed by the correct label. Use the examples to judge "
        "when the classifier can be trusted.\n\n"
        f"{ex}"
        "Now the sentence to classify:\n"
        f'Sentence: "{sentence.strip()}"\n'
        f'{plugin_name} prediction: {test_sig["pred"]} (Confidence: {test_sig["conf"]:.2f})\n\n'
        "Determine the correct sentiment (positive, negative, or neutral), explain your "
        "reasoning clearly by referencing the classifier's prediction and the sentence content, "
        "and provide an actionable business recommendation for a financial decision-maker.\n\n"
        + JSON_INSTRUCTION
    )


def bert_signals(texts: list[str]) -> list[dict]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not BERT_CKPT.exists():
        sys.exit(f"[prep] BERT checkpoint not found: {BERT_CKPT}")
    tok = AutoTokenizer.from_pretrained(str(BERT_CKPT))
    model = AutoModelForSequenceClassification.from_pretrained(str(BERT_CKPT))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, padding=True, max_length=256,
                      return_tensors="pt").to(dev)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().tolist()
            for p in probs:
                j = int(np.argmax(p))
                out.append({"pred": id2label[j], "conf": float(p[j])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-token-check", action="store_true",
                    help="skip the tokenizer length check (use only off-server)")
    ap.add_argument("--skip-bert", action="store_true",
                    help="build everything except the BERT arm (use only off-server)")
    args = ap.parse_args()

    INPUTS.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(TEST_CSV)[["sentence", "label", "svm_pred", "svm_confidence"]].copy()
    test.insert(0, "idx", range(len(test)))
    train = pd.read_csv(TRAIN_CSV)[["sentence", "label"]].copy()
    oof = pd.read_csv(OOF_CSV)
    oof = oof[oof["split"] == "train"]
    print(f"[prep] test={len(test)} train={len(train)} oof(train)={len(oof)}")

    test_norm = set(test["sentence"].map(norm_text))
    train["norm"] = train["sentence"].map(norm_text)
    dup_mask = train["norm"].isin(test_norm)
    print(f"[prep] training rows duplicating a test sentence (excluded): {int(dup_mask.sum())}")
    pool = train[~dup_mask].drop_duplicates("norm").reset_index(drop=True)

    cot = test[["idx", "sentence", "label"]].copy()
    cot["prompt"] = [prompt_cot(s) for s in cot["sentence"]]
    cot["fallback_label"] = "neutral"

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.98,
                          sublinear_tf=True)
    Xp = vec.fit_transform(pool["sentence"])
    Xt = vec.transform(test["sentence"])
    sims = (Xt @ Xp.T).toarray()
    knn_rows, knn_prompts = [], []
    for i in range(len(test)):
        order = np.argsort(-sims[i])
        chosen = [j for j in order if sims[i, j] < KNN_DUP_COS][:KNN_K]
        chosen = chosen[::-1]
        demos = [(pool.at[j, "sentence"], pool.at[j, "label"]) for j in chosen]
        knn_prompts.append(prompt_knn(test.at[i, "sentence"], demos))
        knn_rows.append({"idx": i, "neighbours": " || ".join(pool.at[j, "sentence"] for j in chosen),
                         "neighbour_labels": ",".join(pool.at[j, "label"] for j in chosen),
                         "cosines": ",".join(f"{sims[i, j]:.3f}" for j in chosen)})
    knn = test[["idx", "sentence", "label"]].copy()
    knn["prompt"] = knn_prompts
    knn["fallback_label"] = "neutral"
    pd.DataFrame(knn_rows).to_csv(INPUTS / "knn_neighbours.csv", index=False)

    oof["norm"] = oof["sentence"].map(norm_text)
    oof_map = oof.drop_duplicates("norm").set_index("norm")
    eligible = pool[pool["norm"].isin(oof_map.index)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(eligible), size=SUPERICL_K, replace=False).tolist())
    demo_df = eligible.loc[pick, ["sentence", "label", "norm"]].reset_index(drop=True)
    demo_df["svm_pred"] = [oof_map.at[n, "svm_pred"] for n in demo_df["norm"]]
    demo_df["svm_conf"] = [float(oof_map.at[n, "svm_confidence"]) for n in demo_df["norm"]]

    svm_demos = [{"sentence": r.sentence, "label": r.label, "pred": r.svm_pred, "conf": r.svm_conf}
                 for r in demo_df.itertuples()]
    sup_svm = test[["idx", "sentence", "label"]].copy()
    sup_svm["prompt"] = [prompt_supericl(r.sentence, "Linear SVM", svm_demos,
                                         {"pred": r.svm_pred, "conf": float(r.svm_confidence)})
                         for r in test.itertuples()]
    sup_svm["fallback_label"] = test["svm_pred"].values

    outputs = {"cot": cot, "knn": knn, "supericl_svm": sup_svm}

    if not args.skip_bert:
        demo_sig = bert_signals(demo_df["sentence"].tolist())
        test_sig = bert_signals(test["sentence"].tolist())
        demo_df["bert_pred"] = [d["pred"] for d in demo_sig]
        demo_df["bert_conf"] = [d["conf"] for d in demo_sig]
        bert_test = test[["idx", "sentence", "label"]].copy()
        bert_test["bert_pred"] = [d["pred"] for d in test_sig]
        bert_test["bert_conf"] = [d["conf"] for d in test_sig]
        bert_test.to_csv(INPUTS / "plugin_bert_test.csv", index=False)
        if BERT_PER_ROW.exists():
            ref = pd.read_csv(BERT_PER_ROW)["bert_ft_pred"].tolist()
            agree = float(np.mean([a == b for a, b in zip(ref, bert_test["bert_pred"])]))
            acc = float(np.mean(bert_test["bert_pred"] == bert_test["label"]))
            print(f"[prep] BERT plug-in: test acc={acc*100:.2f}% | agreement with stored "
                  f"bert_per_row.csv = {agree*100:.1f}%")
            if agree < 0.99:
                sys.exit("[prep] ABORT: BERT checkpoint does not reproduce the stored predictions.")
        bert_demos = [{"sentence": r.sentence, "label": r.label, "pred": r.bert_pred,
                       "conf": r.bert_conf} for r in demo_df.itertuples()]
        sup_bert = test[["idx", "sentence", "label"]].copy()
        sup_bert["prompt"] = [prompt_supericl(r.sentence, "BERT", bert_demos,
                                              {"pred": bp, "conf": bc})
                              for r, bp, bc in zip(test.itertuples(), bert_test["bert_pred"],
                                                   bert_test["bert_conf"])]
        sup_bert["fallback_label"] = bert_test["bert_pred"].values
        outputs["supericl_bert"] = sup_bert

    demo_df.drop(columns=["norm"]).to_csv(INPUTS / "supericl_demos.csv", index=False)

    manifest = {"seed": SEED, "knn_k": KNN_K, "knn_dup_cos": KNN_DUP_COS,
                "supericl_k": SUPERICL_K, "max_seq_len": MAX_SEQ_LEN,
                "n_train_dups_excluded": int(dup_mask.sum()), "files": {}, "max_tokens": {}}
    for name, df in outputs.items():
        assert set(df["fallback_label"]).issubset(ALLOWED), name
        path = INPUTS / f"prompts_{name}.csv"
        df.to_csv(path, index=False)
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"[prep] wrote {path.name} ({len(df)} prompts)")

    if not args.skip_token_check:
        from transformers import AutoTokenizer
        for slm in MODELS:
            tok = AutoTokenizer.from_pretrained(resolve_snapshot(slm))
            for name, df in outputs.items():
                n_max = max(len(tok(p)["input_ids"]) for p in df["prompt"])
                gen = COT_EXTRA_TOKENS[slm] if name == "cot" else MODELS[slm]["max_new_tokens"]
                manifest["max_tokens"][f"{slm}/{name}"] = {"prompt": n_max, "generation": gen}
                status = "OK" if n_max + gen + 64 <= MAX_SEQ_LEN else "TOO LONG"
                print(f"[prep] {slm:5s} {name:14s} max prompt tokens={n_max:5d} + gen {gen} -> {status}")
                if status != "OK":
                    sys.exit(f"[prep] ABORT: {slm}/{name} exceeds the {MAX_SEQ_LEN}-token window.")

    (INPUTS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[prep] done — prompts are frozen.")


if __name__ == "__main__":
    main()
