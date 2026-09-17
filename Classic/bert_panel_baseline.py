#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

ALLOWED_LABELS = {"positive", "negative", "neutral"}
OUR_LABELS = ("negative", "neutral", "positive")
SAVE_EVERY = 10

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
TRAIN_CSV = Path("Datasets/financial_phrasebank/train.csv")
VAL_CSV = Path("Datasets/financial_phrasebank/validation.csv")
RESULTS_DIR = Path("Classic/results")
BERT_NAME = "google-bert/bert-base-uncased"
BERT_CKPT = RESULTS_DIR / "bert_ft_checkpoint"
MAX_LEN = 256
EPOCHS = 3
FT_BATCH = 16
FT_LR = 2e-5
EVAL_BATCH = 64
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def macro_metrics(y_true: list[str], y_pred: list[str]):
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    f1s, sups = [], []
    for cls in OUR_LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t != cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != cls and t == cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        sups.append(sum(1 for t in y_true if t == cls))
    macro_f1 = sum(f1s) / len(f1s)
    weighted_f1 = sum(f * s for f, s in zip(f1s, sups)) / n
    return acc, macro_f1, weighted_f1


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


class FormatError(RuntimeError):
    pass


def parse_response(raw: str) -> tuple[str, str, str]:
    text = str(raw).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    json_m = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_m:
        raise FormatError(f"No JSON object found in: {text!r}")
    try:
        data = json.loads(json_m.group())
    except json.JSONDecodeError as exc:
        raise FormatError(f"JSON decode error ({exc}) in: {json_m.group()!r}")
    label = str(data.get("label", "")).strip().lower()
    explanation = str(data.get("explanation", "")).strip()
    recommendation = str(data.get("recommendation", "")).strip()
    if label not in ALLOWED_LABELS:
        raise FormatError(f"Invalid label '{label}' in: {data!r}")
    if not explanation or not recommendation:
        raise FormatError(f"Empty explanation or recommendation in: {data!r}")
    return label, explanation, recommendation


def run_baseline(args):
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {BERT_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_NAME, num_labels=3,
        id2label={0: OUR_LABELS[0], 1: OUR_LABELS[1], 2: OUR_LABELS[2]},
        label2id={OUR_LABELS[0]: 0, OUR_LABELS[1]: 1, OUR_LABELS[2]: 2},
    )
    model.to(device)

    def encode(texts: list[str]):
        enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN,
                        return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    train_rows = load_csv(TRAIN_CSV)
    val_rows = load_csv(VAL_CSV)
    test_rows = load_csv(TEST_CSV)

    def prep(rows):
        ids, mask = encode([r["sentence"] for r in rows])
        ys = torch.tensor([OUR_LABELS.index(r["label"]) for r in rows])
        return TensorDataset(ids, mask, ys)

    train_dl = DataLoader(prep(train_rows), batch_size=FT_BATCH, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
    val_dl = DataLoader(prep(val_rows), batch_size=EVAL_BATCH)

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    try:
        sched = get_linear_schedule_with_warmup(optim, int(0.06 * total_steps),
                                                total_steps)
    except ImportError:
        try:
            from transformers.optimization import get_linear_schedule_with_warmup
            sched = get_linear_schedule_with_warmup(optim, int(0.06 * total_steps),
                                                    total_steps)
        except ImportError:
            sched = None
    for ep in range(EPOCHS):
        model.train()
        running, seen = 0.0, 0
        for ids, mask, ys in train_dl:
            ids, mask, ys = ids.to(device), mask.to(device), ys.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            loss = torch.nn.functional.cross_entropy(out.logits, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if sched is not None:
                sched.step()
            optim.zero_grad()
            running += loss.item() * len(ys); seen += len(ys)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for ids, mask, ys in val_dl:
                logits = model(input_ids=ids.to(device),
                               attention_mask=mask.to(device)).logits
                correct += (logits.argmax(-1).cpu() == ys).sum().item()
                total += len(ys)
        print(f"[bert-ft] epoch {ep+1}/{EPOCHS}  train loss {running/seen:.4f}  "
              f"val acc {correct/total*100:.2f}%")

    model.eval()
    preds = []
    @torch.no_grad()
    def predict(texts):
        out = []
        for i in range(0, len(texts), EVAL_BATCH):
            ids, mask = encode(texts[i:i + EVAL_BATCH])
            logits = model(input_ids=ids.to(device),
                           attention_mask=mask.to(device)).logits
            out += [OUR_LABELS[j] for j in logits.argmax(-1).tolist()]
        return out
    preds = predict([r["sentence"] for r in test_rows])
    acc, mf1, wf1 = macro_metrics([r["label"] for r in test_rows], preds)
    print(f"\n[bert-ft] TEST: acc={acc*100:.2f}%  macroF1={mf1:.4f}  wF1={wf1:.4f}")

    BERT_CKPT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(BERT_CKPT)
    tokenizer.save_pretrained(BERT_CKPT)
    print(f"Checkpoint saved to: {BERT_CKPT}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bert_per_row.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "label", "bert_ft_pred"])
        for r, p in zip(test_rows, preds):
            w.writerow([r["sentence"], r["label"], p])

    mcn = {}
    if "qwen_zs_label" in test_rows[0]:
        qs = [r["qwen_zs_label"] for r in test_rows]
        b, c, chi2, p = mcnemar_exact(preds, qs, [r["label"] for r in test_rows])
        mcn["bert_ft_vs_qwen_zs"] = {"b": b, "c": c, "chi2_cc": chi2,
                                     "p_exact": p}
        print(f"[McNemar] bert-ft vs Qwen ZS: b={b} c={c} chi2={chi2:.2f} p={p:.4f}")

    results = {"bert_fine_tuned": {"accuracy": acc, "macro_f1": mf1,
                                   "weighted_f1": wf1, "n": len(test_rows),
                                   "epochs": EPOCHS, "lr": FT_LR,
                                   "batch": FT_BATCH, "seed": SEED,
                                   "base_model": BERT_NAME}}
    if mcn:
        results["mcnemar"] = mcn
    with open(RESULTS_DIR / "bert_baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Written: {RESULTS_DIR}/bert_per_row.csv, bert_baseline_metrics.json")


def build_bert_prompt(sentence: str, sig: dict) -> str:
    p = sig["probs"]
    return (
        "You are a senior financial sentiment analyst reviewing the output of a "
        "transformer advisory model that has already analyzed the sentence below.\n\n"
        f'SENTENCE: "{sentence.strip()}"\n\n'
        "━━━ ADVISORY MODEL SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "[1] BERT  (transformer fine-tuned on this task's training data)\n"
        f"    Prediction   : {sig['pred']}\n"
        f"    Confidence   : {sig['confidence']*100:.1f}%\n"
        f"    Probabilities: negative {p['negative']*100:.1f}% | "
        f"neutral {p['neutral']*100:.1f}% | positive {p['positive']*100:.1f}%\n"
        f"    Uncertainty  : {sig['entropy']:.4f}  (entropy — lower = more certain)\n"
        "    Driving words: not available (opaque transformer; no lexical "
        "feature attribution provided)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Based on the advisory model signals and the sentence itself, determine "
        "the correct final sentiment, explain your reasoning clearly by "
        "referencing the model signals and the sentence content, and provide an "
        "actionable business recommendation for a financial decision-maker.\n\n"
        'Respond with ONLY a JSON object — no other text, no markdown fences:\n'
        '{"label": "<positive|negative|neutral>", '
        '"explanation": "<2-3 sentences referencing the model signals and the sentence>", '
        '"recommendation": "<1-2 sentences of actionable advice for a financial decision-maker>"}'
    )


from dataclasses import dataclass

@dataclass
class SLMConfig:
    qwen_model_name: str = "unsloth/Qwen3.5-4B"
    gemma_model_name: str = "unsloth/gemma-3-4b-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    qwen_max_new_tokens: int = 512
    qwen_temperature: float = 0.3
    qwen_top_p: float = 0.9
    gemma_max_new_tokens: int = 256
    gemma_temperature: float = 0.7
    gemma_top_p: float = 0.95
    gemma_top_k: int = 64
    generation_retry_attempts: int = 4


def _apply_chat_template_compat(tokenizer, messages, enable_thinking=False, **kwargs):
    call_kwargs = dict(enable_thinking=enable_thinking, **kwargs)
    try:
        return tokenizer.apply_chat_template(messages, **call_kwargs)
    except TypeError as exc:
        err = str(exc)
        if "string indices must be integers" not in err and "enable_thinking" not in err:
            raise
        if "enable_thinking" in err:
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError as exc2:
                if "string indices must be integers" not in str(exc2):
                    raise
        block_msgs = []
        for m in messages:
            content = m.get("content", "")
            block_msgs.append({
                "role": m["role"],
                "content": [{"type": "text", "text": content}]
                if isinstance(content, str) else content,
            })
        try:
            return tokenizer.apply_chat_template(block_msgs, **call_kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(block_msgs, **kwargs)


def bert_probs_for_all(model, tokenizer, device, texts: list[str]) -> list[dict]:
    if torch is None:
        raise RuntimeError("torch is required on the GPU server")
    out = []
    for i in range(0, len(texts), EVAL_BATCH):
        batch = texts[i:i + EVAL_BATCH]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=MAX_LEN, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
        probs = torch.softmax(logits, dim=-1)
        for row_probs in probs.tolist():
            p = dict(zip(OUR_LABELS, row_probs))
            pred = max(p, key=p.get)
            conf = p[pred]
            ent = -sum(q * math.log(q + 1e-12) for q in row_probs)
            out.append({"pred": pred, "probs": p, "confidence": conf,
                        "entropy": ent})
    return out


def _generate_and_parse(model, tokenizer, messages, gen_kwargs, cfg, tag):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if gen_kwargs.get("_gemma"):
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True)
    else:
        inputs = _apply_chat_template_compat(
            tokenizer, messages, enable_thinking=False,
            add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    total_time = 0.0
    max_attempts = max(1, cfg.generation_retry_attempts)
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=gen_kwargs["max_new_tokens"],
                do_sample=True,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs.get("top_p"),
                top_k=gen_kwargs.get("top_k"),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        generated = output[0][inputs["input_ids"].shape[1]:]
        last_raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            return *parse_response(last_raw), total_time
        except FormatError as parse_exc:
            print(f"  [{tag}] Attempt {attempt}/{max_attempts} failed — "
                  f"parse error: {parse_exc}")
            continue
    raise FormatError(last_raw)


def run_slm_over_bert(df: pd.DataFrame, prompts: list[str], cfg: SLMConfig,
                       slm: str, bert_preds: list[str], test_path: Path):
    if slm == "qwen":
        prefix, tag = "bert_qwen_zs", "Qwen"
        from unsloth import FastLanguageModel
        print("Loading Qwen3.5-4B ...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.qwen_model_name, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit)
        FastLanguageModel.for_inference(model)
        sys_msg = "You are a financial sentiment analyst. Follow the output format exactly."
        gen = {"max_new_tokens": cfg.qwen_max_new_tokens,
               "temperature": cfg.qwen_temperature, "top_p": cfg.qwen_top_p}
        mk = lambda p: [{"role": "system", "content": sys_msg},
                        {"role": "user", "content": p}]
    else:
        prefix, tag = "bert_gemma_zs", "Gemma"
        from unsloth import FastModel
        print("Loading Gemma3-4B ...")
        model, tokenizer = FastModel.from_pretrained(
            model_name=cfg.gemma_model_name, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit, load_in_8bit=False,
            full_finetuning=False)
        FastModel.for_inference(model)
        gen = {"max_new_tokens": cfg.gemma_max_new_tokens,
               "temperature": cfg.gemma_temperature, "top_p": cfg.gemma_top_p,
               "top_k": cfg.gemma_top_k, "_gemma": True}
        mk = lambda p: [{"role": "user",
                         "content": [{"type": "text", "text": p}]}]

    for col in (f"{prefix}_label", f"{prefix}_explanation",
                f"{prefix}_recommendation", f"{prefix}_time_sec"):
        if col not in df.columns:
            df[col] = None
    n = len(df)
    n_fb = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.notna(df.at[idx, f"{prefix}_label"]):
            continue
        prompt = prompts[i]
        try:
            label, expl, rec, elapsed = _generate_and_parse(
                model, tokenizer, mk(prompt), gen, cfg, tag)
        except FormatError as exc:
            raw = str(exc)
            print(f"\n[{tag}] ALL RETRIES FAILED — row {i} (idx {idx}). "
                  f"Falling back to BERT panel prediction.")
            label = bert_preds[i]
            expl = f"[parse_error] {raw[:300]}"
            rec = "Unable to generate recommendation due to format error."
            elapsed = 0.0
            n_fb += 1
        df.at[idx, f"{prefix}_label"] = label
        df.at[idx, f"{prefix}_explanation"] = expl
        df.at[idx, f"{prefix}_recommendation"] = rec
        df.at[idx, f"{prefix}_time_sec"] = elapsed
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == n:
            df.to_csv(test_path, index=False)
            print(f"  {tag} (BERT panel): {i+1}/{n} rows done (checkpoint)")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{tag} over BERT panel complete. Fallbacks: {n_fb}")
    return df


def run_panel(args):
    if not BERT_CKPT.exists():
        raise FileNotFoundError(
            f"Fine-tuned BERT checkpoint not found at {BERT_CKPT}.\n"
            "Run `python Classic/bert_panel_baseline.py --mode baseline` first.")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BERT panel checkpoint: {BERT_CKPT} ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_CKPT)
    model.to(device).eval()

    test_path = TEST_CSV
    df = pd.read_csv(test_path)
    n_rows = 20 if args.smoke else len(df)
    df_work = df.iloc[:n_rows].copy()
    texts = df_work["sentence"].astype(str).tolist()
    print(f"Extracting BERT panel signals for {len(texts)} rows ...")
    sigs = bert_probs_for_all(model, tokenizer, device, texts)
    bert_preds = [s["pred"] for s in sigs]
    acc = sum(1 for s, t in zip(bert_preds, df_work["label"]) if s == t) / len(df_work)
    print(f"BERT panel-only accuracy on these {len(df_work)} rows: {acc*100:.2f}%")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompts = [build_bert_prompt(s, sig)
               for s, sig in zip(df_work["sentence"].astype(str), sigs)]
    print(f"Built {len(prompts)} BERT-panel prompts.")

    which = ["qwen", "gemma"] if args.slm == "both" else [args.slm]
    if args.smoke:
        which = which[:1]
    for slm in which:
        df_work = run_slm_over_bert(df_work, prompts, SLMConfig(), slm,
                                    bert_preds, test_path)

    new_cols = [c for c in df_work.columns
                if c.startswith("bert_") and c.endswith(
                    ("_label", "_explanation", "_recommendation", "_time_sec"))]
    for col in new_cols:
        if col not in df.columns:
            df[col] = None
    mask = df.index.isin(df_work.index)
    for col in new_cols:
        df.loc[mask, col] = df_work[col]
    df.to_csv(test_path, index=False)
    print(f"\nUpdated test CSV with new columns: {test_path}")

    from sklearn.metrics import accuracy_score, f1_score
    rows = []
    name_map = {"qwen": ("bert_qwen_zs", "qwen_over_bert"),
                "gemma": ("bert_gemma_zs", "gemma_over_bert")}
    for slm in which:
        prefix, model_name = name_map[slm]
        col = f"{prefix}_label"
        if col not in df_work.columns or df_work[col].isna().all():
            continue
        m = df_work[col].notna() & ~df_work[col].astype(str).str.startswith(
            "[parse_error]", na=False)
        sub = df_work[m]
        if sub.empty:
            continue
        rows.append({
            "model": model_name, "n_evaluated": len(sub),
            "accuracy": accuracy_score(sub["label"].str.lower(), sub[col].str.lower()),
            "macro_f1": f1_score(sub["label"].str.lower(), sub[col].str.lower(),
                                 average="macro"),
            "weighted_f1": f1_score(sub["label"].str.lower(), sub[col].str.lower(),
                                    average="weighted"),
            "avg_time_sec": df_work[f"{prefix}_time_sec"].mean(),
        })
    if rows:
        md = pd.DataFrame(rows)
        path = RESULTS_DIR / "test_metrics_zs_bert_panel.csv"
        md.to_csv(path, index=False)
        print(f"Metrics saved to: {path}")
        print(md.to_string(index=False))
    if args.smoke:
        print("\nSMOKE TEST COMPLETE — BERT-panel → SLM chain OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "panel", "all"], default="all")
    ap.add_argument("--slm", choices=["qwen", "gemma", "both"], default="both")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.mode = "panel"
    if args.mode in ("baseline", "all"):
        run_baseline(args)
    if args.mode in ("panel", "all"):
        run_panel(args)


if __name__ == "__main__":
    main()