#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

TRAIN_CSV = Path("Datasets/financial_phrasebank/train.csv")
VAL_CSV = Path("Datasets/financial_phrasebank/validation.csv")
TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
OUT_DIR = Path("Classic/results/ft_tuning")
ADAPTER_ROOT = Path("Models/tuned")
ALLOWED_LABELS = {"positive", "negative", "neutral"}


@dataclass
class BaseCfg:
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    logging_steps: int = 1
    eval_steps: int = 20
    random_state: int = 42
    generation_retry_attempts: int = 4
    max_new_tokens: int = 96
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 64


QWEN_MODEL = "unsloth/Qwen3.5-4B"
GEMMA_MODEL = "unsloth/gemma-3-4b-it"

GRID = [(8, 120), (8, 480), (16, 120), (16, 480)]


def prediction_prompt(sentence: str) -> str:
    return (
        "Classify financial sentiment for the sentence below. "
        "Allowed labels: positive, negative, neutral.\n"
        "You must return EXACTLY this format:\n"
        "Label: <positive|negative|neutral>\n"
        "Explanation: <brief reason>\n\n"
        f"Sentence: {sentence}"
    )


def parse_prediction(raw: str) -> tuple[str, str]:
    text = str(raw).strip()
    label_match = re.search(r"label\s*:\s*(positive|negative|neutral)", text, flags=re.IGNORECASE)
    expl_match = re.search(r"explanation\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if not label_match or not expl_match:
        raise FormatError(f"Invalid response format: {text!r}")
    label = label_match.group(1).lower().strip()
    explanation = expl_match.group(1).strip()
    if label not in ALLOWED_LABELS or not explanation:
        raise FormatError(f"Invalid label or empty explanation: {text!r}")
    return label, explanation


class FormatError(RuntimeError):
    pass


def parse_prediction_label(raw: str) -> tuple[str, str]:
    text = str(raw).strip()
    label_match = re.search(
        r"label\s*:\s*(positive|negative|neutral)", text, flags=re.IGNORECASE)
    if not label_match:
        raise FormatError(f"Invalid response format: {text!r}")
    label = label_match.group(1).lower().strip()
    if label not in ALLOWED_LABELS:
        raise FormatError(f"Invalid label: {text!r}")
    expl_match = re.search(
        r"explanation\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    explanation = expl_match.group(1).strip() if expl_match else ""
    return label, explanation


def to_chat_content_blocks(messages):
    block_messages = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            block_messages.append(
                {"role": message["role"],
                 "content": [{"type": "text", "text": content}]}
            )
        else:
            block_messages.append(message)
    return block_messages


def apply_chat_template_compat(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "string indices must be integers" not in str(exc):
            raise
        return tokenizer.apply_chat_template(to_chat_content_blocks(messages), **kwargs)


def normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    if text in ALLOWED_LABELS:
        return text
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return "neutral"


def apply_chat_template_no_think(tokenizer, messages, **kwargs):
    call_kwargs = dict(enable_thinking=False, **kwargs)
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
            block_msgs.append(
                {"role": m["role"],
                 "content": [{"type": "text", "text": content}]
                 if isinstance(content, str) else content})
        try:
            return tokenizer.apply_chat_template(block_msgs, **call_kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(block_msgs, **kwargs)


def qwen_training_text(tokenizer, sentence: str, label: str) -> str:
    messages = [
        {"role": "system",
         "content": "You classify financial sentiment into positive, negative, neutral."},
        {"role": "user", "content": prediction_prompt(sentence)},
        {"role": "assistant", "content": f"Label: {label}"},
    ]
    return apply_chat_template_compat(
        tokenizer, messages, tokenize=False, add_generation_prompt=False)


def gemma_training_text(tokenizer, sentence: str, label: str) -> str:
    messages = [
        {"role": "user",
         "content": [{"type": "text", "text": prediction_prompt(sentence)}]},
        {"role": "assistant",
         "content": [{"type": "text", "text": f"Label: {label}"}]},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return formatted.removeprefix("<bos>")


def resolve_local_snapshot(repo_id: str) -> str:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser(
        os.path.join("~", ".cache", "huggingface"))
    model_dir = os.path.join(hf_home, "hub", "models--" + repo_id.replace("/", "--"))
    ref = os.path.join(model_dir, "refs", "main")
    if os.path.isfile(ref):
        try:
            commit = open(ref).read().strip()
        except OSError:
            commit = ""
        snap = os.path.join(model_dir, "snapshots", commit)
        if os.path.isfile(os.path.join(snap, "config.json")):
            return snap
    import glob
    hits = sorted(glob.glob(os.path.join(model_dir, "snapshots", "*", "config.json")))
    if hits:
        return os.path.dirname(hits[-1])
    return repo_id


def load_model(model_key: str, cfg: BaseCfg, adapter_path: str | None = None):
    import torch
    if model_key == "qwen":
        from unsloth import FastLanguageModel
        src = adapter_path or resolve_local_snapshot(QWEN_MODEL)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=src, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit)
        if adapter_path:
            FastLanguageModel.for_inference(model)
    else:
        from unsloth import FastModel
        src = adapter_path or resolve_local_snapshot(GEMMA_MODEL)
        model, tokenizer = FastModel.from_pretrained(
            model_name=src, max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit, load_in_8bit=False,
            full_finetuning=False)
        if adapter_path:
            FastModel.for_inference(model)
    return model, tokenizer


def attach_lora(model_key: str, model, lora_r: int, random_state: int):
    if model_key == "qwen":
        from unsloth import FastLanguageModel
        return FastLanguageModel.get_peft_model(
            model, r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_r,
            lora_dropout=0.0, bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=random_state, use_rslora=False)
    else:
        from unsloth import FastModel
        return FastModel.get_peft_model(
            model, r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_r,
            lora_dropout=0.0, bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=random_state, use_rslora=False)


def cfg_id(model_key: str, lora_r: int, max_steps: int, smoke: bool = False) -> str:
    return f"{model_key}_r{lora_r}_s{max_steps}{'_smoke' if smoke else ''}"


def prepare_train_dataset(model_key: str, tokenizer, df):
    from datasets import Dataset
    rows = []
    text_fn = qwen_training_text if model_key == "qwen" else gemma_training_text
    for _, row in df.iterrows():
        rows.append({"text": text_fn(tokenizer, str(row["sentence"]),
                                    normalize_label(row["label"]))})
    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def train_one(model_key: str, lora_r: int, max_steps: int, cfg: BaseCfg,
              smoke: bool = False) -> Path:
    cid = cfg_id(model_key, lora_r, max_steps, smoke=smoke)
    adapter_dir = ADAPTER_ROOT / cid
    marker = adapter_dir / "TRAIN_DONE"
    if marker.exists():
        print(f"[{cid}] already trained — skipping")
        return adapter_dir

    train_df = pd.read_csv(TRAIN_CSV)
    if smoke:
        max_steps = 10

    print(f"[{cid}] loading base model...")
    model, tokenizer = load_model(model_key, cfg)
    model = attach_lora(model_key, model, lora_r, cfg.random_state)

    train_ds = prepare_train_dataset(model_key, tokenizer, train_df)

    from trl import SFTConfig, SFTTrainer
    sft_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        max_steps=max_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=cfg.random_state,
        output_dir=str(OUT_DIR / "trainer_states" / cid),
        save_strategy="no",
        report_to="none",
    )
    kwargs = {"model": model, "train_dataset": train_ds, "args": sft_args}
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **kwargs)

    print(f"[{cid}] training {max_steps} steps (~{max_steps * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps} samples)...")
    t0 = time.perf_counter()
    out = trainer.train()
    elapsed = time.perf_counter() - t0
    print(f"[{cid}] train_loss={out.metrics.get('train_loss'):.4f} "
          f"elapsed={elapsed/60:.1f} min")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    summary = {
        "config_id": cid, "lora_r": lora_r, "max_steps": max_steps,
        "effective_epochs": round(max_steps * 2 * 4 / len(train_df), 3),
        "train_loss": out.metrics.get("train_loss"),
        "train_runtime_sec": elapsed,
        "base_config": asdict(cfg),
    }
    (adapter_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    del model, tokenizer, trainer
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    marker.write_text("done")
    return adapter_dir


def infer_rows(model_key: str, model, tokenizer, sentences: list[str], cfg: BaseCfg,
               progress_every: int = 5):
    import torch
    preds, expls, times = [], [], []
    use_top_k = (model_key == "gemma")
    print(f"  generating {len(sentences)} predictions "
          f"(first call includes kernel warm-up: a few quiet minutes is NORMAL)...")
    for row_i, sentence in enumerate(sentences):
        if model_key == "qwen":
            messages = [
                {"role": "system",
                 "content": "Return only Label and Explanation fields exactly."},
                {"role": "user", "content": prediction_prompt(sentence)},
            ]
            inputs = apply_chat_template_no_think(
                tokenizer, messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True)
        else:
            messages = [
                {"role": "user",
                 "content": [{"type": "text", "text": prediction_prompt(sentence)}]},
            ]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        label, explanation, elapsed = None, None, 0.0
        last_raw = ""
        attempts = 0
        for _ in range(max(1, cfg.generation_retry_attempts)):
            attempts += 1
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id)
                if use_top_k:
                    gen_kwargs["top_k"] = cfg.top_k
                output = model.generate(**inputs, **gen_kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            generated = output[0][inputs["input_ids"].shape[1]:]
            last_raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
            try:
                label, explanation = parse_prediction_label(last_raw)
                break
            except FormatError:
                print(f"    row {row_i}: attempt {attempts} parse failure, retrying...")
                continue
        if label is None:
            label, explanation = "[parse_error]", f"[parse_error] {last_raw[:200]}"
        preds.append(label)
        expls.append(explanation)
        times.append(elapsed)
        if (row_i + 1) % progress_every == 0 or (row_i + 1) == len(sentences):
            print(f"    {row_i + 1}/{len(sentences)} rows done "
                  f"(last row took {elapsed:.1f}s)")
    return preds, expls, times


def evaluate_on_validation(model_key: str, lora_r: int, max_steps: int,
                           cfg: BaseCfg, smoke: bool = False) -> float:
    cid = cfg_id(model_key, lora_r, max_steps, smoke=smoke)
    adapter_dir = ADAPTER_ROOT / cid
    cache = OUT_DIR / f"val_{cid}.csv"
    val_df = pd.read_csv(VAL_CSV)
    if smoke:
        val_df = val_df.head(20)

    if cache.exists():
        cached = pd.read_csv(cache)
        if len(cached) == len(val_df):
            preds = cached["pred"].astype(str).str.lower().tolist()
            truth = val_df["label"].astype(str).str.lower().tolist()
            acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
            print(f"[{cid}] cached validation accuracy: {acc*100:.2f}%")
            return acc

    print(f"[{cid}] evaluating on validation split ({len(val_df)} rows)...")
    model, tokenizer = load_model(model_key, cfg, adapter_path=str(adapter_dir))
    preds, expls, times = infer_rows(
        model_key, model, tokenizer, val_df["sentence"].astype(str).tolist(), cfg)
    truth = val_df["label"].astype(str).str.lower().tolist()
    acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
    n_err = sum(p.startswith("[parse_error]") for p in preds)
    print(f"[{cid}] validation accuracy: {acc*100:.2f}% (parse_errors={n_err})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sentence": val_df["sentence"], "label": val_df["label"],
                   "pred": preds, "explanation": expls, "time_sec": times}
                  ).to_csv(cache, index=False)

    del model, tokenizer
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc


def final_test_run(model_key: str, lora_r: int, max_steps: int,
                   cfg: BaseCfg, smoke: bool = False) -> None:
    cid = cfg_id(model_key, lora_r, max_steps, smoke=smoke)
    adapter_dir = ADAPTER_ROOT / cid
    prefix = f"{model_key}_ft_tuned{'_smoke' if smoke else ''}"
    test_df = pd.read_csv(TEST_CSV)
    n_rows = 20 if smoke else len(test_df)
    work = test_df.head(n_rows)

    done_col = f"{prefix}_label"
    if done_col in test_df.columns and test_df[done_col].notna().all():
        print(f"[{cid}] test predictions already complete — skipping")
        return

    print(f"[{cid}] FINAL TEST RUN: selected config {lora_r=} {max_steps=} "
          f"on {n_rows} rows...")
    model, tokenizer = load_model(model_key, cfg, adapter_path=str(adapter_dir))
    for col in ("label", "explanation", "time_sec"):
        if f"{prefix}_{col}" not in test_df.columns:
            test_df[f"{prefix}_{col}"] = None

    for i, (idx, row) in enumerate(work.iterrows()):
        if pd.notna(test_df.at[idx, f"{prefix}_label"]):
            continue
        pred, expl, t = infer_rows(model_key, model, tokenizer,
                                   [str(row["sentence"])], cfg)
        test_df.at[idx, f"{prefix}_label"] = pred[0]
        test_df.at[idx, f"{prefix}_explanation"] = expl[0]
        test_df.at[idx, f"{prefix}_time_sec"] = t[0]
        if (i + 1) % 10 == 0 or (i + 1) == n_rows:
            test_df.to_csv(TEST_CSV, index=False)
            print(f"  {i+1}/{n_rows} rows (checkpoint)")
    test_df.to_csv(TEST_CSV, index=False)

    del model, tokenizer
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{cid}] test predictions written: {prefix}_* columns")


def run_model_grid(model_key: str, cfg: BaseCfg, quick: bool, smoke: bool) -> None:
    grid = GRID
    if quick:
        orig_r = 16 if model_key == "qwen" else 8
        grid = [(orig_r, 120), (orig_r, 480)]
    if smoke:
        grid = [grid[0]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / f"tuning_{model_key}{'_smoke' if smoke else ''}.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for lora_r, max_steps in grid:
        cid = cfg_id(model_key, lora_r, max_steps)
        if cid in results and "val_accuracy" in results[cid]:
            print(f"[{cid}] already evaluated — skipping")
            continue
        train_one(model_key, lora_r, max_steps, cfg, smoke=smoke)
        acc = evaluate_on_validation(model_key, lora_r, max_steps, cfg, smoke=smoke)
        tsum_path = ADAPTER_ROOT / cid / "training_summary.json"
        tsum = json.loads(tsum_path.read_text()) if tsum_path.exists() else {}
        results[cid] = {
            "lora_r": lora_r, "max_steps": max_steps,
            "val_accuracy": acc,
            "train_loss": tsum.get("train_loss"),
            "effective_epochs": tsum.get("effective_epochs"),
        }
        results_path.write_text(json.dumps(results, indent=2))

    best_cid = max(results, key=lambda c: (results[c]["val_accuracy"],
                                           -results[c]["max_steps"]))
    best = results[best_cid]
    print(f"\n=== {model_key} tuning summary ===")
    for cid, r in sorted(results.items()):
        print(f"  {cid}: val_acc={r['val_accuracy']*100:.2f}%  "
              f"train_loss={r.get('train_loss')}")
    print(f"  SELECTED: {best_cid} (val_acc={best['val_accuracy']*100:.2f}%)")

    summary = {
        "model": model_key,
        "smoke": bool(smoke),
        "grid": {cid: r for cid, r in results.items()},
        "selected": best_cid,
        "selected_config": {"lora_r": best["lora_r"], "max_steps": best["max_steps"]},
        "original_config": {"lora_r": 16 if model_key == "qwen" else 8,
                            "max_steps": 120},
    }
    (OUT_DIR / f"tuning_{model_key}{'_smoke' if smoke else ''}_selected.json").write_text(
        json.dumps(summary, indent=2))
    print(f"written: {OUT_DIR}/tuning_{model_key}{'_smoke' if smoke else ''}_selected.json")

    if not smoke:
        final_test_run(model_key, best["lora_r"], best["max_steps"], cfg, smoke=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "gemma"), required=True)
    ap.add_argument("--quick", action="store_true",
                    help="reduced grid: original rank at 120 and 480 steps")
    ap.add_argument("--smoke", action="store_true",
                    help="validation: 1 config, 10 train steps, 20 val rows")
    args = ap.parse_args()

    for p in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not p.exists():
            raise FileNotFoundError(p)

    cfg = BaseCfg()
    if args.model == "gemma":
        cfg.temperature = 0.7
        cfg.top_p = 0.95
        cfg.top_k = 64
    run_model_grid(args.model, cfg, quick=args.quick, smoke=args.smoke)
    print("\nDONE.")


if __name__ == "__main__":
    main()