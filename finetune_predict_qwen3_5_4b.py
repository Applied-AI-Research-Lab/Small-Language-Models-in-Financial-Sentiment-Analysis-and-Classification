#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
import unsloth
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


ALLOWED_LABELS = {"positive", "negative", "neutral"}


@dataclass
class QwenRunConfig:
    model_name: str = "unsloth/Qwen3.5-4B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 120
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    logging_steps: int = 1

    finetuned_max_new_tokens: int = 96
    finetuned_temperature: float = 0.3
    finetuned_top_p: float = 0.9

    generation_retry_attempts: int = 4
    random_state: int = 42


class FormatError(RuntimeError):
    pass


def to_chat_content_blocks(messages: list[dict]) -> list[dict]:
    block_messages = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            block_messages.append(
                {
                    "role": message["role"],
                    "content": [{"type": "text", "text": content}],
                }
            )
        else:
            block_messages.append(message)
    return block_messages


def apply_chat_template_compat(tokenizer, messages: list[dict], **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "string indices must be integers" not in str(exc):
            raise
        return tokenizer.apply_chat_template(to_chat_content_blocks(messages), **kwargs)


def build_paths() -> dict[str, Path]:
    paths = {
        "train_csv": Path("Datasets/financial_phrasebank/train.csv"),
        "val_csv": Path("Datasets/financial_phrasebank/validation.csv"),
        "test_csv": Path("Datasets/financial_phrasebank/test.csv"),
        "model_out": Path("Models/qwen3_5_4b_financial_lora"),
        "result_root": Path("Results/Qwen3_5_4b_financial"),
    }
    paths["model_out"].mkdir(parents=True, exist_ok=True)
    paths["result_root"].mkdir(parents=True, exist_ok=True)
    return paths


def normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    if text in ALLOWED_LABELS:
        return text
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return "neutral"


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


def to_training_text(tokenizer, sentence: str, label: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You classify financial sentiment into positive, negative, neutral.",
        },
        {
            "role": "user",
            "content": prediction_prompt(sentence),
        },
        {
            "role": "assistant",
            "content": f"Label: {label}",
        },
    ]
    return apply_chat_template_compat(tokenizer, messages, tokenize=False, add_generation_prompt=False)


def prepare_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "text": to_training_text(
                    tokenizer=tokenizer,
                    sentence=str(row["sentence"]),
                    label=normalize_label(row["label"]),
                )
            }
        )
    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def load_base_model(config: QwenRunConfig):
    requested_model = os.getenv("QWEN_MODEL_NAME", "").strip() or config.model_name
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=requested_model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
        )
        return model, tokenizer
    except RuntimeError as exc:
        if "No config file found" in str(exc) and requested_model != "unsloth/Qwen3.5-4B":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Qwen3.5-4B",
                max_seq_length=config.max_seq_length,
                load_in_4bit=config.load_in_4bit,
            )
            return model, tokenizer
        raise RuntimeError(
            "Unable to load Qwen model. Set QWEN_MODEL_NAME to a valid local path or Hugging Face model id.\n"
            f"Attempted model: {requested_model}\n"
            f"Original error: {exc}"
        ) from exc


def attach_lora(model, config: QwenRunConfig):
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.random_state,
        use_rslora=False,
    )


def build_trainer(model, tokenizer, train_ds: Dataset, val_ds: Dataset, config: QwenRunConfig, output_dir: Path):
    sft_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=config.random_state,
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=20,
        report_to="none",
    )

    kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "args": sft_args,
    }

    try:
        return SFTTrainer(tokenizer=tokenizer, **kwargs)
    except TypeError as exc:
        if "tokenizer" not in str(exc):
            raise
        return SFTTrainer(processing_class=tokenizer, **kwargs)


def train_model(config: QwenRunConfig, paths: dict[str, Path]) -> None:
    train_df = pd.read_csv(paths["train_csv"])
    val_df = pd.read_csv(paths["val_csv"])

    for col in ["sentence", "label"]:
        if col not in train_df.columns or col not in val_df.columns:
            raise ValueError("Train/validation CSV must include sentence,label columns.")

    model, tokenizer = load_base_model(config)
    model = attach_lora(model, config)

    train_ds = prepare_dataset(train_df, tokenizer)
    val_ds = prepare_dataset(val_df, tokenizer)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        config=config,
        output_dir=paths["result_root"] / "trainer_outputs",
    )

    start = time.perf_counter()
    out = trainer.train()
    train_elapsed = time.perf_counter() - start

    model.save_pretrained(str(paths["model_out"]))
    tokenizer.save_pretrained(str(paths["model_out"]))

    summary = {
        "model": config.model_name,
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "train_runtime_sec_wallclock": train_elapsed,
        "train_runtime_sec_reported": out.metrics.get("train_runtime"),
        "train_loss": out.metrics.get("train_loss"),
        "config": asdict(config),
    }
    (paths["result_root"] / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def load_finetuned_model(config: QwenRunConfig, adapter_path: Path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def infer_one(
    model,
    tokenizer,
    sentence: str,
    config: QwenRunConfig,
) -> tuple[str, str, float, str]:
    messages = [
        {"role": "system", "content": "Return only Label and Explanation fields exactly."},
        {"role": "user", "content": prediction_prompt(sentence)},
    ]

    inputs = apply_chat_template_compat(
        tokenizer,
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    total = 0.0
    last_raw = ""

    for _ in range(max(1, config.generation_retry_attempts)):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=config.finetuned_max_new_tokens,
                do_sample=True,
                temperature=config.finetuned_temperature,
                top_p=config.finetuned_top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()

        total += time.perf_counter() - start
        generated = output[0][inputs["input_ids"].shape[1] :]
        last_raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

        try:
            label, explanation = parse_prediction(last_raw)
            return label, explanation, total, last_raw
        except FormatError:
            continue

    raise FormatError(last_raw)


def run_predictions(config: QwenRunConfig, paths: dict[str, Path]) -> None:
    test_df = pd.read_csv(paths["test_csv"])
    if "sentence" not in test_df.columns or "label" not in test_df.columns:
        raise ValueError("Test CSV must include sentence,label columns.")

    model, tokenizer = load_finetuned_model(config, paths["model_out"])

    labels = []
    explanations = []
    times = []

    for idx, sentence in enumerate(test_df["sentence"].astype(str).tolist()):
        try:
            label, explanation, elapsed, _ = infer_one(model, tokenizer, sentence, config)
            labels.append(label)
            explanations.append(explanation)
            times.append(elapsed)
        except FormatError as exc:
            print("Prediction failed strict format validation.")
            print("Prompt:")
            print(prediction_prompt(sentence))
            print("Raw response:")
            print(str(exc))
            raise RuntimeError(f"Stopped at test row {idx} due to invalid model output.") from exc

    test_df["qwen_ft_label"] = labels
    test_df["qwen_ft_explanation"] = explanations
    test_df["qwen_ft_time_sec"] = times
    test_df.to_csv(paths["test_csv"], index=False)


def main() -> None:
    config = QwenRunConfig()
    paths = build_paths()

    missing = [p for p in [paths["train_csv"], paths["val_csv"], paths["test_csv"]] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing split files. Run split_financial_phrasebank.py first. Missing: "
            + ", ".join(str(m) for m in missing)
        )

    train_model(config, paths)
    run_predictions(config, paths)

    print("Qwen financial run completed.")
    print(f"Updated test file: {paths['test_csv']}")
    print(f"Adapter directory: {paths['model_out']}")
    print(f"Result directory: {paths['result_root']}")


if __name__ == "__main__":
    main()
