from __future__ import annotations

import glob
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA_DIR = PROJECT / "Datasets" / "financial_phrasebank"
TEST_CSV = DATA_DIR / "test.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
OOF_CSV = PROJECT / "Classic" / "results" / "ft_panel_aug" / "oof_train_signals.csv"
BERT_CKPT = PROJECT / "Classic" / "results" / "bert_ft_checkpoint"
BERT_PER_ROW = PROJECT / "Classic" / "results" / "bert_per_row.csv"

RESULTS = ROOT / "results"
INPUTS = RESULTS / "inputs"
C1_DIR = RESULTS / "c1"
SMOKE_DIR = RESULTS / "smoke"
STATE_DIR = RESULTS / "_state"
LOGS = ROOT / "logs"

SEED = 42
MAX_SEQ_LEN = 2048
RETRIES = 4
SAVE_EVERY = 10
KNN_K = 4
KNN_DUP_COS = 0.90
SUPERICL_K = 16
COT_EXTRA_TOKENS = {"qwen": 1024, "gemma": 768}

METHODS = ("cot", "knn", "supericl_bert", "supericl_svm")
SLMS = ("qwen", "gemma")

MODELS = {
    "qwen": dict(repo="unsloth/Qwen3.5-4B", commit="3764fa3",
                 max_new_tokens=512, temperature=0.3, top_p=0.9, top_k=None),
    "gemma": dict(repo="unsloth/gemma-3-4b-it-unsloth-bnb-4bit", commit="316726c",
                  max_new_tokens=256, temperature=0.7, top_p=0.95, top_k=64),
}
QWEN_SYSTEM = "You are a financial sentiment analyst. Follow the output format exactly."
ALLOWED = ("negative", "neutral", "positive")

FRAMEWORK_COL = {"qwen": "qwen_zs_label", "gemma": "gemma_zs_label"}
SENTONLY_COL = {"qwen": "qwen_sentonly_zs_label", "gemma": "gemma_sentonly_zs_label"}


def norm_text(s: str) -> str:
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def hf_hub_dirs() -> list[str]:
    cands = []
    if os.environ.get("HF_HOME"):
        cands.append(os.path.join(os.environ["HF_HOME"], "hub"))
    cands.append(str(PROJECT / ".cache" / "huggingface" / "hub"))
    cands.append(os.path.expanduser("~/.cache/huggingface/hub"))
    seen, out = set(), []
    for c in cands:
        if c not in seen and os.path.isdir(c):
            seen.add(c)
            out.append(c)
    return out


def resolve_snapshot(slm: str) -> str:
    repo, commit = MODELS[slm]["repo"], MODELS[slm]["commit"]
    tried = []
    for hub in hf_hub_dirs():
        pattern = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", commit + "*")
        tried.append(pattern)
        for snap in sorted(glob.glob(pattern)):
            if os.path.isfile(os.path.join(snap, "config.json")):
                return snap
    raise FileNotFoundError(
        f"Pinned snapshot {repo}@{commit} not found in the local HF cache.\n"
        f"Searched: {tried}\n"
        "Source ../activate_project.sh first (it sets HF_HOME), or check `ls $HF_HOME/hub`.")


def load_slm(slm: str):
    path = resolve_snapshot(slm)
    print(f"[load] {slm}: {path}", flush=True)
    if slm == "qwen":
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=path, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
    else:
        from unsloth import FastModel
        model, tok = FastModel.from_pretrained(
            model_name=path, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
            load_in_8bit=False, full_finetuning=False)
        FastModel.for_inference(model)
    return model, tok


def _qwen_template(tok, messages, **kw):
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError as exc:
        if "enable_thinking" in str(exc):
            return tok.apply_chat_template(messages, **kw)
        if "string indices must be integers" not in str(exc):
            raise
        block = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                 for m in messages]
        return tok.apply_chat_template(block, enable_thinking=False, **kw)


def encode(tok, slm: str, prompt: str):
    kw = dict(add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    if slm == "qwen":
        msgs = [{"role": "system", "content": QWEN_SYSTEM}, {"role": "user", "content": prompt}]
        return _qwen_template(tok, msgs, **kw)
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    return tok.apply_chat_template(msgs, **kw)


class FormatError(RuntimeError):
    pass


def parse_json(raw: str, need_reasoning: bool = False) -> dict:
    text = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise FormatError(f"no JSON object in: {text[:200]!r}")
    data = None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError as exc:
                raise FormatError(f"JSON decode error ({exc}) in: {text[:200]!r}")
    if not isinstance(data, dict):
        raise FormatError(f"JSON is not an object: {text[:200]!r}")
    label = str(data.get("label", "")).strip().lower()
    expl = str(data.get("explanation", "")).strip()
    rec = str(data.get("recommendation", "")).strip()
    reasoning = str(data.get("reasoning", "")).strip()
    if label not in ALLOWED:
        raise FormatError(f"invalid label {label!r}")
    if not expl or not rec:
        raise FormatError("empty explanation or recommendation")
    if need_reasoning and not reasoning:
        raise FormatError("empty reasoning")
    return {"label": label, "explanation": expl, "recommendation": rec, "reasoning": reasoning}


def generate(model, tok, slm: str, prompt: str, max_new_tokens: int,
             need_reasoning: bool = False) -> dict:
    import torch
    cfg = MODELS[slm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in encode(tok, slm, prompt).items()}
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=True,
                  temperature=cfg["temperature"], top_p=cfg["top_p"],
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    if cfg["top_k"] is not None:
        gen_kw["top_k"] = cfg["top_k"]
    total, last_raw = 0.0, ""
    for attempt in range(1, RETRIES + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        if device == "cuda":
            torch.cuda.synchronize()
        total += time.perf_counter() - t0
        last_raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            res = parse_json(last_raw, need_reasoning)
            res.update(time_sec=total, attempts=attempt, parse_error=0, raw="")
            return res
        except FormatError as exc:
            print(f"    attempt {attempt}/{RETRIES} parse error: {exc}", flush=True)
    return {"label": None, "explanation": "", "recommendation": "", "reasoning": "",
            "time_sec": total, "attempts": RETRIES, "parse_error": 1, "raw": last_raw[:2000]}


def set_seed(seed: int = SEED):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
