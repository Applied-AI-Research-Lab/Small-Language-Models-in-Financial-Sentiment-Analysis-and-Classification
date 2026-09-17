
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

TEST_CSV = Path("Datasets/financial_phrasebank/test.csv")
TRAIN_CSV = Path("Datasets/financial_phrasebank/train.csv")
VAL_CSV = Path("Datasets/financial_phrasebank/validation.csv")
OUT_DIR = Path("Classic/results")
MODEL_NAME = "ProsusAI/finbert"
SEED = 42
MAX_LEN = 256
EPOCHS = 3
FT_BATCH = 16
FT_LR = 2e-5
EVAL_BATCH = 64
OUR_LABELS = ("negative", "neutral", "positive")


def set_seed(seed: int):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def macro_metrics(y_true: list[str], y_pred: list[str]):
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    f1s, supports = [], []
    for cls in OUR_LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == cls and t != cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != cls and t == cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        supports.append(sum(1 for t in y_true if t == cls))
    macro_f1 = sum(f1s) / len(f1s)
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / n
    return acc, macro_f1, weighted_f1


def mcnemar_exact(y_a_correct, y_b_correct):
    b = sum(1 for a, w in zip(y_a_correct, y_b_correct) if a and not w)
    c = sum(1 for a, w in zip(y_a_correct, y_b_correct) if not a and w)
    nd = b + c
    if nd == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / nd
    k = min(b, c)
    tail = sum(math.comb(nd, i) for i in range(k + 1)) / 2 ** nd
    p_exact = min(1.0, 2.0 * tail)
    return b, c, chi2, p_exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "zero-shot", "ft"], default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="20-row zero-shot environment check")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-ft", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    try:
        from transformers import get_linear_schedule_with_warmup
    except ImportError:
        try:
            from transformers.optimization import get_linear_schedule_with_warmup
        except ImportError:
            get_linear_schedule_with_warmup = None

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected - fine-tuning will be very slow.")

    test_rows = load_csv(TEST_CSV)
    if args.smoke:
        test_rows = test_rows[:20]
        args.mode = "zero-shot"
        args.no_ft = True

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    raw_map = model.config.id2label
    id2label = {int(k): str(v).lower() for k, v in raw_map.items()}
    hf_labels = [id2label[i] for i in range(model.config.num_labels)]
    assert set(hf_labels) == set(OUR_LABELS), f"unexpected label set: {hf_labels}"
    hf2our = {i: OUR_LABELS.index(hf_labels[i]) for i in range(len(hf_labels))}
    model.to(device)

    def encode(texts: list[str]):
        enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN,
                        return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    @torch.no_grad()
    def predict(texts: list[str]) -> list[str]:
        model.eval()
        preds = []
        for i in range(0, len(texts), EVAL_BATCH):
            ids, mask = encode(texts[i:i + EVAL_BATCH])
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            probs = torch.softmax(logits, dim=-1)
            reord = torch.zeros_like(probs)
            for hf_i in range(len(hf_labels)):
                reord[:, hf2our[hf_i]] = probs[:, hf_i]
            preds += [OUR_LABELS[j] for j in reord.argmax(dim=-1).tolist()]
        return preds

    results = {}
    zs_preds, ft_preds = None, None

    if args.mode in ("all", "zero-shot"):
        print(f"\n[zero-shot] evaluating {len(test_rows)} test rows ...")
        zs_preds = predict([r["sentence"] for r in test_rows])
        acc, mf1, wf1 = macro_metrics([r["label"] for r in test_rows], zs_preds)
        results["finbert_zero_shot"] = {"accuracy": acc, "macro_f1": mf1,
                                       "weighted_f1": wf1, "n": len(test_rows)}
        print(f"[zero-shot] acc={acc*100:.2f}%  macroF1={mf1:.4f}  wF1={wf1:.4f}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "finbert_metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    if args.mode in ("all", "ft") and not args.no_ft:
        print("\n[fine-tune] loading training data ...")
        train_rows = load_csv(TRAIN_CSV)
        val_rows = load_csv(VAL_CSV)

        def prep(rows):
            texts = [r["sentence"] for r in rows]
            ys = torch.tensor([OUR_LABELS.index(r["label"]) for r in rows])
            ids, mask = encode(texts)
            return TensorDataset(ids, mask, ys)

        train_ds = prep(train_rows)
        val_ds = prep(val_rows)
        g = torch.Generator().manual_seed(SEED)
        train_dl = DataLoader(train_ds, batch_size=FT_BATCH, shuffle=True,
                              generator=g, drop_last=False)
        val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH)

        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=0.01)
        total_steps = len(train_dl) * args.epochs
        if get_linear_schedule_with_warmup is not None:
            sched = get_linear_schedule_with_warmup(optim, int(0.06 * total_steps),
                                                    total_steps)
        else:
            sched = None
        for ep in range(args.epochs):
            model.train()
            running, seen = 0.0, 0
            for ids, mask, ys in train_dl:
                ids, mask, ys = ids.to(device), mask.to(device), ys.to(device)
                out = model(input_ids=ids, attention_mask=mask)
                reord = torch.zeros_like(out.logits)
                for hf_i in range(len(hf_labels)):
                    reord[:, hf2our[hf_i]] = out.logits[:, hf_i]
                loss = torch.nn.functional.cross_entropy(reord, ys)
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
                    reord = torch.zeros_like(logits)
                    for hf_i in range(len(hf_labels)):
                        reord[:, hf2our[hf_i]] = logits[:, hf_i]
                    correct += (reord.argmax(-1).cpu() == ys).sum().item()
                    total += len(ys)
            print(f"[fine-tune] epoch {ep+1}/{args.epochs}  "
                  f"train loss {running/seen:.4f}  val acc {correct/total*100:.2f}%")

        print(f"\n[fine-tune] evaluating {len(test_rows)} test rows ...")
        ft_preds = predict([r["sentence"] for r in test_rows])
        acc, mf1, wf1 = macro_metrics([r["label"] for r in test_rows], ft_preds)
        results["finbert_fine_tuned"] = {"accuracy": acc, "macro_f1": mf1,
                                         "weighted_f1": wf1, "n": len(test_rows),
                                         "epochs": args.epochs, "lr": FT_LR,
                                         "batch": FT_BATCH, "seed": SEED}
        print(f"[fine-tune] acc={acc*100:.2f}%  macroF1={mf1:.4f}  wF1={wf1:.4f}")

        ckpt_dir = OUT_DIR / "finbert_ft_checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Fine-tuned checkpoint saved to: {ckpt_dir}")

    if args.smoke:
        print("\nSMOKE TEST COMPLETE - environment OK.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "finbert_per_row.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "label", "finbert_zs_pred", "finbert_ft_pred"])
        for r, z, t in zip(test_rows, zs_preds or [None]*len(test_rows),
                           ft_preds or [None]*len(test_rows)):
            w.writerow([r["sentence"], r["label"], z, t])

    mcn = {}
    if "qwen_zs_label" in test_rows[0]:
        qs = [r["qwen_zs_label"] for r in test_rows]
        truth = [r["label"] for r in test_rows]
        for name, preds in [("finbert_zero_shot", zs_preds),
                            ("finbert_fine_tuned", ft_preds)]:
            if preds is None:
                continue
            ok_a = [p == t for p, t in zip(preds, truth)]
            ok_b = [p == t for p, t in zip(qs, truth)]
            b, c, chi2, p_exact = mcnemar_exact(ok_a, ok_b)
            mcn[f"{name}_vs_qwen_zs"] = {"b": b, "c": c, "chi2_cc": chi2,
                                        "p_exact": p_exact}
            print(f"\n[McNemar] {name} vs Qwen ZS: b={b} c={c} "
                  f"chi2={chi2:.2f} p_exact={p_exact:.4f}")
        results["mcnemar"] = mcn

    with open(OUT_DIR / "finbert_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "finbert_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "accuracy", "macro_f1", "weighted_f1", "n"])
        for k, v in results.items():
            if k == "mcnemar":
                continue
            w.writerow([k, f"{v['accuracy']:.4f}", f"{v['macro_f1']:.4f}",
                        f"{v['weighted_f1']:.4f}", v["n"]])
    print(f"\nWritten: {OUT_DIR}/finbert_per_row.csv, finbert_metrics.json/csv"
          + (", finbert_mcnemar inside finbert_metrics.json" if mcn else ""))


if __name__ == "__main__":
    main()