#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    source_csv = Path("Datasets/sentences_allagree.csv")
    out_dir = Path("Datasets/financial_phrasebank")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source_csv.exists():
        raise FileNotFoundError(f"Input dataset not found: {source_csv}")

    df = pd.read_csv(source_csv)
    required = {"sentence", "label"}
    if not required.issubset(df.columns):
        raise ValueError("Input CSV must contain columns: sentence,label")

    df = df[["sentence", "label"]].dropna().copy()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    train_path = out_dir / "train.csv"
    val_path = out_dir / "validation.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Split complete.")
    print(f"Train: {len(train_df)} -> {train_path}")
    print(f"Validation: {len(val_df)} -> {val_path}")
    print(f"Test: {len(test_df)} -> {test_path}")


if __name__ == "__main__":
    main()
