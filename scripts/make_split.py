"""Re-split Banking77 so no duplicate or near-duplicate group leaks.

Combines the original train and test splits, groups duplicates and
near-duplicates, then splits by group and saves the result as CSV.

Run from the repo root:
    PYTHONPATH=src python scripts/make_split.py
"""

from pathlib import Path

import pandas as pd

from data import load_data
from near_duplicates import DEFAULT_THRESHOLD, assign_near_duplicate_groups
from split import split_by_group

TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = Path("data/splits")


def load_combined() -> pd.DataFrame:
    """Stack the original splits into one DataFrame.

    The index is reset so every row has a unique position, and the
    original split is kept in an "original_split" column.
    """
    frames = [load_data(name).assign(original_split=name) for name in ("train", "test")]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load_combined()
    groups = assign_near_duplicate_groups(df, DEFAULT_THRESHOLD)
    df["group_id"] = groups

    # How much did the original split leak?
    sides_per_group = df.groupby("group_id")["original_split"].nunique()
    leaked = int((sides_per_group > 1).sum())

    train_df, test_df = split_by_group(df, TEST_SIZE, RANDOM_STATE, DEFAULT_THRESHOLD)

    overlap = set(train_df["group_id"]) & set(test_df["group_id"])
    assert not overlap, f"{len(overlap)} groups appear in both splits"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    group_sizes = df["group_id"].value_counts()
    print(f"Threshold: {DEFAULT_THRESHOLD}")
    print(f"Rows: {len(df)}")
    print(f"Groups: {len(group_sizes)} ({int((group_sizes > 1).sum())} with 2+ rows)")
    print(f"Largest group: {group_sizes.max()} rows")
    print(f"Groups that leaked in the original split: {leaked}")
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows "
          f"({len(test_df) / len(df):.1%} of rows)")
    print(f"Intents in train: {train_df['label_text'].nunique()}, "
          f"in test: {test_df['label_text'].nunique()}")
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
