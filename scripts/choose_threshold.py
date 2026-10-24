"""Print evidence for choosing the near-duplicate similarity threshold.

For each candidate threshold it prints how many pairs count as
near-duplicates, how many of those pairs have different labels (a sign
the threshold is too loose), and sample pairs just above it to read.

Run from the repo root:
    PYTHONPATH=src python scripts/choose_threshold.py
"""

import random

import pandas as pd

from data import load_data
from near_duplicates import find_similar_pairs
from normalize import normalize_data

THRESHOLDS = (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
EXAMPLES_PER_BAND = 8
SEED = 0


def load_unique_messages() -> pd.DataFrame:
    """Load both splits and keep one row per normalized text.

    Exact duplicates are already grouped by exact matching, so they are
    dropped here to show only pairs that exact matching would miss.
    """
    df = pd.concat([load_data("train"), load_data("test")], ignore_index=True)
    df["normalized"] = df["text"].map(normalize_data)
    return df.drop_duplicates("normalized").reset_index(drop=True)


def main() -> None:
    df = load_unique_messages()
    texts = df["text"].tolist()
    labels = df["label_text"].tolist()

    # Search once at the loosest threshold, then filter for the stricter ones.
    pairs = find_similar_pairs(texts, min(THRESHOLDS))
    print(f"Unique messages: {len(df)}")

    print("\nthreshold  pairs  different-label pairs")
    for threshold in THRESHOLDS:
        kept = [p for p in pairs if p[2] >= threshold]
        different = [p for p in kept if labels[p[0]] != labels[p[1]]]
        share = len(different) / len(kept) if kept else 0
        print(f"{threshold:>9}  {len(kept):>5}  {len(different):>5} ({share:.0%})")

    rng = random.Random(SEED)
    bands = list(zip(THRESHOLDS, THRESHOLDS[1:] + (1.01,)))
    for low, high in bands:
        band = [p for p in pairs if low <= p[2] < high]
        print(f"\n=== similarity {low} to {high}: {len(band)} pairs ===")
        for i, j, similarity in rng.sample(band, min(EXAMPLES_PER_BAND, len(band))):
            same = "same label" if labels[i] == labels[j] else "DIFFERENT"
            print(f"  {similarity:.2f}  [{same}]")
            print(f"        {texts[i]!r} -> {labels[i]}")
            print(f"        {texts[j]!r} -> {labels[j]}")


if __name__ == "__main__":
    main()
