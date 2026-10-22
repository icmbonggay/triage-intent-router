"""Find exact duplicates in Banking77 after normalization.

Run from the repo root:
    PYTHONPATH=src python scripts/find_duplicates.py
"""

import pandas as pd

from data import load_data
from duplicates import find_duplicate

SPLITS = ("train", "test")
EXAMPLES_TO_SHOW = 5


def load_splits() -> dict[str, pd.DataFrame]:
    """Load each Banking77 split into its own DataFrame.

    Returns:
        A dict mapping each split name to its DataFrame.
    """
    return {name: load_data(name) for name in SPLITS}


def combine_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack all splits into one DataFrame with a "split" column.

    Original indices are kept, so a row is identified by its split and
    index together.

    Args:
        splits: A dict mapping each split name to its DataFrame.

    Returns:
        One DataFrame containing every row from every split.
    """
    frames = [df.assign(split=name) for name, df in splits.items()]
    return pd.concat(frames)


def label_of(row: dict, splits: dict[str, pd.DataFrame]) -> str:
    """Look up the label_text of a duplicate-group row in its own split.

    Args:
        row: A row from a duplicate group, with "split" and "index" keys.
        splits: A dict mapping each split name to its DataFrame.

    Returns:
        The row's label_text.
    """
    return splits[row["split"]].loc[row["index"], "label_text"]


def print_examples(
    title: str,
    groups: dict[str, list[dict]],
    splits: dict[str, pd.DataFrame],
) -> None:
    """Print a count of the groups and the first few in detail.

    Args:
        title: Heading for this section of the report.
        groups: Duplicate groups, as returned by find_duplicate.
        splits: A dict mapping each split name to its DataFrame.
    """
    print(f"\n{title}: {len(groups)}")
    for key, rows in list(groups.items())[:EXAMPLES_TO_SHOW]:
        print(f"  {key!r}")
        for row in rows:
            print(
                f"    [{row['split']} #{row['index']}] "
                f"{row['text']!r} -> {label_of(row, splits)}"
            )


def main() -> None:
    splits = load_splits()
    combined = combine_splits(splits)
    duplicates = find_duplicate(combined)

    rows_involved = sum(len(rows) for rows in duplicates.values())

    cross_split = {
        key: rows
        for key, rows in duplicates.items()
        if len({row["split"] for row in rows}) > 1
    }
    conflicting_labels = {
        key: rows
        for key, rows in duplicates.items()
        if len({label_of(row, splits) for row in rows}) > 1
    }

    print(f"Rows checked: {len(combined)}")
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Rows in duplicate groups: {rows_involved}")
    print(f"Extra copies (rows minus groups): {rows_involved - len(duplicates)}")
    print(f"Empty or punctuation-only group present: {'' in duplicates}")

    print_examples("Cross-split groups (train/test leakage)", cross_split, splits)
    print_examples("Groups with conflicting labels", conflicting_labels, splits)
    print_examples("First few duplicate groups", duplicates, splits)


if __name__ == "__main__":
    main()