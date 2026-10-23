from collections import defaultdict

import pandas as pd

from normalize import normalize_data


def find_duplicate(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Group rows whose text is identical after normalization.

    Each row's text is passed through `normalize`, and rows that produce
    the same normalized text are grouped together. Only groups with two
    or more rows are returned.

    Args:
        df: DataFrame with a "text" column holding the messages and a
            "split" column naming the split each row came from.

    Returns:
        A dict mapping each normalized text to a list of the rows that
        share it, in index order. Each row is a dict with "index",
        "split", and "text" keys. Returns an empty dict if no
        duplicates are found.
    """

    groups = defaultdict(list)

    for row in df.itertuples():
        key = normalize_data(row.text)
        groups[key].append({
            "index": row.Index,
            "split": row.split,
            "text": row.text,
        })

    return {key: rows for key, rows in groups.items() if len(rows) > 1}


def assign_groups(df: pd.DataFrame) -> list:
    """Label each row with its normalized text, so duplicates share a label.

    Every row gets a label, including rows with no duplicate. Two rows
    get the same label exactly when their text is identical after
    `normalize_data`.

    Args:
        df: DataFrame with a "text" column holding the messages.

    Returns:
        A list with one label per row of `df`, in the same order as the
        rows. Each label is that row's normalized text.
    """
    groups = []

    for row in df.itertuples():
        key = normalize_data(row.text)
        groups.append(key)
    
    
    return groups

