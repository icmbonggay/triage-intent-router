import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from near_duplicates import DEFAULT_THRESHOLD, assign_near_duplicate_groups


def split_by_group(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train and test so no duplicate group is in both.

    Rows are grouped with `assign_near_duplicate_groups`, so exact
    duplicates and near-duplicates always land on the same side.

    Args:
        df: DataFrame with a "text" column holding the messages.
        test_size: Fraction of groups (not rows) to put in test.
        random_state: Seed, so the same inputs always give the same split.
        threshold: Minimum cosine similarity for two messages to count as
            near-duplicates.

    Returns:
        A (train_df, test_df) tuple of the rows on each side.
    """
    groups = assign_near_duplicate_groups(df, threshold)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    result = splitter.split(X=df, groups=groups)

    train_positions, test_positions = next(result)

    train_df = df.iloc[train_positions]
    test_df = df.iloc[test_positions]

    return train_df, test_df
