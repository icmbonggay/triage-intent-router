from duplicates import assign_groups
from split import split_by_group
import pandas as pd


def test_no_group_in_both_train_and_test():
    df = pd.DataFrame({
        "text": ["How do I top up?", "my card is lost", "how do i top-up",
                 "My card is Lost!", "i got charged."],
    })

    for seed in range(20):
        train_df, test_df = split_by_group(df, 0.3, seed)

        train_groups = set(assign_groups(train_df))
        test_groups = set(assign_groups(test_df))

        assert train_groups & test_groups == set(), f"leak with seed {seed}"


def test_every_row_ends_up_in_exactly_one_split():
    df = pd.DataFrame({
        "text": ["How do I top up?", "my card is lost", "how do i top-up",
                 "My card is Lost!", "i got charged."],
    })

    train_df, test_df = split_by_group(df, 0.3, 0)

    assert len(train_df) + len(test_df) == len(df)
    assert sorted(list(train_df.index) + list(test_df.index)) == list(df.index)