import pandas as pd

from near_duplicates import (
    assign_near_duplicate_groups,
    find_similar_pairs,
    groups_from_pairs,
)


# groups_from_pairs: pairs -> groups (connected components)

def test_no_pairs_gives_every_item_its_own_group():
    assert groups_from_pairs(3, []) == [0, 1, 2]


def test_empty_input_gives_no_groups():
    assert groups_from_pairs(0, []) == []


def test_chain_of_pairs_ends_up_in_one_group():
    # A~B and B~C, but no direct A~C pair. All three must share a group.
    labels = groups_from_pairs(4, [(0, 1), (1, 2)])

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_two_separate_pairs_give_two_groups():
    labels = groups_from_pairs(4, [(0, 1), (2, 3)])

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


# find_similar_pairs: texts -> similar pairs

def test_typo_is_found_as_a_similar_pair():
    texts = ["my card still has not arrived", "my card still has not arived",
             "how do i change my pin"]

    pairs = find_similar_pairs(texts, threshold=0.8)

    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_unrelated_messages_are_not_paired():
    texts = ["how do i change my pin", "where is the nearest atm",
             "why was i charged a fee"]

    assert find_similar_pairs(texts, threshold=0.8) == []


def test_each_pair_is_reported_once_with_smaller_position_first():
    texts = ["how do i top up", "How do I top up?", "how do i top-up"]

    pairs = find_similar_pairs(texts, threshold=0.8)

    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (0, 2), (1, 2)]


def test_similarity_is_at_least_the_threshold():
    texts = ["my card still has not arrived", "my card still has not arived"]

    for _, _, similarity in find_similar_pairs(texts, threshold=0.8):
        assert similarity >= 0.8


def test_punctuation_only_texts_are_skipped_without_crashing():
    assert find_similar_pairs(["???", "!!!"], threshold=0.8) == []


# assign_near_duplicate_groups: DataFrame -> one label per row

def test_one_label_per_row():
    df = pd.DataFrame({"text": ["a b c", "d e f", "g h i"]})

    assert len(assign_near_duplicate_groups(df)) == len(df)


def test_exact_and_near_duplicates_share_a_group():
    df = pd.DataFrame({"text": [
        "My card still has not arrived",   # 0
        "my card still has not arrived!",  # 1: exact duplicate of 0
        "my card still has not arived",    # 2: typo of 0
        "how do i change my pin",          # 3: unrelated
    ]})

    labels = assign_near_duplicate_groups(df, threshold=0.8)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_punctuation_only_rows_are_still_grouped_as_exact_duplicates():
    df = pd.DataFrame({"text": ["???", "!!!", "how do i change my pin"]})

    labels = assign_near_duplicate_groups(df)

    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
