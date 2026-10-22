from src.duplicates import find_duplicate
import pandas as pd




def test_empty_input_returns_empty_dict():
    df = pd.DataFrame(columns=["text", "split"])

    result = find_duplicate(df)

    assert result == {}

def test_no_duplicates_returns_empty_dict():
    df = pd.DataFrame({
        "text":  [ "I lost my card", "how do I withdraw?", "i got charged!"],
        "split": [ "train", "test", "train" ],
    })
    
    result = find_duplicate(df)
    assert result == {}

def test_case_and_punctuation_differences_are_grouped():
    df = pd.DataFrame({
        "text":  ["How do I top up?", "my card is lost", "how do i top-up"],
        "split": ["train", "train", "test"],
    })

    result = find_duplicate(df)

    assert result == {
        "how do i top up": [{"index": 0, "split": "train","text":"How do I top up?"}, {"index": 2, "split": "test", "text":"how do i top-up"} ]
    }


def test_two_separate_groups_are_both_reported():
    df = pd.DataFrame({
        "text":  ["How do I top up?", "my card is lost", "how do i top-up", "My card is Lost!", "i got charged."],
        "split": ["train", "train", "test", "train", "test"]
    })

    result = find_duplicate(df)

    assert result == {
        "how do i top up": [{"index": 0, "split": "train", "text":"How do I top up?"}, {"index": 2, "split": "test", "text":"how do i top-up"}],
        "my card is lost": [{"index": 1, "split": "train", "text":"my card is lost"}, {"index": 3, "split": "train", "text":"My card is Lost!"}]
    }


def test_two_punctuation_only_are_reported():
    df = pd.DataFrame({
        "text":  [ "???", "!!!!!", "Hello, I'm having a hard time"],
        "split": [ "train", "test", "train" ],
    })

    result = find_duplicate(df)

    assert result == {
        "": [{"index": 0, "split": "train","text":"???"}, {"index": 1, "split": "test", "text":"!!!!!"}]
    }

def test_group_with_three_members_keeps_all_of_them():
    df = pd.DataFrame({
        "text":  ["How do I top up?", "my card is lost", "how do i top-up", "HOW DO I TOP UP!!"],
        "split": ["train", "train", "test", "test"],
    })

    result = find_duplicate(df)

    assert result == {
        "how do i top up": [
            {"index": 0, "split": "train", "text": "How do I top up?"},
            {"index": 2, "split": "test", "text": "how do i top-up"},
            {"index": 3, "split": "test", "text": "HOW DO I TOP UP!!"},
        ]
    }