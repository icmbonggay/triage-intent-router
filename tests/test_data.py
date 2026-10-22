import pytest
from data import load_data

COLUMN_NAMES = ["text", "label", "label_text"]

@pytest.mark.parametrize(
    "split, row_count",
    [
        ("train", 9993),
        ("test", 3076),
    ]
)
def test_split_row_count_is_correct(split, row_count):
    df = load_data(split)
    assert len(df) == row_count

@pytest.mark.parametrize("split", ["train","test"])
def test_columns_are_exactly_text_label_labeltext_only(split):
    df = load_data(split)
    assert COLUMN_NAMES == df.columns.to_list()

def test_bad_input_raises_value_error():
    with pytest.raises(ValueError):
        load_data("hello")


