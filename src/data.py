from typing import Literal
from datasets import load_dataset
import pandas as pd


DATASET_ID = "mteb/banking77"
COMMIT = "18072d2685ea682290f7b8924d94c62acc19c0b2"

def load_data(split: Literal["train", "test"] = "train") -> pd.DataFrame:
    """Load a dataset from mteb/banking77 based on the split and pinned commit.

    Args:
        split: The split to be used in loading the dataset ('train' or 'test').
    
    Returns:
        The dataframe with the given split containing features such as text, label, and label_text.

    Raises:
        ValueError: If split is not 'train' or 'test'.
    """
 
    if split not in ["train", "test"]:
        raise ValueError(f"Invalid split '{split}'. Must be either 'train' or 'test'.")

    dataset = load_dataset(DATASET_ID, split=split, revision=COMMIT)
    return dataset.to_pandas()

