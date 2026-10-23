import pandas as pd
from duplicates import assign_groups
from sklearn.model_selection import GroupShuffleSplit

def split_by_group(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = assign_groups(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    result = splitter.split(X=df, groups=groups)

    train_positions, test_positions = next(result)

    train_df = df.iloc[train_positions]
    test_df = df.iloc[test_positions]

    
    return train_df, test_df

