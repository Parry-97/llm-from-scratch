from typing import Tuple
from pandas import DataFrame
import pandas as pd


def balance_dataset(df: DataFrame) -> DataFrame:
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([df[df["Label"] == "spam"], ham_subset])  # pyright: ignore


def random_split(
    df: DataFrame, train_frac: float, validation_frac: float
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    # INFO: Shuffle the entire dataframe
    df = df.sample(frac=1, random_state=123)

    # NOTE: Calculate the split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # NOTE: Split the dataframe into train, validation, and test sets
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df  # pyright: ignore
