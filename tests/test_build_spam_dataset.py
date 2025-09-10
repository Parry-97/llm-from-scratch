import os
import pandas as pd
import pytest
from llm_from_scratch.clf_finetuning.utils import balance_dataset, random_split

# Pytest fixture to create a dummy spam dataset
@pytest.fixture
def spam_data(tmp_path):
    data = {
        "Label": ["ham", "spam", "ham", "spam"],
        "Text": [
            "Go until jurong point, crazy..",
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
            "U dun say so early hor...",
            "FreeMsg Hey there darling it's been 3 week's now and no word back!",
        ],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "SMSSpamCollection.tsv"
    df.to_csv(file_path, sep="\t", header=False, index=False)
    return file_path

# Test the balance_dataset function
def test_balance_dataset(spam_data):
    df = pd.read_csv(spam_data, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = balance_dataset(df)
    assert len(balanced_df[balanced_df["Label"] == "spam"]) == len(
        balanced_df[balanced_df["Label"] == "ham"]
    )

# Test the random_split function and CSV writing
def test_random_split_and_save(spam_data, tmp_path):
    df = pd.read_csv(spam_data, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = balance_dataset(df)

    # Perform the split
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # Define output paths
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"

    # Save to CSV
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Verify that the files are created
    assert os.path.exists(train_path)
    assert os.path.exists(validation_path)
    assert os.path.exists(test_path)

    # Verify the contents of the CSV files
    train_read_df = pd.read_csv(train_path)
    assert not train_read_df.empty
    assert list(train_read_df.columns) == ["Label", "Text"]