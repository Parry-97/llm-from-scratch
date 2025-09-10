import pandas as pd
import pytest
import torch
from llm_from_scratch.clf_finetuning.spam_dataset import SpamDataset


# Test the SpamDataset initialization and properties
def test_spam_dataset_initialization(tokenizer):
    train_path = "tests/train.csv"
    val_path = "tests/validation.csv"

    # Test with a specified max_length
    train_dataset = SpamDataset(
        file_path=train_path, tokenizer=tokenizer, max_length=20
    )
    assert train_dataset.max_length == 20

    # Test with max_length derived from the training set
    val_dataset = SpamDataset(
        file_path=val_path, tokenizer=tokenizer, max_length=train_dataset.max_length
    )
    assert val_dataset.max_length == train_dataset.max_length


# Test the __len__ and __getitem__ methods
def test_spam_dataset_len_and_getitem(tokenizer):
    train_path = "tests/train.csv"
    train_df = pd.read_csv(train_path)
    train_dataset = SpamDataset(file_path=train_path, tokenizer=tokenizer)

    # Check the length of the dataset
    assert len(train_dataset) == len(train_df)

    # Check the output of __getitem__
    idx = 0
    encoded_text, label = train_dataset[idx]

    assert isinstance(encoded_text, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert label.item() in [0, 1]  # Assuming ham is 0 and spam is 1
