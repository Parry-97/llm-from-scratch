from typing import Tuple
from torch import Tensor
import tiktoken
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

"""
We first need to implement a PyTorch Dataset, which specifies how the data is
loaded and processed before we can instantiate the data loaders.
"""


class SpamDataset(Dataset):
    """
    This `SpamDataset` class handles several key tasks: it identifies the longest sequence in the
    training dataset, encodes the text messages, and ensures that all other sequences are
    padded with a padding token to match the length of the longest sequence.
    """

    def __init__(
        self,
        file_path: str | os.PathLike,
        tokenizer: tiktoken.Encoding,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ):
        """
        The `SpamDataset` class loads data from a `.csv` file, tokenizes
        the text using the GPT-2 tokenizer from tiktoken, and allows us to pad or truncate
        the sequences to a uniform length determined by either the longest sequence or a
        predefined maximum length. This ensures each input tensor is of the same size,
        which is necessary to create the batches in the training data loader
        """
        super().__init__()
        self.data = pd.read_csv(file_path)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        self.max_length = max_length or self._longest_encoded_length()

        # NOTE: Truncates the text to the maximum length
        self.encoded_texts = [
            text[:max_length] + [pad_token_id] * (self.max_length - len(text))
            for text in self.encoded_texts
        ]

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        encoded_text = self.encoded_texts[index]
        label = self.data["Label"][index]
        # Convert string labels to integers: "ham" -> 0, "spam" -> 1
        label_int = 1 if label == "spam" else 0
        return torch.tensor(encoded_text, dtype=torch.long), torch.tensor(
            label_int, dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        return max(len(encoded_text) for encoded_text in self.encoded_texts)
