import torch
import tiktoken
from typing import Tuple
from torch import Tensor
from tiktoken import Encoding
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    NOTE:  With stride < max_len, sequences will overlap.
    In the default case (stride=128, max_len=256), each sequence overlaps by 50% with the previous one.
    Dataset Size: Smaller stride values create more training samples from the same data, which can help with:
    •  Better coverage of the text
    •  More training examples
    •  Learning different contexts for the same tokens
    Training Efficiency Trade-offs:
    •  Smaller stride: More samples, potentially better learning, but more computational cost
    •  Larger stride: Fewer samples, less redundancy, faster training
    •  stride = max_len: No overlap between sequences (non-overlapping windows)
    """

    def __init__(
        self,
        data: str,
        tokenizer: Encoding,
        max_len: int = 256,
        stride: int = 128,
    ):
        self.input_ids: list[Tensor] = []
        self.target_ids: list[Tensor] = []

        token_ids = tokenizer.encode(data)
        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i : i + max_len]
            target_chunk = token_ids[i + 1 : i + max_len + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __get__item(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # drop_last drops the last batch if it is shorter than the specified
        # batch size to prevent loss spikes during training
        num_workers=num_workers,
    )

    return dataloader
