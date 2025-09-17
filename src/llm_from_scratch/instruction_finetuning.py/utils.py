import json
from functools import partial
import torch
from typing import Tuple

from torch import Tensor

# NOTE: Instruction fine-tuning involves training a model on a dataset where the input-output
# pairs, like those we extracted from the JSON file, are explicitly provided.
# There are various methods to format these entries for LLMs and are known
# as prompy styles.
#
# NOTE: Alpaca was one of the early LLMs to publicly detail its instruction fine-tuning pro-
# cess. The rest of this chapter uses the Alpaca prompt style since it is one of the most
# popular ones, largely because it helped define the original approach to fine-tuning.


def load_json(file_path: str) -> list:
    with open(file_path, "r") as f:
        return json.load(f)


def partition_dataset(data: list[dict]) -> Tuple[list[dict], list[dict], list[dict]]:
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    return train_data, val_data, test_data


def format_input(entry: dict) -> str:
    """
    Takes a dictionary `entry` as input and constructs a formatted string.

    Args:
        entry (dict): The entry from the JSON file.

    Returns:
        str: The formatted input.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def custom_collate(
    batch: list[list[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_len: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    # NOTE: Finds the longest sequence in the batch + 1.
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()

        # NOTE: This is needed to later append the pad_token
        # in the targets
        new_item += [pad_token_id]

        # INFO: Pads and prepares the input
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])  # NOTE: Removes the extra padded token
        # WARN: We retain the extra padded token because the targets are shifted
        # Retaining it allows the LLM to learn when to generate an end-of-text
        # token in response to the instructions, which we use as an indicator
        # that the generated response is complete.
        targets = torch.tensor(padded[1:])  # NOTE: Shifts the inputs to get the targets

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            # NOTE: We replace pad token except the first
            # with -100 so that they are ignored during the
            # cross entropy loss calculation. -100 is the
            # default ignore_index for CrossEntropyLoss
            targets[indices[1:]] = ignore_index

        if allowed_max_len is not None:
            inputs = inputs[:allowed_max_len]
            targets = targets[:allowed_max_len]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # INFO: Converts the list of inputs/targets into a batch tensor
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# NOTE: The custom_collate function includes code
# to move the input and target tensors to a specified device,
# which can be either "cpu" or "cuda" (for NVIDIA
# GPUs) or, optionally, "mps" for Macs with Apple Silicon chips.
# Previously, we moved the data onto the target device (for example, the GPU memory
# when device="cuda") in the main training loop. Having this as part of the collate
# function offers the advantage of performing this device transfer process as a back-
# ground process outside the training loop, preventing it from blocking the GPU
# during model training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(custom_collate, device=device, allowed_max_len=1024)
