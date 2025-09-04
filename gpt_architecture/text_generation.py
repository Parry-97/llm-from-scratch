import torch
from torch import Tensor
from .dummy_gpt_model import DummyGPTModel

"""This code demonstrates a simple implementation of a generative loop for a lan-
guage model using PyTorch. It iterates for a specified number of new tokens to be
generated, crops the current context to fit the model’s maximum context size, com-
putes predictions, and then selects the next token based on the highest probability
prediction"""


def generate_text(
    model: DummyGPTModel,
    idx: Tensor,  # INFO: this has shape (batch_size, num_tokens)
    max_new_tokens: int,
    context_size: int,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[
            :, -context_size:
        ]  # NOTE: we pick only the last context_size tokens

        with torch.no_grad():
            logits: Tensor = model(idx_cond)

        logits = logits[
            :, -1, :
        ]  # NOTE: we pick only the last logit, with shape (batch, vocab_size)

        # WARN: The softmax function is monotonic, meaning it preserves the order of its
        # inputs when transformed into outputs. So, in practice, the
        # softmax step is redundant since the position with the highest score in the softmax out-
        # put tensor is the same position in the logit tensor. In other words, we could apply the
        # torch.argmax function to the logits tensor directly and get identical results.

        probas = torch.softmax(
            logits, dim=-1
        )  # we calculate the normalized probabilities. Shape: (batch_size, vocab_size)

        # NOTE: keepdim only ensure the NUMBER OF DIMENSIONS is the same NOT the shape, which of course
        # here is reduced in the column dimension (but that does not impact the cat operation afterwards).
        # Check its documentation
        idx_next = torch.argmax(
            probas, dim=-1, keepdim=True
        )  # NOTE: we pick the most probable. Shape: (batch_size, 1)
        idx = torch.cat([idx, idx_next], dim=-1)

    return idx
