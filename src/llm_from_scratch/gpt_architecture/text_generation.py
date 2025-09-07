import torch
from torch import Tensor
from .dummy_gpt_model import GPTModel

"""This code demonstrates a simple implementation of a generative loop for a lan-
guage model using PyTorch. It iterates for a specified number of new tokens to be
generated, crops the current context to fit the model’s maximum context size, com-
putes predictions, and then selects the next token based on the highest probability
prediction"""


def generate_text_simple(
    model: GPTModel,
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


def generate_text(
    model: GPTModel,
    idx: Tensor,  # INFO: this has shape (batch_size, num_tokens)
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
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

        if top_k is not None:
            top_logits, _ = torch.topk(
                logits, top_k
            )  # NOTE: Filters logits with topk sampling
            min_val = top_logits[:, -1]  # NOTE: getting the minimum valued logit

            # WARN: When condition is True for the index, yield input, otherwise yield other
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
        if temperature > 0.0:  # NOTE: Applies temperature scaling to the logits
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
        else:
            idx_next = torch.argmax(
                logits, dim=-1, keepdim=True
            )  # NOTE: carries out greedy next token selection as before when temperature sampling is disabled

        if idx_next == eos_token_id:  # Stop generating early if eos_token_id is reached
            break
        idx = torch.cat([idx, idx_next], dim=-1)

    return idx


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
