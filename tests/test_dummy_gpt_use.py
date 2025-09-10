import torch
import pytest
import tiktoken
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel


def test_gpt_model_output_shape(gpt_config, tokenizer):
    torch.manual_seed(123)
    model = GPTModel(gpt_config)

    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch)

    logits = model(batch)

    assert logits.shape == (2, 4, gpt_config["vocab_size"])