import torch
import pytest
from llm_from_scratch.gpt_architecture.text_generation import generate_text_simple
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel
from tiktoken import get_encoding

# Pytest fixture for the GPT model configuration
@pytest.fixture
def gpt_config():
    return {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

# Pytest fixture for the GPT model
@pytest.fixture
def gpt_model(gpt_config):
    return GPTModel(gpt_config)

# Pytest fixture for the tokenizer
@pytest.fixture
def tokenizer():
    return get_encoding("cl100k_base")

# Test the text generation function
def test_generate_text_simple(gpt_model, tokenizer):
    gpt_model.eval()
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # Generate text
    generated_tokens = generate_text_simple(
        model=gpt_model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=gpt_model.pos_emb.weight.shape[0],
    )

    # Assertions
    assert generated_tokens.shape[1] == len(encoded) + 6
    assert generated_tokens.shape[0] == 1

    # Decode and check the output
    decoded_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())
    assert decoded_text.startswith(start_context)