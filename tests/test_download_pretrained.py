"""Test loading pretrained GPT-2 weights and generating text."""

import torch
import tiktoken
from unittest.mock import patch
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel
from llm_from_scratch.gpt_architecture.text_generation import generate_text
from llm_from_scratch.pretraining.utils import (
    load_weights_into_gpt,
    text_to_token,
    token_ids_to_text,
)

import pytest


@pytest.fixture
def gpt_config():
    return {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }


# Mocked settings and params for the GPT-2 model
@pytest.fixture
def mock_gpt2_data():
    settings = {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    }
    params = {
        "blocks": [
            {
                "attn": {
                    "c_attn": {"w": torch.randn(768, 2304), "b": torch.randn(2304)},
                    "c_proj": {"w": torch.randn(768, 768), "b": torch.randn(768)},
                },
                "ln_1": {"b": torch.randn(768), "g": torch.randn(768)},
                "ln_2": {"b": torch.randn(768), "g": torch.randn(768)},
                "mlp": {
                    "c_fc": {"w": torch.randn(768, 3072), "b": torch.randn(3072)},
                    "c_proj": {"w": torch.randn(3072, 768), "b": torch.randn(768)},
                },
            }
            for _ in range(12)
        ],
        "wpe": torch.randn(1024, 768),
        "wte": torch.randn(50257, 768),
        "ln_f": {"b": torch.randn(768), "g": torch.randn(768)},
        "g": torch.randn(768),  # Final layer norm scale
        "b": torch.randn(768),  # Final layer norm shift
    }
    return settings, params


def test_generate_text_with_mocked_model(gpt_config, tokenizer, mock_gpt2_data):
    # Mock the download_and_load_gpt2 function
    with patch("llm_from_scratch.gpt_download.download_and_load_gpt2") as mock_download:
        mock_download.return_value = mock_gpt2_data

        # Initialize the model with the mocked config
        gpt = GPTModel(gpt_config)
        gpt.eval()

        # Load the mocked weights
        _, params = mock_download.return_value
        load_weights_into_gpt(gpt, params)
        gpt.to("cpu")

        # Set seed for reproducibility
        torch.manual_seed(123)

        # Generate text
        token_ids = generate_text(
            model=gpt,
            idx=text_to_token("Every effort moves you", tokenizer),
            max_new_tokens=10,
            context_size=gpt_config["context_length"],
            top_k=50,
            temperature=1.5,
        )

        # Decode the generated tokens
        generated_text = token_ids_to_text(token_ids, tokenizer)

        # Assertions to verify the output
        assert isinstance(generated_text, str)
        assert "Every effort moves you" in generated_text
