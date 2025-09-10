"""Test script to verify that TransformerBlock correctly uses MultiHeadAttention"""

import torch
import pytest
from llm_from_scratch.gpt_architecture.transformer import TransformerBlock

# Configuration for the transformer
@pytest.fixture
def transformer_config():
    """Fixture providing transformer configuration parameters."""
    return {
        "emb_dim": 768,
        "n_heads": 12,
        "context_length": 1024,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

# Test the TransformerBlock with MultiHeadAttention
def test_transformer_block(transformer_config):
    # Create a transformer block
    transformer_block = TransformerBlock(transformer_config)

    # Create sample input tensor
    batch_size = 2
    seq_length = 100
    input_tensor = torch.randn(batch_size, seq_length, transformer_config["emb_dim"])

    # Forward pass
    output = transformer_block(input_tensor)

    # Verify output shape
    assert output.shape == input_tensor.shape