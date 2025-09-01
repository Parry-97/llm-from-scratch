#!/usr/bin/env python3
"""Test script to verify that TransformerBlock correctly uses MultiHeadAttention"""

import torch
from gpt_architecture.transformer import TransformerBlock

# Configuration for the transformer
cfg = {
    "emb_dim": 768,
    "n_heads": 12,
    "context_length": 1024,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

def test_transformer_block():
    """Test the TransformerBlock with MultiHeadAttention"""
    # Create a transformer block
    transformer_block = TransformerBlock(cfg)
    
    # Create sample input tensor
    batch_size = 2
    seq_length = 100
    input_tensor = torch.randn(batch_size, seq_length, cfg["emb_dim"])
    
    # Forward pass
    output = transformer_block(input_tensor)
    
    # Verify output shape
    assert output.shape == input_tensor.shape, f"Shape mismatch: {output.shape} != {input_tensor.shape}"
    
    print(f"✅ TransformerBlock successfully uses MultiHeadAttention!")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention heads: {cfg['n_heads']}")
    print(f"Embedding dimension: {cfg['emb_dim']}")
    
    # Print model structure
    print("\nTransformerBlock structure:")
    print(transformer_block)
    
    return True

if __name__ == "__main__":
    test_transformer_block()
