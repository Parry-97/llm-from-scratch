import sys
import os

import torch
from attention.multi_head_attention import MultiHeadAttention
from torch import nn


class TransformerBlock(nn.Module):
    """Transformer Block implementation"""

    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg.get("qkv_bias", False),
        )
        self.feed_forward = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention block with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_resid(x)
        x = x + shortcut

        # Feed-forward block with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x


class FeedForward(nn.Module):
    """Feed-forward network for transformer block"""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x):
        return self.layers(x)
