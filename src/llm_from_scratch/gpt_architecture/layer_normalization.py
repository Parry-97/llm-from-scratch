import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    This specific implementation of layer normalization operates on the last dimension of
    the input tensor x, which represents the embedding dimension (emb_dim). The vari-
    able eps is a small constant (epsilon) added to the variance to prevent division by zero
    during normalization. The scale and shift are two trainable parameters (of the
    same dimension as the input) that the LLM automatically adjusts during training if it
    is determined that doing so would improve the model’s performance on its training
    task. This allows the model to learn appropriate scaling and shifting that best suit the
    data it is processing.
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
