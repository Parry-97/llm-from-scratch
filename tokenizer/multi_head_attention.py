import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with efficient attention computation."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(
            d_out, d_out, bias=qkv_bias
        )  # use a linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape
        keys: torch.Tensor = self.W_key(x)
        queries: torch.Tensor = self.W_query(x)
        values: torch.Tensor = self.W_value(x)

        """
        We split the matrices by adding num_heads dimension. Then we unroll the last dimension
        (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        """
        keys: torch.Tensor = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values: torch.Tensor = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries: torch.Tensor = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )

        """ Transpose from shape (b, num_tokens, num_heads, head_dim) to 
        (b, num_heads, num_tokens, head_dim) """
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        """
        Computes dot product to get attention scores for each head
        """
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights: torch.Tensor = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec: torch.Tensor = self.out_proj(context_vec)
        return context_vec
