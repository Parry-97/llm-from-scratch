import torch


class CausalAttention(torch.nn.Module):
    """Example of a causal attention layer."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)

        """
        Using `register_buffer` buffers are automatically moved to the appropriate
        device, by PyTorch, along with our model, which will be relevant when training.
        This means we don't need to manually ensure tensors are on the correct device.
        It `attaches` it to the module.

        Furthermore notice that the dimesions of the mask are based on the context length.
        Imagine a matrix of attentions weights where we match them one by one by row
        and column so that a given query can only access the ones before it since the
        rest(after the diagonal) are masked out.
        """
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape
        keys: torch.Tensor = self.W_key(x)
        queries: torch.Tensor = self.W_query(x)
        values: torch.Tensor = self.W_value(x)

        attn_scores = queries @ keys.transpose(
            1, 2
        )  # we tranpose dimensions 1 and 2, keeping the batch dimension at position 0

        attn_scores.masked_fill_(
            self.get_buffer("mask").bool()[:num_tokens, :num_tokens], -torch.inf
        )  # the bool mask is retrieved from the bigger context sized mask matrix, that's why we have [:num_tokens, :num_tokens]

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(
            attn_weights
        )  # we apply further dropout to avoid overfitting and co-dependence

        context_vec = attn_weights @ values
        return context_vec
