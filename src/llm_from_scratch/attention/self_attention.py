import torch


class SelfAttention(torch.nn.Module):
    """
    Python class for a self-attention layer.
    The class inherits from the torch.nn.Module class, which
    is a fundamental building block of Pytorch models that provide
    necessary functionalities.

    NOTE: In GPT-like models, the input and output dimension are usually
    the same but here we are using different values to better follow the computation
    """

    def __init__(self, d_in=3, d_out=2):
        """
        Initialize the self-attention layer with the given input and output dimensions.
        """
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key  # shape: [6,2]
        queries = x @ self.W_query  # shape: [6,2]
        values = x @ self.W_value  # shape: [6,2]
        attn_scores = queries @ keys.T  # omega with shape: [6,2] @ [2,6] =  [6,6]
        """
        the softmax function ensures that the attention weights are always posi-
        tive. This makes the output interpretable as probabilities or relative importance,
        where higher weights indicate greater importance.
        """
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values  # shape: [6,6] @ [6,2] = [6,2]
        return context_vec


class SelfAttention_v2(torch.nn.Module):
    """
    Python class for a self-attention layer.
    The class inherits from the torch.nn.Module class, which
    is a fundamental building block of Pytorch models that provide
    necessary functionalities.

    NOTE: In GPT-like models, the input and output dimension are usually
    the same but here we are using different values to better follow the computation
    """

    def __init__(self, d_in=3, d_out=2, qkv_bias=False):
        """
        We can improve the implementation by using the Linear layers, which effectively perform
        matrix multiplication when the bias units are disabled
        """
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
