# coding: utf-8
import torch
from simple_attention import inputs

# the input shape is [6,3]

x_2 = inputs[1]
d_in = inputs.shape[1]  # 3
d_out = 2
"""We will implement the self attention mechanism step by step by introducing the three trainable weight matrices Wq, Wk, Wv. These three matrices are used to project the embedded input tokens xi into query, key and value vectors respectively."""
# We initialize the three weight matrices
torch.manual_seed(123)
W_query = torch.nn.Parameter(
    torch.rand(d_in, d_out), requires_grad=False
)  # shape in [3,2]
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# We set requires_grad=False for now but if we were to use the weight matrices for model training we would of course set it to True
# We compute the query, key and value vectors
query_2 = x_2 @ W_query  # [1,3] @ [3, 2]
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
"""The output for the query results in a two-dimensional vector since we set the number of columns of the corresponding weight matrices via d_out=2."""
"""Even though  our temporary goal is only to compute the one context vector,z2, we still require the key and values vectors for all input elements as they are involved in computing the attention weights with respect to the query q2. We can obtain them via matmul"""
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
"""The next step is to compute attention scores"""
attn_scores = query_2 @ keys.T
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)
print(f"Shape of attn_scores_2: {attn_scores_2.shape}")
"""Now we can compute the attention weights by scaling the attention scores and using the ssoftmax function"""
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(
    attn_scores_2 / d_k**0.5, dim=-1
)  # however now we scale by dividing them by the square root of embedding dimesion of the keys
print(attn_weights_2)
"""The scaling by the square root of embedding dimension is the reason why this self attention mechanism is also called scaled dot attention. """
"""We now compute the context vector as a weighted sum over the value vectors"""
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
