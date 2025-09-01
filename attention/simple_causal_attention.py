# coding: utf-8
"""For many LLM tasks, we want the self-attention mechanism to consider only tokens that
appear prior to the current position when predicting the next token in a sequence.
Causal attention, also known as masked attention, is a specialized form of attention that restricts
a model to only consider previous and current inputs in a sequence when processing any given token"""
"""To achieve this we can mask out the attention weights above the diagonal and we normalize the non masked attention weights so that they sum up to 1 in each row"""
import torch
torch.manual_seed(789)
from self_attention import SelfAttention_v2
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],
    ]  # step
)
sa_v2 = SelfAttention_v2(3, 2)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
print(attn_weights)
"""We can implement the second step using Pytorch's tril function to create a mask where the values above the diagonal are zero"""
context_len = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_len, context_len))
print(mask_simple)
masked_simple = attn_weights * mask_simple
masked_simple
"""The final step is to renormalize once again so that the values in each row sum up to 1. We can achieve that by dividing each element by the sum in each row"""
rows_sum = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / rows_sum
print(masked_simple_norm)
