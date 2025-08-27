# coding: utf-8
import torch

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
"""The first step in implementing self-attention is to compute the intermediate
values $\omega$, referred to as attention scores. We determine these scores by computing the dot product of the query token with every other input token"""
query = inputs[1]  # here we use the second token as a query
attn_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores[i] = torch.dot(x_i, query)
print(attn_scores)
"""In the context of self attention, the dot product determines the extent to which each element in a sequence focuses on or "attends to" any other element"""
"""A best practice that is useful for interpretation and maintaining training stability is to normalize the attention scores"""
attn_scores_normalized = attn_scores / attn_scores.sum()
print(f"Attention scores: {attn_scores_normalized}")
print("Sum: ", attn_scores_normalized.sum())
"""In practice it is more common and advisable to use the softmax function for normalization. It offers more favorable properties during training and ensures that attention weights are positive, which makes the output more interpretable"""
attn_scores_normalized = torch.softmax(attn_scores, dim=0)
print(f"Attention scores: {attn_scores_normalized}")
attn_weights = torch.softmax(
    attn_scores, dim=0
)  # once normalized, scores are the weights we use to compute the context_vector
"""We can now calculate the context vector z2, which is none other than the weighted sum of all input vectors, obtained by multiplying each input vector by its corresponding attention weight"""
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)


for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights[i] * x_i
print(context_vec_2)
"""We can now compute all the context vectors instead of a single one"""
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
"""Each element in the tensor represents an attention score between each pair of inputs. But for loops are slow, and we can achieve the same results using matrix multiplication"""
attn_scores = inputs @ inputs.T
print(attn_scores)
attn_weights = torch.softmax(
    attn_scores, dim=-1
)  # we normalize across the last dimension which in this case are the columns so that values in rows sum up to 1
"""As the final step, we use the attention weights to compute all the context vectors via matrix multiplication"""
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
