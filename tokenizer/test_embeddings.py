# coding: utf-8
import torch
from text_splitting import raw_text
from gpt_dataset import create_dataloader_v1

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
# Suppose we have a vocabulary_size of only 6 words and we
# want to create embeddings of size 3
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
embedding_layer.weight
"""The weight matrix of the embedding layer contains small
random values. These values are optimized during LLM training
as part of the LLM optimization itself"""
embedding_layer(input_ids)

""" Let’s consider more realistic and useful embedding sizes and encode the input tokens into a
 256-dimensional vector representation, which is smaller than what the original GPT-3
 model used (in GPT-3, the embedding size is 12,288 dimensions) but still reasonable
 for experimentation"""
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

"""
if we sample data from the data loader,
we embed each token in each batch into a 256-dimensional vector.
"""
max_len = 4
data_loader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_len, stride=max_len, shuffle=False
)
data_iter = iter(data_loader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("Inputs shape:", inputs.shape)

"""
Let's now embed the token IDs into a 256-dimensional vector
"""

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

"""
For a GPT model’s absolute embedding approach, we just need to create another
embedding layer that has the same embedding dimension as the token_embedding_
layer
"""
context_length = max_len
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

"""
We can now add these directly to the token embeddings, where PyTorch will add
the 4 × 256–dimensional pos_embeddings tensor to each 4 × 256–dimensional token
embedding tensor in each of the eight batches
"""
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
