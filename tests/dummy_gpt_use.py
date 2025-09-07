import tiktoken
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel
import torch

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

"""
We will prepare the input data and initialize a new GPT model to illustrate
its usage. Building on our coding of the tokenizer, let’s now con-
sider a high-level overview of how data flows in and out of a GPT model,
"""

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch)
print(batch)

# INFO: Next, we initialize a new 124-million-parameter DummyGPTModel instance and feed it
# the tokenized batch:
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

logits = model(batch)
# INFO: The output tensor has two rows corresponding to the two text samples. Each text sam-
# ple consists of four tokens; each token is a 50,257-dimensional vector, which matches
# the size of the tokenizer’s vocabulary.
# The embedding has 50,257 dimensions because each of these dimensions refers to
# a unique token in the vocabulary. When we implement the postprocessing code, we
# will convert these 50,257-dimensional vectors back into token IDs, which we can then
# decode into words.
print(f"Output shape: {logits.shape}")
print(f"Output: {logits}")
