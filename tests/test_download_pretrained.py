from gpt_download import download_and_load_gpt2
import tiktoken
from llm_from_scratch.gpt_architecture.text_generation import generate_text
from llm_from_scratch.pretraining.utils import (
    load_weights_into_gpt,
    text_to_token,
    token_ids_to_text,
)
import torch
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print(f"Settings: {settings}")
print(f"Parameter Dictionary: {params.keys()}")

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])

# WARN: OpenAI used bias vectors in the multi-head attention module's linear layers
# to implement the query , key and value matrix computations.
# Bias vectors are not commonly used in LLMs anymore as they don't improve the
# modeling performance of the model. However since we are working with pretrained
# weights, we need to match the settings for consistency and enable them here.
NEW_CONFIG.update({"qkv_bias": True})

# NOTE: By default, the `GPTModel` instance is initialized with random weights.
# The last step is now to override them with the weights from the pretrained model.
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt, params)
gpt.to("cpu")

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text(
    model=gpt,
    idx=text_to_token("Every effort moves you", tokenizer),
    max_new_tokens=10,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5,
)

print(f"Output text:\n {token_ids_to_text(token_ids, tokenizer)}")
