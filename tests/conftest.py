import pytest
import tiktoken


@pytest.fixture(scope="session")
def gpt_config():
    return {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }


@pytest.fixture(scope="session")
def tokenizer():
    return tiktoken.get_encoding("gpt2")
