import torch
import pytest
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel
from llm_from_scratch.pretraining.utils import train_model_simple
from llm_from_scratch.tokenizer.gpt_dataset import create_dataloader_v1
from tiktoken import get_encoding

# Pytest fixture for the GPT model configuration
@pytest.fixture
def gpt_config():
    return {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

# Pytest fixture for the GPT model
@pytest.fixture
def gpt_model(gpt_config):
    return GPTModel(gpt_config)

# Pytest fixture for the tokenizer
@pytest.fixture
def tokenizer():
    return get_encoding("gpt2")

# Pytest fixture for the data loaders
@pytest.fixture
def data_loaders(gpt_config):
    text_data = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z" * 100  # Dummy data
    train_ratio = 0.90
    split_idx = int(len(text_data) * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader

# Test the simple training function
def test_train_model_simple(gpt_model, data_loaders, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt_model.to(device)
    # NOTE: AdamW is a variant of Adam that
    # improves the weight decay approach, which aims to minimize model complexity and
    # prevent overfitting by penalizing larger weights. This adjustment allows AdamW to
    # achieve more effective regularization and better generalization; thus, AdamW is fre-
    # quently used in the training of LLMs.
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=0.0004, weight_decay=0.1)
    train_loader, val_loader = data_loaders

    # Train for a small number of epochs and iterations for testing purposes
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=gpt_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,  # Reduced for faster testing
        eval_freq=1,
        eval_iter=1,
        start_context="A B C",
        tokenizer=tokenizer,
    )

    # Assertions to ensure the training ran without errors
    assert train_losses is not None
    assert val_losses is not None
    assert tokens_seen is not None
    assert len(train_losses) > 0
    assert len(val_losses) > 0
