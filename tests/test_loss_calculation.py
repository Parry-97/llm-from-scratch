import os
import torch
import pytest
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel
from llm_from_scratch.tokenizer.gpt_dataset import create_dataloader_v1
from llm_from_scratch.pretraining.utils import calc_loss_loader


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


@pytest.fixture
def text_data(tmp_path):
    # Create a much longer text to ensure we have enough data for the context_length
    # We need at least context_length * batch_size characters for the dataloader
    alphabet = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
    text = alphabet * 100  # Repeat to get enough characters (52 * 100 = 5200 chars)
    file_path = tmp_path / "the-verdict.txt"
    file_path.write_text(text)
    return text


@pytest.fixture
def data_loaders(text_data, gpt_config):
    train_ratio = 0.80  # Reduced to ensure validation set has enough data
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


def test_calc_loss_loader(gpt_config, data_loaders):
    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTModel(gpt_config).to(device)
    train_loader, val_loader = data_loaders

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert train_loss > 0
    assert val_loss > 0