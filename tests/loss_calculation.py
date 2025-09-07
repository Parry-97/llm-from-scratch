import os
from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel
from llm_from_scratch.tokenizer.gpt_dataset import create_dataloader_v1
from llm_from_scratch.pretraining.utils import calc_loss_loader
import torch
from tiktoken import get_encoding


# Get the directory where this test file is located
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_PATH = os.path.join(TEST_DIR, "the-verdict.txt")

with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
    text_data = f.read()


torch.manual_seed(123)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


def main():
    total_characters = len(text_data)
    tokenizer = get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Total characters: {total_characters}")
    print(f"Total tokens: {total_tokens}")

    train_ratio = 0.90
    split_idx = int(total_characters * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTModel(GPT_CONFIG_124M).to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"Train loss: {train_loss}")
    print(f"Validation loss: {val_loss}")


if __name__ == "__main__":
    main()
