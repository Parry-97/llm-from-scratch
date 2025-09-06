import torch

from llm_from_scratch.gpt_architecture.dummy_gpt_model import DummyGPTModel
from llm_from_scratch.pretraining.utils import train_model_simple
import os
from tiktoken import get_encoding

from llm_from_scratch.tokenizer.gpt_dataset import create_dataloader_v1

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

model = DummyGPTModel(GPT_CONFIG_124M)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device=device)

# NOTE: AdamW is a variant of Adam that
# improves the weight decay approach, which aims to minimize model complexity and
# prevent overfitting by penalizing larger weights. This adjustment allows AdamW to
# achieve more effective regularization and better generalization; thus, AdamW is fre-
# quently used in the training of LLMs.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10

train_ratio = 0.90
split_idx = int(len(text_data) * train_ratio)
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

tokenizer = get_encoding("gpt2")
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)
