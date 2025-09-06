import tiktoken
from torch.utils.data import DataLoader
import torch
from torch import Tensor

from llm_from_scratch.gpt_architecture.dummy_gpt_model import DummyGPTModel
from llm_from_scratch.gpt_architecture.text_generation import generate_text


def text_to_token(text: str, tokenizer: tiktoken.Encoding):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(
        0
    )  # INFO: unsqueeze adds the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding):
    flat = token_ids.squeeze(0)  # NOTE: remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(
    input_batch: Tensor, target_batch: Tensor, model: DummyGPTModel, device: str
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    # INFO:  Previously, we applied the softmax function, selected the probability
    # scores corresponding to the target IDs, and computed the negative average log
    # probabilities.PyTorch’s cross_entropy function will take care of all these
    # steps for us

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: DummyGPTModel,
    device: str,
    num_batches: int | None = None,
):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(
            data_loader
        )  # INFO: iterates over all batches if not specified
    else:
        num_batches = min(num_batches, len(data_loader))

    # INFO: We can specify a smaller number of batches via
    # num_batches to speed up the evaluation during model
    # training.
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # NOTE: sums loss for each batch
        else:
            break
    return total_loss / num_batches  # NOTE: Aveerages over all batches


def main():
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    token_ids = generate_text(
        model=model,
        context_size=GPT_CONFIG_124M["context_length"],
        max_new_tokens=10,
        idx=text_to_token(start_context, tokenizer),
    )
    print(f"Output text:\n {token_ids_to_text(token_ids, tokenizer)}")


if __name__ == "__main__":
    main()
