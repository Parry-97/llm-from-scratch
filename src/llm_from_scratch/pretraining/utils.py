import tiktoken
from llm_from_scratch.gpt_architecture.text_generation import generate_text
from torch.utils.data import DataLoader
import torch
from torch import Tensor

from llm_from_scratch.gpt_architecture.dummy_gpt_model import DummyGPTModel


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


def train_model_simple(
    model: DummyGPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter,
    start_context,
    tokenizer: tiktoken.Encoding,
):
    """
    A typical training loop for training deep neural networks in
    PyTorch consists of numerous steps starting with iterating
    over each epoch, processing batches, resetting gradients,
    calculating the loss and new gradients, and updating weights and concluding with monitoring steps like printing
    losses and generating text samples
    """
    train_losses, val_losses, track_tokens_seen = (
        [],
        [],
        [],
    )  # initialize lists to store losses and tokens seen

    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # NOTE: Resets loss gradients from previous batch iteration
            loss = calc_loss_batch(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )
            loss.backward()  # NOTE: Computes gradients
            optimizer.step()  # NOTE: Updates weights
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # NOTE: Sets the model to evaluation mode to disable dropout
    with torch.no_grad():  # NOTE: Disables gradient computation to reduce overhead
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()


if __name__ == "__main__":
    main()
