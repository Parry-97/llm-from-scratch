import tiktoken
import numpy as np
import os
from llm_from_scratch.gpt_architecture.text_generation import generate_text_simple
from torch.utils.data import DataLoader
import torch
from torch import Tensor

from llm_from_scratch.gpt_architecture.dummy_gpt_model import GPTModel


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
    input_batch: Tensor, target_batch: Tensor, model: GPTModel, device: str
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    # INFO:  Previously, we applied the softmax function, selected the probability
    # scores corresponding to the target IDs, and computed the negative average log
    # probabilities.PyTorch’s cross_entropy function will take care of all these
    # steps for us. The idea is to basically maximiize the probability of the right
    # token ids by minimizing their negative average log probability

    # WARN: We are getting a single score across different batch sizes because
    # cross entropy averages the log probabilities in order to uniformly also consider
    # varying length sequences
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: GPTModel,
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


def train_model_simple(
    model: GPTModel,
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
    calculating the loss and new gradients, and updating weights and
    concluding with monitoring steps like printing losses and generating text samples
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
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()


def save_model_and_optimizer(
    model: GPTModel,
    optimizer: torch.optim.AdamW,  # pyright: ignore
    path: os.PathLike,
):
    """
    Saving a PyTorch model is relatively straightforward. The recommended
    way is to save a model’s state_dict, a dictionary mapping each layer to its parameters,
    using the torch.save function:

    If we plan to continue pre-training a model later—for example, using the
    train_model_simple function we defined earlier—saving the optimizer
    state is also recommended.
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_model_and_optimizer(path: os.PathLike):
    """
    Then we can restore the model and optimizer states by first loading the saved data via
    torch.load and then using the load_state_dict method:
    """
    checkpoint = torch.load(path)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)  # pyright: ignore
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape Mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params):
    """
    We carefully match the weights from
    OpenAI’s implementation with our GPTModel implementation.
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # INFO: The np.split function is used to divide the attention and
        # bias weights into three equal parts for the query, key and value
        # components
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].attention.W_query.weight = assign(
            gpt.trf_blocks[b].attention.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].attention.W_key.weight = assign(
            gpt.trf_blocks[b].attention.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].attention.W_value.weight = assign(
            gpt.trf_blocks[b].attention.W_value.weight, v_w.T
        )
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].attention.W_query.bias = assign(
            gpt.trf_blocks[b].attention.W_query.bias, q_b
        )
        gpt.trf_blocks[b].attention.W_key.bias = assign(
            gpt.trf_blocks[b].attention.W_key.bias, k_b
        )
        gpt.trf_blocks[b].attention.W_value.bias = assign(
            gpt.trf_blocks[b].attention.W_value.bias, v_b
        )
        gpt.trf_blocks[b].attention.out_proj.weight = assign(
            gpt.trf_blocks[b].attention.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].attention.out_proj.bias = assign(
            gpt.trf_blocks[b].attention.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )
        gpt.trf_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.trf_blocks[b].feed_forward.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.trf_blocks[b].feed_forward.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"],
        )
        gpt.trf_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.trf_blocks[b].feed_forward.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.trf_blocks[b].feed_forward.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # NOTE: The original GPT-2 model by OpenAI reused the token embedding weights
    # in the output layer to reduce the total number of parameters, which is a concept
    # known as weight tying
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


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
    model = GPTModel(GPT_CONFIG_124M)

    token_ids = generate_text_simple(
        model=model,
        context_size=GPT_CONFIG_124M["context_length"],
        max_new_tokens=10,
        idx=text_to_token(start_context, tokenizer),
    )

    print(f"Output text:\n {token_ids_to_text(token_ids, tokenizer)}")


if __name__ == "__main__":
    main()
