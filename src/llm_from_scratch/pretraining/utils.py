import tiktoken
import torch

from gpt_architecture.dummy_gpt_model import DummyGPTModel
from gpt_architecture.text_generation import generate_text


def text_to_token(text: str, tokenizer: tiktoken.Encoding):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(
        0
    )  # INFO: unsqueeze adds the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding):
    flat = token_ids.squeeze(0)  # NOTE: remove batch dimension
    return tokenizer.decode(flat.tolist())


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
