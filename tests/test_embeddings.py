
import torch
import pytest
from llm_from_scratch.tokenizer.gpt_dataset import create_dataloader_v1


@pytest.fixture
def gpt_config():
    """Configuration for the GPT model with realistic embedding sizes.
    Using 256-dimensional vector representation, which is smaller than GPT-3's
    12,288 dimensions but still reasonable for experimentation.
    """
    return {
        "vocab_size": 50257,
        "context_length": 4,
        "emb_dim": 256
    }


@pytest.fixture
def raw_text(tmp_path):
    text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
    file_path = tmp_path / "the-verdict.txt"
    file_path.write_text(text)
    return text


@pytest.fixture
def data_loader(raw_text, gpt_config):
    return create_dataloader_v1(
        raw_text, batch_size=8, max_length=gpt_config["context_length"], stride=gpt_config["context_length"], shuffle=False
    )


def test_embedding_layers(data_loader, gpt_config):
    """Test that embedding layers work correctly.
    
    The weight matrix of the embedding layer contains small random values.
    These values are optimized during LLM training as part of the LLM optimization itself.
    """
    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)

    # Create token embedding layer - embeds each token into a 256-dimensional vector
    token_embedding_layer = torch.nn.Embedding(gpt_config["vocab_size"], gpt_config["emb_dim"])
    token_embeddings = token_embedding_layer(inputs)

    assert token_embeddings.shape == (8, gpt_config["context_length"], gpt_config["emb_dim"])

    # For GPT model's absolute embedding approach, create another embedding layer
    # with the same embedding dimension as the token_embedding_layer
    pos_embedding_layer = torch.nn.Embedding(gpt_config["context_length"], gpt_config["emb_dim"])
    pos_embeddings = pos_embedding_layer(torch.arange(gpt_config["context_length"]))

    assert pos_embeddings.shape == (gpt_config["context_length"], gpt_config["emb_dim"])

    # Add positional embeddings directly to token embeddings
    # PyTorch will add the 4 × 256–dimensional pos_embeddings tensor to each
    # 4 × 256–dimensional token embedding tensor in each of the eight batches
    input_embeddings = token_embeddings + pos_embeddings

    assert input_embeddings.shape == (8, gpt_config["context_length"], gpt_config["emb_dim"])
