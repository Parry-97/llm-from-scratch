from torch import nn
import torch
from .transformer import TransformerBlock
from .layer_normalization import LayerNorm


class DummyGPTModel(nn.Module):
    """
    The DummyGPTModel class in this code defines a simplified version of a GPT-like
    model using PyTorch’s neural network module (nn.Module)
    """

    def __init__(self, cfg: dict):
        """
        The model architecture in the DummyGPTModel class consists of token and positional
        embeddings, dropout, a series of transformer blocks (DummyTransformerBlock), a final
        layer normalization (DummyLayerNorm), and a linear output layer (out_head).
        The configuration is passed in via a Python dictionary
        """
        super().__init__()
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # NOTE: Following the transformer blocks,
        # a LayerNorm layer is applied, standardizing the outputs from the transformer blocks to
        # stabilize the learning process.
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # NOTE:
        # The output from the final transformer block then goes through a final layer normal-
        # ization step before reaching the linear output layer. This layer maps the transformer’s
        # output to a high-dimensional space (in this case, 50,257 dimensions, corresponding to
        # the model’s vocabulary size) to predict the next token in the sequence.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        The forward method describes the data flow through the model: it computes token
        and positional embeddings for the input indices, applies dropout, processes the data
        through the transformer blocks, applies normalization, and finally produces logits
        with the linear output layer
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
