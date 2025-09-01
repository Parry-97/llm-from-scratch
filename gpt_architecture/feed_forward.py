import torch
from torch import nn


class FeedForward(nn.Module):
    """
    A simple feed-forward layer with GELU activation.
    The FeedForward module plays a crucial role in enhancing the model’s ability to learn
    from and generalize the data. Although the input and output dimensions of this
    module are the same, it internally expands the embedding dimension into a higher-
    dimensional space through the first linear layer, as illustrated in figure 4.10. This expan-
    sion is followed by a nonlinear GELU activation and then a contraction back to the orig-
    inal dimension with the second linear transformation. Such a design allows for the
    exploration of a richer representation space.
    """

    def __init__(self, cfg):
        super().__init__()
        # NOTE: the uniformity in input and output dimensions simplifies the architecture
        # by enabling the stacking of multiple layers, as we will do later, without the need to
        # adjust dimensions between them, thus making the model more scalable
        self.layers = nn.Sequential(
            # NOTE: The output  dimensions of this layer are obtained expanding the inputs
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(
                4 * cfg["emb_dim"],
                cfg["emb_dim"],
            ),
        )

    def forward(self, x):
        return self.layers(x)
