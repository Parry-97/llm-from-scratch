# INFO : We will now implement a small neural network submodule used
# as part of the transformer blocks in LLMs
# We begin by implementing the GELU activation function

import torch
from torch import nn


class GELU(nn.Module):
    """Subclass of nn.Module implementing the GELU activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
