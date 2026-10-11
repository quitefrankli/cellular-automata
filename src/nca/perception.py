import torch
from torch import nn
from torch.nn import functional as F


class Perception(nn.Module):
    """Fixed identity/Sobel features, interleaved per channel as in the notebook."""

    def __init__(self) -> None:
        super().__init__()
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1
        dx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        self.register_buffer("kernels", torch.stack([identity, dx, dx.T])[:, None].repeat(16, 1, 1, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.conv2d(state, self.kernels, padding=1, groups=16)
