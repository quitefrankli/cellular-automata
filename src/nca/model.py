import torch
from torch import nn
from torch.nn import functional as F
from .perception import Perception


def make_seed(size: int, batch_size: int = 1, device: str | torch.device = "cpu") -> torch.Tensor:
    state = torch.zeros(batch_size, 16, size, size, device=device)
    state[:, 3:, size // 2, size // 2] = 1
    return state


def alive_mask(state: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > 0.1


class NCA(nn.Module):
    def __init__(self, update_probability: float = 0.5) -> None:
        super().__init__()
        if not 0 <= update_probability <= 1:
            raise ValueError("update probability must be between 0 and 1")
        self.update_probability = update_probability
        self.perception = Perception()
        self.update = nn.Sequential(nn.Conv2d(48, 128, 1), nn.ReLU(), nn.Conv2d(128, 16, 1))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, state: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        before = alive_mask(state)
        delta = self.update(self.perception(state))
        mask = torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device, generator=generator) < self.update_probability
        result = state + delta * mask
        return result * (before & alive_mask(result))

    def rollout(self, state: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            state = self(state)
        return state
