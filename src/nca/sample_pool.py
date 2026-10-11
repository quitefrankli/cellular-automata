import torch


class SamplePool:
    """Detached storage on the simulation device; sampled indices retain their order."""
    def __init__(self, seed: torch.Tensor, size: int) -> None:
        self.states = seed[:1].detach().repeat(size, 1, 1, 1)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(len(self.states), device=self.states.device)[:batch_size]
        return indices, self.states[indices].clone()

    def commit(self, indices: torch.Tensor, states: torch.Tensor) -> None:
        self.states[indices] = states.detach()
