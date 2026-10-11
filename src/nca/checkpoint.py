from pathlib import Path
import random
import numpy as np
import torch


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path: Path, model, optimizer, config: dict, iteration: int, target: torch.Tensor, pool=None, metrics=None) -> None:
    payload = {"version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
               "iteration": iteration, "target": target.detach().cpu(), "pool": pool.states.detach().cpu() if pool else None,
               "rng": rng_state(), "metrics": metrics or []}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    # Local checkpoints include Python/NumPy RNG state: load only trusted files.
    return torch.load(path, map_location=device, weights_only=False)
