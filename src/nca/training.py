from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random
import numpy as np
import torch
from .model import NCA, make_seed
from .sample_pool import SamplePool
from .damage import damage
from .images import load_target, rgba_image
from .checkpoint import load_checkpoint, save_checkpoint, restore_rng, rng_state
from .visualization import animate, plot_loss


@dataclass
class TrainConfig:
    target: str = ""
    mode: str = "grow"
    grid_size: int = 64
    target_size: int = 40
    batch_size: int = 32
    learning_rate: float = 0.002
    iterations: int = 8000
    pool_size: int = 1024
    update_probability: float = 0.5
    min_steps: int = 64
    max_steps: int = 96
    checkpoint_interval: int = 100
    seed: int = 0
    device: str = "auto"
    damage_radius: float | None = None
    damage_count: int = 3
    damage_probability: float = 1.0
    gradient_mode: str = "normalize"
    clip_norm: float = 1.0
    scale: int = 4
    background: tuple[int, int, int] = (255, 255, 255)
    threads: int = 4

    def validate(self) -> None:
        for name in ("grid_size", "target_size", "batch_size", "iterations", "pool_size", "min_steps", "max_steps", "checkpoint_interval", "scale", "threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.target_size > self.grid_size or self.min_steps > self.max_steps:
            raise ValueError("target size must fit the grid and min steps must not exceed max steps")
        if self.mode not in ("grow", "persist", "regenerate"):
            raise ValueError("invalid training mode")
        if self.mode != "grow" and self.batch_size > self.pool_size:
            raise ValueError("batch size must not exceed pool size")
        if self.damage_count < 0 or (self.mode == "regenerate" and self.damage_count >= self.batch_size):
            raise ValueError("damage count must be nonnegative and smaller than batch size")
        if not 0 <= self.update_probability <= 1 or not 0 <= self.damage_probability <= 1:
            raise ValueError("probabilities must be between 0 and 1")
        if self.learning_rate <= 0 or self.clip_norm <= 0 or (self.damage_radius is not None and self.damage_radius <= 0):
            raise ValueError("learning rate, clip norm, and damage radius must be positive")


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but PyTorch cannot access a CUDA device. Check the NVIDIA driver and PyTorch installation, or use --device cpu.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def reconstruction_loss(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (state[:, :4] - target).square().mean(dim=(1, 2, 3))


def prepare_batch(states: torch.Tensor, target: torch.Tensor, seed: torch.Tensor,
                  config: TrainConfig) -> torch.Tensor:
    # Sort positions, not states: preserve original pool index mapping on commit.
    order = reconstruction_loss(states, target).argsort(descending=True)
    states[order[0]] = seed[0]
    if config.mode == "regenerate" and config.damage_count:
        selected = order[-config.damage_count:]
        selected = selected[torch.rand(len(selected), device=states.device) < config.damage_probability]
        states[selected] = damage(states[selected], radius=config.damage_radius)
    return states


def train(config: TrainConfig, output: Path, resume: Path | None = None, report=print) -> NCA:
    saved = load_checkpoint(resume) if resume else None
    if saved:
        # Resume the saved experiment; CLI iterations is the total desired iteration count.
        config = TrainConfig(**(saved["config"] | {"iterations": config.iterations, "device": config.device, "threads": config.threads}))
    config.validate()
    device = select_device(config.device)
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = NCA(config.update_probability).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    target = saved["target"].to(device) if saved else load_target(config.target, config.grid_size, config.target_size).to(device)
    seed = make_seed(config.grid_size, device=device)
    pool = SamplePool(seed, config.pool_size) if config.mode != "grow" else None
    metrics = []
    start = 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if pool:
            pool.states = saved["pool"].to(device).detach()
        metrics = saved["metrics"]
        start = saved["iteration"]
        restore_rng(saved["rng"])
    if start > config.iterations:
        raise ValueError("iterations is a total count and cannot precede the checkpoint iteration")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    rgba_image(target).save(output / "target.png")
    report(f"Training {config.mode} on {device}; {sum(p.numel() for p in model.parameters())} parameters; iterations {start}–{config.iterations}")
    def export(iteration: int, snapshot: bool = False) -> None:
        save_checkpoint(output / "checkpoint.pt", model, optimizer, asdict(config), iteration, target, pool, metrics)
        with (output / "metrics.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss", "steps"])
            writer.writerows(metrics)
        plot_loss(metrics, output / "loss.png")
        if snapshot:
            # Visualization must not perturb training's stochastic trajectory.
            rng = rng_state()
            animate(model, config.grid_size, output / "snapshots" / f"growth_{iteration:06d}.gif", steps=config.max_steps, scale=config.scale, background=config.background)
            restore_rng(rng)
    for iteration in range(start + 1, config.iterations + 1):
        if pool:
            indices, states = pool.sample(config.batch_size)
            states = prepare_batch(states, target, seed, config)
        else:
            states = seed.repeat(config.batch_size, 1, 1, 1)
        steps = random.randint(config.min_steps, config.max_steps)
        optimizer.zero_grad(set_to_none=True)
        states = model.rollout(states, steps)
        loss = reconstruction_loss(states, target).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at iteration {iteration}; try a smaller learning rate")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"nonfinite gradient at iteration {iteration}")
                if config.gradient_mode == "normalize":
                    parameter.grad.div_(parameter.grad.norm() + 1e-8)
        if config.gradient_mode == "clip":
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
        optimizer.step()
        if pool:
            pool.commit(indices, states)
        metrics.append((iteration, float(loss.detach()), steps))
        if iteration % config.checkpoint_interval == 0:
            report(f"iteration {iteration}: loss={metrics[-1][1]:.6f}")
            export(iteration, snapshot=True)
    export(config.iterations)
    rng = rng_state()
    animate(model, config.grid_size, output / "growth.gif", scale=config.scale, background=config.background)
    if config.mode == "regenerate":
        animate(model, config.grid_size, output / "regeneration.gif", grow_steps=100, steps=200,
                radius=config.damage_radius, scale=config.scale, background=config.background)
    restore_rng(rng)
    return model
