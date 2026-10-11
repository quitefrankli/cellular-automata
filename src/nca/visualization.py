from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .model import make_seed
from .damage import damage
from .images import display_image


@torch.no_grad()
def animate(model, size: int, path: Path, steps: int = 300, *, grow_steps: int | None = None,
            kind: str = "circle", radius: float | None = None, scale: int = 4,
            background: tuple[int, int, int] = (255, 255, 255), frame_every: int = 2) -> None:
    state = make_seed(size, device=next(model.parameters()).device)
    frames = [display_image(state, scale, background)]
    total = steps + (grow_steps or 0)
    for step in range(total):
        if grow_steps is not None and step == grow_steps:
            frames.append(display_image(state, scale, background))
            state = damage(state, kind, radius)
            frames.extend([display_image(state, scale, background)] * 10)
        state = model(state)
        if step % frame_every == 0 or step == total - 1:
            frames.append(display_image(state, scale, background))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0)


def plot_loss(metrics: list[tuple[int, float, int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x[0] for x in metrics], [x[1] for x in metrics])
    ax.set(xlabel="Iteration", ylabel="RGBA mean squared error", yscale="log")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
