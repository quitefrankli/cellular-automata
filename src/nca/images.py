from pathlib import Path
import numpy as np
import torch
from PIL import Image


def load_target(path: str | Path, grid_size: int, footprint: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGBA")
    image.thumbnail((footprint, footprint), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (grid_size, grid_size))
    canvas.paste(image, ((grid_size - image.width) // 2, (grid_size - image.height) // 2))
    rgba = np.asarray(canvas, dtype=np.float32).copy() / 255
    rgba[..., :3] *= rgba[..., 3:4]
    return torch.from_numpy(rgba).permute(2, 0, 1).unsqueeze(0)


def rgba_image(state: torch.Tensor) -> Image.Image:
    rgba = state.detach().cpu()[0, :4].permute(1, 2, 0).numpy().copy()
    alpha = np.clip(rgba[..., 3:4], 0, 1)
    rgba[..., :3] = np.divide(rgba[..., :3], alpha, out=np.zeros_like(rgba[..., :3]), where=alpha > 0)
    rgba[..., 3:4] = alpha
    return Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))


def display_image(state: torch.Tensor, scale: int = 4, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    rgba = state.detach().cpu()[0, :4].permute(1, 2, 0).numpy()
    rgb = rgba[..., :3] + (1 - np.clip(rgba[..., 3:4], 0, 1)) * (np.array(background) / 255)
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
