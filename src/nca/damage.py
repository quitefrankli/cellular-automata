import torch


def damage(state: torch.Tensor, kind: str = "circle", radius: float | None = None) -> torch.Tensor:
    """Erase all channels; circle/rectangle positions are independent per sample."""
    result = state.clone()
    b, _, h, w = state.shape
    if radius is not None and radius <= 0:
        raise ValueError("damage radius must be positive")
    yy, xx = torch.meshgrid(torch.arange(h, device=state.device), torch.arange(w, device=state.device), indexing="ij")
    if kind in ("left", "right"):
        erased = xx < w // 2 if kind == "left" else xx >= w // 2
        result.masked_fill_(erased[None, None], 0)
    elif kind in ("circle", "rectangle"):
        centers = torch.rand(b, 2, device=state.device) * 0.5 + 0.25
        cy = centers[:, 0, None, None] * (h - 1)
        cx = centers[:, 1, None, None] * (w - 1)
        r = radius if radius is not None else (torch.rand(b, 1, 1, device=state.device) * 0.15 + 0.05) * (min(h, w) - 1)
        if kind == "circle":
            erased = (xx - cx).square() + (yy - cy).square() < r ** 2
        else:
            erased = ((xx - cx).abs() < r) & ((yy - cy).abs() < r)
        result.masked_fill_(erased[:, None], 0)
    else:
        raise ValueError(f"unknown damage type: {kind}")
    return result
