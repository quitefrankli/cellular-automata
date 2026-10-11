# Growing Neural Cellular Automata

A local PyTorch reproduction of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) (Mordvintsev et al., 2020), checked against the authors' [TensorFlow notebook](https://github.com/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb). A shared neural rule lets cells grow an image from one seed. Pool training encourages persistence; training with erased regions encourages repair. **Each model learns one specific target**, not arbitrary images.

Each grid is float32 with shape `(batch, 16, height, width)`. Channels 0–2 contain premultiplied RGB, channel 3 alpha, and channels 4–15 learned hidden state. The seed has channels 3–15 set to one at the center and zeros elsewhere.

Fixed depthwise 3×3 identity, Sobel X, and Sobel Y filters produce three features per channel: **48 inputs**, rather than 144 raw neighborhood values. Sobel filters are divided by eight. Features are interleaved by state channel, matching the notebook. A shared `48 → 128 → 16` MLP uses biased 1×1 convolutions and ReLU, totaling **8,336 trainable parameters**. Its final layer starts at zero. Each recurrent step adds the predicted change with independent probability 0.5 per cell, sharing the random mask across channels. A 3×3 alpha maximum exceeding 0.1 defines life. All channels are multiplied by the intersection of pre-update and post-update life masks. No internal states are clamped.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
uv run pytest
uv run nca --help
```

Python 3.12+ is required. Dependencies are PyTorch, Click, Pillow, NumPy, and Matplotlib; pytest is a development dependency. No notebooks, remote APIs, or hosted training services are used. Dependency installation needs package access or a populated local cache. If the default cache is read-only, prefix commands with `UV_CACHE_DIR=/tmp/nca-uv-cache`. In this restricted workspace, validation uses that prefix and `MPLCONFIGDIR=/tmp/nca-mpl`.

## Train

Supply a local PNG. It is converted to RGBA, resized without changing aspect ratio (downscaling only), premultiplied, and centered on a transparent canvas. The default footprint is at most 40×40 in a 64×64 grid. The provided `assets/circle.png` is a synthetic example; supply your own lizard PNG for the requested lizard experiment.

```bash
uv run nca train --target assets/circle.png --mode grow --iterations 8000 --output runs/grow
uv run nca train --target assets/circle.png --mode persist --iterations 8000 --output runs/persist
uv run nca train --target assets/circle.png --mode regenerate --iterations 8000 --output runs/regenerate
# Your own target, explicitly on CUDA:
uv run nca train --target assets/lizard.png --mode regenerate --device cuda --iterations 8000 --output runs/lizard
```

`grow` starts each batch from seeds. `persist` samples a pool, replaces the worst state with a seed, and saves detached rollout states back to the same pool indices. `regenerate` additionally erases circles in the three best sampled states. Defaults are 32 states per growth/persistence batch, 8 per regeneration batch, and a pool of 1,024. Loss is mean squared error on the four RGBA channels only, after a uniformly sampled rollout of 64–96 steps. Adam starts at 0.002. Each parameter tensor's gradient is divided by its L2 norm plus `1e-8`; compare `--gradient-mode clip --clip-norm 1` or `--gradient-mode none`.

Default circular damage uses centers in the middle half of the grid and independent radii of approximately 5–20% of its size, as in the notebook. `--damage-radius` overrides the radius in cells, `--damage-count` controls how many best states are damaged, and `--damage-probability` controls the independent probability for each selected state. The count must leave room for the freshly seeded state.

Use `uv run nca train --help` for all flags, including grid/target size, pool/batch size, rollout lengths, random seed, learning rate, and checkpoint interval. GPU memory scales with batch size, grid area, and rollout length because training differentiates through every step. Reduce batch size first; the pool itself holds about 256 MiB at the default size. Float32 is used throughout.

A small CPU example:

```bash
uv run nca train --target assets/circle.png --mode grow --device cpu \
  --grid-size 24 --target-size 12 --batch-size 4 --min-steps 24 --max-steps 32 \
  --iterations 300 --checkpoint-interval 100 --threads 2 --output runs/circle
```

Every run writes `config.json`, `checkpoint.pt`, `metrics.csv`, `loss.png`, `target.png`, and `growth.gif`. Regeneration runs also write `regeneration.gif`. Checkpoint intervals export intermediate growth animations to `snapshots/`. GIFs composite against white by default; `--background 20,20,20` changes the background. `--scale 4` magnifies cells with nearest-neighbor sampling. `target.png` preserves straight-alpha PNG transparency; tensors use premultiplied RGB. Display conversion alone clamps values. Animation export preserves training RNG state.

## Simulate and damage

```bash
uv run nca simulate --checkpoint runs/circle/checkpoint.pt --steps 300 --output runs/circle/growth.gif
uv run nca regenerate --checkpoint runs/regenerate/checkpoint.pt \
  --grow-steps 100 --recovery-steps 200 --damage circle --output runs/regenerate/recovery.gif
uv run nca regenerate --checkpoint runs/regenerate/checkpoint.pt \
  --damage left --output runs/regenerate/left.gif
uv run nca inspect --checkpoint runs/regenerate/checkpoint.pt
```

Damage choices are `circle`, `rectangle`, `left`, and `right`. Every erasure clears all 16 channels. Inference uses `torch.no_grad()` and runs completely on the selected device. A GIF demonstrates behavior; producing one alone does not establish successful regeneration.

## Resume and devices

```bash
uv run nca train --resume runs/regenerate/checkpoint.pt --iterations 12000 --output runs/regenerate
uv run nca train --resume runs/regenerate/checkpoint.pt --iterations 12000 --device cpu --output runs/resumed
```

Iterations means the **total** desired count. Resume restores the saved experiment configuration, processed target, model, Adam state, pool, metrics, and Python/NumPy/PyTorch RNG states. Only total iterations, device, and CPU thread count come from resume CLI flags. CUDA RNG states are restored when the local device count matches. Checkpoints are mapped onto the destination device and can move between CPU and CUDA. Only load trusted checkpoints: Python/NumPy RNG serialization requires PyTorch's full pickle loader.

`--device auto` selects CUDA when `torch.cuda.is_available()` is true, otherwise CPU. An explicit `--device cuda` fails with an actionable driver/install diagnostic if unavailable. `inspect` reports compatibility and PyTorch version. Seeds reproduce runs on the same software/device configuration; exact results across CPU/CUDA or different library versions are not guaranteed.

## Fidelity and limits

The architecture, stochastic updates, life-mask ordering, premultiplied target, loss, gradient normalization, seed, and three experiment procedures follow the original implementation. Engineering differences are configurable 64×64 grids (the notebook used target padding yielding 72×72), inclusive 64–96 rollout sampling, PyTorch initialization/Adam numerical behavior, larger default persistence batches, and a constant configurable learning rate rather than the notebook's scheduled decay. Pool indices retain their original mapping while selection uses loss ranks. Rotation experiments, browser deployment, and pretrained emoji models are outside scope.

Long training is normally needed. Growth-only models can decay or explode after the supervised rollout interval. Persistence and damage training improve robustness but do not guarantee repair after every injury; total erasure leaves no seed from which to regrow. A low loss at one horizon does not establish long-term stability. Use extended simulations and multiple stochastic seeds to assess a trained model.

Tests cover the exact parameter count, perception, seed, stochastic and life masks, BPTT gradients, pool mapping, damage, image alpha, checkpoint equality, exact CPU resume trajectory, all training modes, and CLI inference/help/errors. The GPU smoke test is skipped when CUDA is absent:

```bash
uv run pytest -m gpu
```

See `VERIFICATION.md` for results obtained in this workspace.

## Local web viewer

Browse saved animations, processed targets, loss plots, and intermediate snapshots in a small local gallery:

```bash
uv run nca serve --open-browser
# View a specific run or choose a different port:
uv run nca serve --runs-dir runs/circle --port 8080
```

Open `http://127.0.0.1:8000/` (or the printed port). GIFs play automatically; click any image for its full-size view. Click **Refresh outputs** to discover outputs saved during training. Configuration and loss CSV links accompany each run. Stop with Ctrl+C. The viewer binds to localhost, uses Python's standard-library HTTP server, and adds no dependencies. It displays saved outputs; generate new animations using `simulate` and `regenerate`.

For the existing restricted-workspace installation, use `uv run --no-sync nca serve` after installing the updated package.
