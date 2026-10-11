# Local verification — 2026-10-11

Environment: Python 3.12.12, PyTorch 2.10.0+cu128. Both `nvidia-smi` and `torch.cuda.is_available()` showed that no usable CUDA device was available. CUDA training remains unverified here; the test suite includes a real CUDA training smoke test and explicit CUDA requests produce an actionable error.

## Tests and package

Executed:

```bash
UV_CACHE_DIR=/tmp/nca-uv-cache MPLCONFIGDIR=/tmp/nca-mpl uv run --no-sync pytest -q
```

Result: **17 passed, 1 skipped** (CUDA). Tests exercise all three modes, BPTT, finite gradients, pool ordering/detachment, checkpoint equivalence, exact CPU resume trajectories, actual CLI training, simulation, regeneration, help, and CUDA diagnostics. The model has exactly 8,336 trainable parameters.

Normal dependency installation was attempted with `uv sync`. The default cache was read-only; setting `UV_CACHE_DIR` resolved that issue. Package index access then failed with DNS errors due to the restricted network. An offline attempt using a copy of the machine's existing cache could not resolve all dependencies because its older index cache is incompatible with the installed uv 0.12.10. **A clean `uv sync` and a generated `uv.lock` remain blocked and unverified. No fabricated lockfile is supplied.** Run `uv sync` on a machine with package-index access to resolve and generate the lockfile.

To validate software despite that restriction, a uv virtual environment with system-site-packages used the already-installed scientific dependencies. Cached Hatchling build prerequisites were installed offline, and the package was built and installed with:

```bash
UV_CACHE_DIR=/tmp/nca-uv-cache uv pip install --offline --no-deps --no-build-isolation .
```

`uv run --no-sync nca inspect` successfully executed the installed entry point. This workaround validates the package build and functionality, not a clean dependency installation.

## Actual training and visual evidence

Ran 300 growth iterations on CPU, seed 0, 24×24 grid, 12-pixel target footprint, batch 4, 24–32 rollout steps, learning rate 0.002, two CPU threads. Target: `assets/circle.png`. Training loss decreased from **0.04223425** to **0.00040823**. Output is in `runs/circle/` (ignored by version control): config, checkpoint, metrics, plot, processed target, growth GIF, and three intermediate animations.

`runs/circle/contact.png` was rendered and visually inspected. At steps 24–32, the initial single black cell has developed into a green disk resembling the target. At 64 steps the disk develops a ring, and by 128–300 steps it becomes unstable and deteriorates. This is evidence of target growth at the short training horizon, **not persistence or successful regeneration**. This reduced-grid, reduced-horizon trial is not a full reproduction of an 8,000-iteration experiment.

An additional ten-iteration regeneration-mode smoke run uses a 32-state pool, four-state batch, three damaged states, and the same small grid/horizon, saved in `runs/regenerate-smoke/`. It exercises damage, pool writes, snapshots, checkpoint storage, and regeneration animation export. It is too short to establish learned repair. The deliberate left-half erasure CLI also runs against the circle checkpoint, producing `runs/circle/damage.gif`.

Full-resolution CUDA training, full-length training in all three modes, and reliable persistence/regeneration remain unverified. They require a working GPU or a longer CPU run. The README documents commands and differences from the original notebook.

## Local viewer addition

Added `nca serve` using only the standard library, with a localhost gallery for GIF/PNG artifacts, configuration, metrics, and intermediate snapshots. Updated package build/install succeeded. The updated suite reports **19 passed, 2 skipped**: CUDA and the live HTTP integration test. This environment rejects socket creation with `PermissionError`; an actual server launch was attempted and failed for that reason. Socket-free tests exercise the actual HTTP handler's request parsing, HTML escaping, image streaming/MIME types, HEAD responses, refresh discovery, and traversal/symlink boundaries. A live-server test will execute on a normal machine. Browser rendering and live serving remain unverified here.
