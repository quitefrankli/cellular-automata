from pathlib import Path
from functools import wraps
import json
import click
import torch
from .training import TrainConfig, train as run_train, select_device
from .checkpoint import load_checkpoint
from .model import NCA
from .visualization import animate


def errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            raise click.ClickException(str(exc)) from exc
    return wrapped


def background_value(ctx, param, value):
    try:
        rgb = tuple(int(v) for v in value.split(","))
        if len(rgb) != 3 or any(not 0 <= v <= 255 for v in rgb):
            raise ValueError
        return rgb
    except ValueError:
        raise click.BadParameter("use R,G,B integers from 0 to 255")


@click.group()
def main():
    """Train and visualize local Growing Neural Cellular Automata."""


@main.command()
@click.option("--target", type=click.Path(exists=True, dir_okay=False), help="Target PNG; required for a new run.")
@click.option("--mode", type=click.Choice(["grow", "persist", "regenerate"]), default="grow", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--resume", type=click.Path(exists=True, path_type=Path), help="Trusted checkpoint. Restores configuration; iterations is the desired total.")
@click.option("--grid-size", default=64, show_default=True)
@click.option("--target-size", default=40, show_default=True)
@click.option("--batch-size", type=int, help="Default: 8 for regenerate; 32 otherwise.")
@click.option("--learning-rate", default=0.002, show_default=True)
@click.option("--iterations", default=8000, show_default=True)
@click.option("--pool-size", default=1024, show_default=True)
@click.option("--update-probability", default=0.5, show_default=True)
@click.option("--min-steps", default=64, show_default=True)
@click.option("--max-steps", default=96, show_default=True)
@click.option("--checkpoint-interval", default=100, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
@click.option("--damage-radius", type=float, help="Circle radius in cells; default random 5–20% of grid size.")
@click.option("--damage-count", default=3, show_default=True)
@click.option("--damage-probability", default=1.0, show_default=True)
@click.option("--gradient-mode", type=click.Choice(["normalize", "clip", "none"]), default="normalize", show_default=True)
@click.option("--clip-norm", default=1.0, show_default=True)
@click.option("--scale", default=4, show_default=True)
@click.option("--background", default="255,255,255", callback=background_value)
@click.option("--threads", default=4, show_default=True, help="PyTorch CPU threads.")
@errors
def train(output, resume, **kwargs):
    """Train one target pattern; save metrics, checkpoints, and GIFs."""
    if not resume and not kwargs["target"]:
        raise click.ClickException("--target is required unless --resume is provided")
    kwargs["batch_size"] = kwargs["batch_size"] or (8 if kwargs["mode"] == "regenerate" else 32)
    run_train(TrainConfig(**kwargs), output, resume, click.echo)


def inference_options(function):
    for decorator in [click.option("--checkpoint", required=True, type=click.Path(exists=True, path_type=Path)),
                      click.option("--output", required=True, type=click.Path(path_type=Path)),
                      click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto"),
                      click.option("--scale", type=click.IntRange(min=1), default=4),
                      click.option("--seed", type=int, default=0),
                      click.option("--background", default="255,255,255", callback=background_value)]:
        function = decorator(function)
    return function


def read_model(checkpoint, device):
    saved = load_checkpoint(checkpoint, select_device(device))
    model = NCA(saved["config"]["update_probability"]).to(select_device(device))
    model.load_state_dict(saved["model"])
    model.eval()
    torch.set_num_threads(saved["config"].get("threads", 4))
    return model, saved


@main.command()
@inference_options
@click.option("--steps", type=click.IntRange(min=1), default=300)
@errors
def simulate(checkpoint, output, device, scale, seed, background, steps):
    """Export growth from a single seed using a trusted checkpoint."""
    model, saved = read_model(checkpoint, device)
    torch.manual_seed(seed)
    animate(model, saved["config"]["grid_size"], output, steps, scale=scale, background=background)
    click.echo(f"Saved {output}")


@main.command()
@inference_options
@click.option("--grow-steps", type=click.IntRange(min=1), default=100)
@click.option("--recovery-steps", type=click.IntRange(min=1), default=200)
@click.option("--damage", "kind", type=click.Choice(["circle", "rectangle", "left", "right"]), default="circle")
@click.option("--damage-radius", type=click.FloatRange(min=0, min_open=True))
@errors
def regenerate(checkpoint, output, device, scale, seed, background, grow_steps, recovery_steps, kind, damage_radius):
    """Grow, erase all channels in a region, and visualize recovery."""
    model, saved = read_model(checkpoint, device)
    torch.manual_seed(seed)
    animate(model, saved["config"]["grid_size"], output, recovery_steps, grow_steps=grow_steps,
            kind=kind, radius=damage_radius, scale=scale, background=background)
    click.echo(f"Saved {output}")


@main.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True, path_type=Path))
@errors
def inspect(checkpoint):
    """Show architecture, saved configuration, and local device compatibility."""
    model, saved = read_model(checkpoint, "cpu")
    click.echo(model)
    click.echo(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    click.echo(f"Checkpoint version: {saved['version']}; iteration: {saved['iteration']}; pool saved: {saved['pool'] is not None}")
    click.echo(json.dumps(saved["config"], indent=2))
    click.echo(f"CPU compatible: yes; CUDA available: {torch.cuda.is_available()}; PyTorch: {torch.__version__}")


@main.command()
@click.option("--runs-dir", type=click.Path(path_type=Path, file_okay=False), default="runs", show_default=True,
              help="Folder containing run outputs, or a single run folder.")
@click.option("--port", type=click.IntRange(1, 65535), default=8000, show_default=True)
@click.option("--open-browser", is_flag=True, help="Open the viewer in your default browser.")
@errors
def serve(runs_dir: Path, port: int, open_browser: bool) -> None:
    """Serve a local gallery of growth GIFs, damage demos, targets, and loss plots."""
    import webbrowser
    from .server import local_network_ip, make_server

    try:
        server = make_server(runs_dir, port)
    except PermissionError as exc:
        raise click.ClickException("This environment prohibits local sockets. Run nca serve from your local terminal.") from exc
    with server:
        url = f"http://127.0.0.1:{server.server_port}/"
        click.echo(f"Viewing {runs_dir.resolve()} at {url}")
        network_ip = local_network_ip()
        if network_ip:
            click.echo(f"Local network: http://{network_ip}:{server.server_port}/")
        else:
            click.echo("Local network: IP address could not be detected (listening on all IPv4 interfaces).")
        click.echo("GIFs play automatically. Press Ctrl+C to stop.")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            click.echo("\nViewer stopped.")
