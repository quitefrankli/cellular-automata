from pathlib import Path
import torch
import pytest
from PIL import Image, ImageDraw
from click.testing import CliRunner
from nca.model import NCA, make_seed, alive_mask
from nca.perception import Perception
from nca.sample_pool import SamplePool
from nca.damage import damage
from nca.checkpoint import load_checkpoint
from nca.training import TrainConfig, train, prepare_batch, reconstruction_loss
from nca.cli import main
from nca.images import load_target


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def test_architecture_and_seed():
    model = NCA()
    assert sum(p.numel() for p in model.parameters()) == 8336
    state = make_seed(9, 2)
    assert state.shape == (2, 16, 9, 9)
    assert state.sum() == 26
    assert torch.all(state[:, 3:, 4, 4] == 1)
    assert torch.equal(model(state), state)
    assert alive_mask(state).sum() == 18
    assert not model.perception.kernels.requires_grad


def test_perception():
    p = Perception()
    state = torch.zeros(1, 16, 7, 7)
    state[:, 0] = torch.arange(7).expand(7, 7)
    result = p(state)
    assert result.shape == (1, 48, 7, 7)
    assert torch.equal(result[:, 0], state[:, 0])
    assert torch.allclose(result[0, 1, 1:-1, 1:-1], torch.ones(5, 5))
    assert torch.allclose(result[0, 2, 1:-1, 1:-1], torch.zeros(5, 5))
    assert torch.equal(p.kernels[1, 0], torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8)
    assert torch.equal(p.kernels[2, 0], p.kernels[1, 0].T)


def test_stochastic_shared_mask_and_determinism():
    model = NCA()
    with torch.no_grad():
        model.update[-1].bias.fill_(0.25)
    state = torch.ones(2, 16, 12, 12)
    a = model(state, generator=torch.Generator().manual_seed(12))
    b = model(state, generator=torch.Generator().manual_seed(12))
    assert torch.equal(a, b)
    delta = a - state
    assert torch.equal(delta[:, :1].expand_as(delta), delta)
    assert (delta == 0).any() and (delta == 0.25).any()
    model.update_probability = 0
    assert torch.equal(model(state), state)
    model.update_probability = 1
    assert torch.equal(model(state), state + 0.25)


def test_pre_and_post_life_masks():
    model = NCA(1)
    state = make_seed(9)
    state[:, 4, 0, 0] = 5
    with torch.no_grad():
        model.update[-1].bias[3] = 1
    result = model(state)
    assert result[:, :, 0, 0].sum() == 0  # newly mature outside pre-mask dies
    with torch.no_grad():
        model.update[-1].bias[3] = -2
    assert model(make_seed(9)).sum() == 0  # post-mask death


def test_bptt():
    model = NCA()
    loss = model.rollout(make_seed(8), 4)[:, :4].square().mean()
    loss.backward()
    assert model.update[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_pool_mapping_and_detach():
    pool = SamplePool(make_seed(5), 8)
    indices = torch.tensor([6, 1, 4])
    states = torch.stack([torch.full((16, 5, 5), float(i)) for i in [2, 3, 4]]).requires_grad_()
    pool.commit(indices, states)
    assert torch.equal(pool.states[indices], states)
    assert pool.states.grad_fn is None and not pool.states.requires_grad
    sampled_indices, sampled = pool.sample(4)
    assert len(sampled_indices.unique()) == 4
    assert torch.equal(sampled, pool.states[sampled_indices])


def test_replacement_and_damage_selection():
    target = torch.zeros(1, 4, 8, 8)
    states = torch.stack([torch.full((16, 8, 8), float(i)) for i in [4, 1, 3, 2]])
    config = TrainConfig(mode="regenerate", batch_size=4, damage_count=2, damage_radius=2)
    result = prepare_batch(states.clone(), target, make_seed(8), config)
    assert torch.equal(result[0], make_seed(8)[0])
    assert torch.equal(result[2], states[2])
    assert (result[1] == 0).any() and (result[3] == 0).any()
    assert torch.all(reconstruction_loss(torch.ones(2, 16, 8, 8), target) == 1)


@pytest.mark.parametrize("kind", ["circle", "rectangle", "left", "right"])
def test_damage(kind):
    state = torch.ones(2, 16, 16, 16)
    result = damage(state, kind, radius=4)
    assert (result == 0).any() and (result == 1).any()
    assert torch.equal(result[:, :1].expand_as(result), result)
    assert state.min() == 1


def make_target(path):
    image = Image.new("RGBA", (8, 8))
    ImageDraw.Draw(image).ellipse((1, 1, 6, 6), fill=(50, 200, 100, 255))
    image.save(path)


def test_alpha_processing(tmp_path):
    path = tmp_path / "target.png"
    Image.new("RGBA", (8, 4), (255, 0, 0, 128)).save(path)
    target = load_target(path, 16, 8)
    assert target.shape == (1, 4, 16, 16)
    assert target[0, 0, 8, 8] == target[0, 3, 8, 8]
    assert target[:, :, 0].sum() == 0


@pytest.mark.parametrize("mode", ["grow", "persist", "regenerate"])
def test_training_checkpoint_resume_cli(tmp_path, mode):
    path = tmp_path / "target.png"
    make_target(path)
    config = TrainConfig(target=str(path), mode=mode, grid_size=12, target_size=8, batch_size=4,
                         pool_size=8, iterations=2, min_steps=2, max_steps=3, checkpoint_interval=2, threads=1)
    output = tmp_path / "run"
    model = train(config, output, report=lambda _: None)
    saved = load_checkpoint(output / "checkpoint.pt")
    restored = NCA()
    restored.load_state_dict(saved["model"])
    seed = make_seed(12)
    torch.manual_seed(5)
    a = model(seed)
    torch.manual_seed(5)
    assert torch.equal(a, restored(seed))
    assert saved["iteration"] == 2
    for filename in ["config.json", "metrics.csv", "loss.png", "target.png", "growth.gif"]:
        assert (output / filename).exists()
    if mode == "regenerate":
        assert (output / "regeneration.gif").exists()
    config.iterations = 3
    train(config, output, output / "checkpoint.pt", report=lambda _: None)
    assert load_checkpoint(output / "checkpoint.pt")["iteration"] == 3
    # Compare uninterrupted training with resume, including pool and RNG restoration.
    train(config, tmp_path / "continuous", report=lambda _: None)
    continued = load_checkpoint(tmp_path / "continuous" / "checkpoint.pt")
    resumed = load_checkpoint(output / "checkpoint.pt")
    for key in continued["model"]:
        assert torch.equal(continued["model"][key], resumed["model"][key])
    runner = CliRunner()
    assert runner.invoke(main, ["inspect", "--checkpoint", str(output / "checkpoint.pt")]).exit_code == 0
    for command in ["simulate", "regenerate"]:
        args = [command, "--checkpoint", str(output / "checkpoint.pt"), "--output", str(tmp_path / f"{command}.gif")]
        args += ["--steps", "3"] if command == "simulate" else ["--grow-steps", "2", "--recovery-steps", "3"]
        result = runner.invoke(main, args)
        assert result.exit_code == 0, result.output


def test_cli_help_and_errors():
    runner = CliRunner()
    for command in ["train", "simulate", "regenerate", "inspect"]:
        assert runner.invoke(main, [command, "--help"]).exit_code == 0
    assert "--target is required" in runner.invoke(main, ["train", "--output", "unused"]).output


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_training(tmp_path):
    path = tmp_path / "target.png"
    make_target(path)
    model = train(TrainConfig(target=str(path), grid_size=12, target_size=8, batch_size=4, iterations=2,
                             min_steps=4, max_steps=6, device="cuda", threads=1), tmp_path / "gpu", report=lambda _: None)
    assert next(model.parameters()).is_cuda
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_cli_train_smoke(tmp_path):
    target = tmp_path / "target.png"
    make_target(target)
    result = CliRunner().invoke(main, ["train", "--target", str(target), "--output", str(tmp_path / "cli"),
                                      "--iterations", "1", "--grid-size", "12", "--target-size", "8",
                                      "--batch-size", "2", "--min-steps", "2", "--max-steps", "2", "--threads", "1"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli" / "checkpoint.pt").exists()
    if not torch.cuda.is_available():
        result = CliRunner().invoke(main, ["simulate", "--checkpoint", str(tmp_path / "cli" / "checkpoint.pt"),
                                          "--device", "cuda", "--output", str(tmp_path / "cuda.gif")])
        assert result.exit_code != 0
        assert "Check the NVIDIA driver" in result.output
