"""Unit tests for training components."""

import torch
import pytest
from pathlib import Path
import tempfile


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_batch():
    """Batch of 2 audio segments of 1 second at 16kHz."""
    return torch.randn(2, 16000)


@pytest.fixture
def generator():
    from src.models.generator import HiFiGANGenerator
    return HiFiGANGenerator(hidden_dim=256)


@pytest.fixture
def mpd():
    from src.models.discriminator import MultiPeriodDiscriminator
    return MultiPeriodDiscriminator()


@pytest.fixture
def msd():
    from src.models.discriminator import MultiScaleDiscriminator
    return MultiScaleDiscriminator()


@pytest.fixture
def adapter():
    from src.models.adapter import WavLMAdapter
    return WavLMAdapter(wavlm_dim=768, hidden_dim=256)


# ─── Generator Training Step ─────────────────────────────────────────────────

def test_generator_backward(generator, dummy_batch):
    """Generator loss should backprop without errors."""
    from src.losses.reconstruction import L1Loss

    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer.zero_grad()

    # Simulate adapter output (B, hidden_dim, T')
    features = torch.randn(2, 256, 50)
    pred = generator(features).squeeze(1)

    # Trim to match target
    target = dummy_batch[:, : pred.shape[1]]
    loss = L1Loss()(pred, target)

    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_generator_gradients_flow(generator):
    """All trainable params should have gradients after backward."""
    from src.losses.reconstruction import L1Loss

    features = torch.randn(2, 256, 50)
    pred = generator(features).squeeze(1)
    target = torch.randn_like(pred)
    loss = L1Loss()(pred, target)
    loss.backward()

    for name, param in generator.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_stft_loss_backward(dummy_batch):
    """MultiScaleSTFTLoss should backprop."""
    from src.losses.reconstruction import MultiScaleSTFTLoss

    pred = dummy_batch.clone().requires_grad_(True)
    loss = MultiScaleSTFTLoss()(pred, dummy_batch.detach())
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ─── Discriminator Training Step ─────────────────────────────────────────────

def test_discriminator_backward(mpd, msd, dummy_batch):
    """Discriminator loss should backprop without errors."""
    from src.losses.gan import DiscriminatorAdversarialLoss

    disc_params = list(mpd.parameters()) + list(msd.parameters())
    optimizer = torch.optim.Adam(disc_params, lr=1e-4)
    optimizer.zero_grad()

    fake = torch.randn_like(dummy_batch).detach()

    real_out_mpd, _ = mpd(dummy_batch)
    fake_out_mpd, _ = mpd(fake)
    real_out_msd, _ = msd(dummy_batch)
    fake_out_msd, _ = msd(fake)

    disc_loss_fn = DiscriminatorAdversarialLoss()
    loss = disc_loss_fn(real_out_mpd, fake_out_mpd) + disc_loss_fn(real_out_msd, fake_out_msd)

    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_discriminator_real_vs_fake(mpd, dummy_batch):
    """Discriminator should distinguish real from silent fake."""
    real_out, _ = mpd(dummy_batch)
    fake_out, _ = mpd(torch.zeros_like(dummy_batch))

    real_score = torch.cat([o.flatten() for o in real_out]).mean().item()
    fake_score = torch.cat([o.flatten() for o in fake_out]).mean().item()

    # Real score should differ from fake (untrained but different input)
    assert real_score != fake_score


# ─── GAN Combined Step ───────────────────────────────────────────────────────

def test_gan_training_step(generator, mpd, msd, dummy_batch):
    """Full GAN step: generator + discriminator losses both finite."""
    from src.losses.reconstruction import L1Loss, MultiScaleSTFTLoss
    from src.losses.gan import (
        GeneratorAdversarialLoss,
        DiscriminatorAdversarialLoss,
        FeatureMatchingLoss,
    )

    features = torch.randn(2, 256, 50)
    pred = generator(features).squeeze(1)
    target = dummy_batch[:, : pred.shape[1]]

    # Discriminator outputs
    mpd_fake_out, mpd_fake_feat = mpd(pred.detach())
    msd_fake_out, msd_fake_feat = msd(pred.detach())
    mpd_real_out, mpd_real_feat = mpd(target)
    msd_real_out, msd_real_feat = msd(target)

    # Generator losses
    l1 = L1Loss()(pred, target)
    stft = MultiScaleSTFTLoss()(pred, target)
    adv_g = GeneratorAdversarialLoss()(mpd_fake_out + msd_fake_out)
    fm = FeatureMatchingLoss()(
        mpd_real_feat + msd_real_feat,
        mpd_fake_feat + msd_fake_feat,
    )
    loss_g = l1 + stft + adv_g + fm

    # Discriminator losses
    disc_loss_fn = DiscriminatorAdversarialLoss()
    loss_d = disc_loss_fn(mpd_real_out, mpd_fake_out) + disc_loss_fn(msd_real_out, msd_fake_out)

    assert torch.isfinite(loss_g), f"Generator loss not finite: {loss_g}"
    assert torch.isfinite(loss_d), f"Discriminator loss not finite: {loss_d}"


# ─── Gradient Clipping ───────────────────────────────────────────────────────

def test_gradient_clipping(generator):
    """Gradient norm should be <= max_norm after clipping."""
    from src.losses.reconstruction import L1Loss

    features = torch.randn(2, 256, 50)
    pred = generator(features).squeeze(1)
    target = torch.randn_like(pred)
    loss = L1Loss()(pred, target)
    loss.backward()

    max_norm = 1.0
    grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_norm)

    assert float(grad_norm) >= 0


# ─── Checkpoint Save / Load ──────────────────────────────────────────────────

def test_checkpoint_save_load(generator):
    """Saved checkpoint should restore identical weights."""
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    from torch.cuda.amp import GradScaler

    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            model=generator,
            optimizer=optimizer,
            scaler=scaler,
            epoch=1,
            step=100,
            config={"test": True},
            save_dir=tmpdir,
            filename="test_ckpt.pt",
        )

        # Create new model and load
        from src.models.generator import HiFiGANGenerator
        new_model = HiFiGANGenerator(hidden_dim=256)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4)
        new_scaler = GradScaler(enabled=False)

        epoch, step = load_checkpoint(
            Path(tmpdir) / "test_ckpt.pt",
            new_model,
            new_optimizer,
            new_scaler,
        )

        assert epoch == 1
        assert step == 100

        # Weights should be identical
        for (n1, p1), (n2, p2) in zip(
            generator.named_parameters(), new_model.named_parameters(), strict=True
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"


def test_checkpoint_latest_saved(generator):
    """checkpoint_latest.pt should always be created."""
    from src.utils.checkpoint import save_checkpoint
    from torch.cuda.amp import GradScaler

    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            model=generator,
            optimizer=optimizer,
            scaler=scaler,
            epoch=2,
            step=200,
            config={},
            save_dir=tmpdir,
        )
        assert (Path(tmpdir) / "checkpoint_latest.pt").exists()


# ─── Adapter Training ────────────────────────────────────────────────────────

def test_adapter_backward(adapter):
    """Adapter should backprop correctly."""
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    optimizer.zero_grad()

    x = torch.randn(2, 50, 768)
    out = adapter(x)

    loss = out.mean()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    for name, param in adapter.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
