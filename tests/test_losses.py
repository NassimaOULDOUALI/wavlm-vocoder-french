"""Unit tests for loss functions."""

import torch
import pytest


def test_l1_loss():
    from src.losses.reconstruction import L1Loss
    loss = L1Loss()
    pred = torch.randn(2, 16000)
    target = torch.randn(2, 16000)
    out = loss(pred, target)
    assert out.item() > 0
    assert out.shape == ()


def test_l1_loss_zero():
    from src.losses.reconstruction import L1Loss
    loss = L1Loss()
    x = torch.randn(2, 16000)
    out = loss(x, x)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_multiscale_stft_loss():
    from src.losses.reconstruction import MultiScaleSTFTLoss
    loss = MultiScaleSTFTLoss()
    pred = torch.randn(2, 16000)
    target = torch.randn(2, 16000)
    out = loss(pred, target)
    assert out.item() > 0
    assert torch.isfinite(out)


def test_generator_adversarial_loss():
    from src.losses.gan import GeneratorAdversarialLoss
    loss = GeneratorAdversarialLoss()
    fake_outs = [torch.randn(2, 10) for _ in range(3)]
    out = loss(fake_outs)
    assert torch.isfinite(out)


def test_discriminator_adversarial_loss():
    from src.losses.gan import DiscriminatorAdversarialLoss
    loss = DiscriminatorAdversarialLoss()
    real_outs = [torch.randn(2, 10) for _ in range(3)]
    fake_outs = [torch.randn(2, 10) for _ in range(3)]
    out = loss(real_outs, fake_outs)
    assert out.item() > 0
    assert torch.isfinite(out)


def test_feature_matching_loss():
    from src.losses.gan import FeatureMatchingLoss
    loss = FeatureMatchingLoss()
    real_feats = [[torch.randn(2, 32, 10), torch.randn(2, 64, 5)] for _ in range(3)]
    fake_feats = [[torch.randn(2, 32, 10), torch.randn(2, 64, 5)] for _ in range(3)]
    out = loss(real_feats, fake_feats)
    assert torch.isfinite(out)
