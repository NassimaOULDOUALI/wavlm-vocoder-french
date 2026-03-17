"""Tests for loss functions (CPU-only, no GPU required)."""
import pytest
import torch
from losses import MultiScaleSTFTLoss, CombinedLoss


@pytest.fixture
def batch():
    B, T = 2, 16000
    y = torch.randn(B, T).clamp(-1, 1)
    return y, y + torch.randn(B, T) * 0.05


def test_stft_loss_scalar(batch):
    y_true, y_pred = batch
    loss = MultiScaleSTFTLoss()(y_pred, y_true)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_stft_loss_zero_on_identity(batch):
    y, _ = batch
    loss = MultiScaleSTFTLoss()(y, y)
    assert loss.item() < 1e-4, "Loss should be ~0 when pred == target"


def test_combined_loss_dict(batch):
    y_true, y_pred = batch
    total, d = CombinedLoss(stft_loss_weight=1.0, l1_loss_weight=0.5)(y_pred, y_true)
    assert torch.isfinite(total)
    assert {"loss", "l1", "stft"} == set(d.keys())


def test_combined_loss_zero_l1_weight(batch):
    y_true, y_pred = batch
    _, d = CombinedLoss(stft_loss_weight=1.0, l1_loss_weight=0.0)(y_pred, y_true)
    assert abs(d["l1"]) < 1e-9


def test_combined_loss_zero_stft_weight(batch):
    y_true, y_pred = batch
    _, d = CombinedLoss(stft_loss_weight=0.0, l1_loss_weight=1.0)(y_pred, y_true)
    assert abs(d["stft"]) < 1e-9
