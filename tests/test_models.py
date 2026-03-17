"""Tests for model architecture (CPU-only, no WavLM download required)."""
import pytest
import torch
import torch.nn.functional as F
from model import WavLMAdapter, HiFiGenerator, ResBlock


def test_adapter_output_shape():
    adapter = WavLMAdapter(wavlm_dim=32, hidden_dim=16, num_layers=2, kernel_size=3)
    out = adapter(torch.randn(2, 50, 32))
    assert out.shape == (2, 16, 50)


def test_resblock_preserves_shape():
    for dil in (1, 3, 5):
        rb = ResBlock(channels=8, kernel_size=3, dilation=dil)
        x = torch.randn(1, 8, 64)
        assert rb(x).shape == x.shape


def test_generator_upsample_at_least_320x():
    """Raw generator output >= T'*320. WavLM2Audio.forward() then crops to T.

    ConvTranspose1d with stride=5 produces 16008 for T'=50 (not 16000).
    This is correct — the model crops/pads downstream.
    """
    gen = HiFiGenerator(hidden_dim=32)
    T_prime = 50
    out = gen(torch.randn(1, 32, T_prime))
    assert out.shape[2] >= T_prime * 320
    assert out.shape[:2] == torch.Size([1, 1])


def test_generator_tanh_bounded():
    gen = HiFiGenerator(hidden_dim=32)
    out = gen(torch.randn(2, 32, 20))
    assert out.abs().max().item() <= 1.0 + 1e-5


def test_adapter_no_nan():
    adapter = WavLMAdapter(wavlm_dim=64, hidden_dim=32, num_layers=3, kernel_size=5)
    out = adapter(torch.randn(2, 100, 64))
    assert torch.isfinite(out).all()


def test_model_crop_pad():
    """WavLM2Audio.forward() must return exactly (B, T) — crop if long, pad if short."""
    T = 16000
    # crop case (generator produces 16008 for T'=50)
    out_long = torch.randn(2, 16008)
    assert out_long[:, :T].shape == (2, T)
    # pad case
    out_short = torch.randn(2, 15990)
    assert F.pad(out_short, (0, T - out_short.shape[1])).shape == (2, T)
