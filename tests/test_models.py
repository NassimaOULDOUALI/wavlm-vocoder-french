"""Tests for model architecture (CPU-only, no WavLM download required)."""
import pytest
import torch
from model import WavLMAdapterImproved as WavLMAdapter, HiFiGeneratorImproved as HiFiGenerator, ResBlockImproved as ResBlock


def test_adapter_output_shape():
    adapter = WavLMAdapter(wavlm_dim=32, hidden_dim=16, num_layers=2, kernel_size=3)
    out = adapter(torch.randn(2, 50, 32))
    assert out.shape == (2, 16, 50)


def test_resblock_preserves_shape():
    for dil in (1, 3, 5):
        rb = ResBlock(channels=8, kernel_size=3, dilation=(dil,))
        x = torch.randn(1, 8, 64)
        assert rb(x).shape == x.shape


def test_generator_upsample_factor():
    """Generator must upsample T' by exactly 320×."""
    gen = HiFiGenerator(hidden_dim=32)
    T_prime = 50
    out = gen(torch.randn(1, 32, T_prime))
    assert out.shape == (1, 1, T_prime * 320)


def test_generator_tanh_bounded():
    gen = HiFiGenerator(hidden_dim=32)
    out = gen(torch.randn(2, 32, 20))
    assert out.abs().max().item() <= 1.0 + 1e-5


def test_adapter_no_nan():
    adapter = WavLMAdapter(wavlm_dim=64, hidden_dim=32, num_layers=3, kernel_size=5)
    out = adapter(torch.randn(2, 100, 64))
    assert torch.isfinite(out).all()
