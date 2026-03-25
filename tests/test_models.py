"""Unit tests for model architectures."""

import torch
import pytest


# ─── ResBlock ────────────────────────────────────────────────────────────────

def test_resblock_shape():
    from src.models.generator import ResBlock
    rb = ResBlock(64, 3, 1)
    x = torch.randn(2, 64, 100)
    out = rb(x)
    assert out.shape == x.shape


def test_resblock_residual_zero_input():
    from src.models.generator import ResBlock
    rb = ResBlock(64, 3, 1)
    x = torch.zeros(2, 64, 100)
    out = rb(x)
    assert out.shape == x.shape


def test_resblock_different_dilations():
    from src.models.generator import ResBlock
    for dilation in [1, 3, 5]:
        rb = ResBlock(32, 7, dilation)
        x = torch.randn(2, 32, 50)
        out = rb(x)
        assert out.shape == x.shape


# ─── HiFiGAN Generator ───────────────────────────────────────────────────────

def test_generator_output_shape():
    from src.models.generator import HiFiGANGenerator
    gen = HiFiGANGenerator(hidden_dim=256)
    x = torch.randn(2, 256, 50)
    out = gen(x)
    assert out.shape[0] == 2
    assert out.shape[1] == 1
    assert abs(out.shape[2] - 50 * 320) < 100


def test_generator_output_range():
    from src.models.generator import HiFiGANGenerator
    gen = HiFiGANGenerator(hidden_dim=256)
    x = torch.randn(2, 256, 50)
    out = gen(x)
    assert out.abs().max().item() <= 1.0 + 1e-5


def test_generator_batch_size_1():
    from src.models.generator import HiFiGANGenerator
    gen = HiFiGANGenerator(hidden_dim=256)
    x = torch.randn(1, 256, 50)
    out = gen(x)
    assert out.shape[0] == 1


def test_generator_eval_mode_deterministic():
    from src.models.generator import HiFiGANGenerator
    gen = HiFiGANGenerator(hidden_dim=256)
    gen.eval()
    x = torch.randn(2, 256, 50)
    with torch.no_grad():
        out1 = gen(x)
        out2 = gen(x)
    assert torch.allclose(out1, out2)


# ─── LayerFusion ─────────────────────────────────────────────────────────────

def test_layer_fusion_learned_shape():
    from src.models.adapter import LayerFusion
    lf = LayerFusion(num_layers=9, fusion_type="learned")
    hidden = tuple(torch.randn(2, 50, 768) for _ in range(13))
    out = lf(hidden)
    assert out.shape == (2, 50, 768)


def test_layer_fusion_average_shape():
    from src.models.adapter import LayerFusion
    lf = LayerFusion(num_layers=9, fusion_type="average")
    hidden = tuple(torch.randn(2, 50, 768) for _ in range(13))
    out = lf(hidden)
    assert out.shape == (2, 50, 768)


def test_layer_fusion_learned_weights_sum_to_one():
    from src.models.adapter import LayerFusion
    import torch.nn.functional as F
    lf = LayerFusion(num_layers=9, fusion_type="learned")
    weights = F.softmax(lf.weights, dim=0)
    assert abs(weights.sum().item() - 1.0) < 1e-5


def test_layer_fusion_invalid_type():
    from src.models.adapter import LayerFusion
    lf = LayerFusion(num_layers=9, fusion_type="invalid")
    hidden = tuple(torch.randn(2, 50, 768) for _ in range(13))
    with pytest.raises(ValueError):
        lf(hidden)


def test_layer_fusion_n1():
    from src.models.adapter import LayerFusion
    lf = LayerFusion(num_layers=1, fusion_type="learned")
    hidden = tuple(torch.randn(2, 50, 768) for _ in range(13))
    out = lf(hidden)
    assert out.shape == (2, 50, 768)


# ─── WavLMAdapter ────────────────────────────────────────────────────────────

def test_adapter_output_shape():
    from src.models.adapter import WavLMAdapter
    adapter = WavLMAdapter(wavlm_dim=768, hidden_dim=256)
    x = torch.randn(2, 50, 768)
    out = adapter(x)
    assert out.shape == (2, 256, 50)


def test_adapter_batch_size_1():
    from src.models.adapter import WavLMAdapter
    adapter = WavLMAdapter(wavlm_dim=768, hidden_dim=256)
    x = torch.randn(1, 50, 768)
    out = adapter(x)
    assert out.shape == (1, 256, 50)


def test_adapter_different_hidden_dims():
    from src.models.adapter import WavLMAdapter
    for hidden_dim in [128, 256, 512]:
        adapter = WavLMAdapter(wavlm_dim=768, hidden_dim=hidden_dim)
        x = torch.randn(2, 50, 768)
        out = adapter(x)
        assert out.shape == (2, hidden_dim, 50)


# ─── Discriminators ──────────────────────────────────────────────────────────

def test_mpd_output_count():
    from src.models.discriminator import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator()
    audio = torch.randn(2, 16000)
    outputs, features = mpd(audio)
    assert len(outputs) == 5
    assert len(features) == 5


def test_msd_output_count():
    from src.models.discriminator import MultiScaleDiscriminator
    msd = MultiScaleDiscriminator()
    audio = torch.randn(2, 16000)
    outputs, features = msd(audio)
    assert len(outputs) == 3
    assert len(features) == 3


def test_mpd_features_not_empty():
    from src.models.discriminator import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator()
    audio = torch.randn(2, 16000)
    _, features = mpd(audio)
    for feat_list in features:
        assert len(feat_list) > 0


def test_discriminators_different_inputs():
    from src.models.discriminator import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator()
    real = torch.randn(2, 16000)
    fake = torch.zeros(2, 16000)
    real_out, _ = mpd(real)
    fake_out, _ = mpd(fake)
    real_score = torch.cat([o.flatten() for o in real_out]).mean().item()
    fake_score = torch.cat([o.flatten() for o in fake_out]).mean().item()
    assert real_score != fake_score
