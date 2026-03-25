"""Tests for AudioDataset (CPU-only, no GPU required)."""
import pytest
import torch
import torchaudio
from dataset import AudioDataset, ValidationDataset, collate_fn


@pytest.fixture
def audio_dir(tmp_path):
    """Temp dir with 4 synthetic 3-second wav files."""
    sr = 16000
    for i in range(4):
        wav = torch.randn(1, sr * 3) * 0.1
        torchaudio.save(str(tmp_path / f"audio_{i}.wav"), wav, sr)
    return str(tmp_path)


def test_dataset_loads(audio_dir):
    ds = AudioDataset(audio_dir, segment_length=16000, sample_rate=16000)
    assert len(ds) == 4


def test_dataset_shape(audio_dir):
    ds = AudioDataset(audio_dir, segment_length=16000, sample_rate=16000)
    sample = ds[0]
    assert sample.shape == (16000,)
    assert sample.dtype == torch.float32


def test_dataset_normalised(audio_dir):
    ds = AudioDataset(audio_dir, segment_length=16000, sample_rate=16000, peak_target=0.99)
    sample = ds[0]
    assert sample.abs().max().item() <= 1.0 + 1e-5


def test_dataset_no_nan(audio_dir):
    ds = AudioDataset(audio_dir, segment_length=16000, sample_rate=16000)
    for i in range(len(ds)):
        assert torch.isfinite(ds[i]).all()


def test_validation_deterministic(audio_dir):
    ds = ValidationDataset(audio_dir, segment_length=16000, sample_rate=16000)
    assert torch.allclose(ds[0], ds[0]), "ValidationDataset must be deterministic"


def test_collate_fn(audio_dir):
    ds = AudioDataset(audio_dir, segment_length=16000, sample_rate=16000)
    batch = collate_fn([ds[i] for i in range(3)])
    assert batch.shape == (3, 16000)
    assert torch.isfinite(batch).all()


def test_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No audio files"):
        AudioDataset(str(tmp_path))
