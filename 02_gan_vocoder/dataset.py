"""Audio dataset for WavLM vocoder training.

Loads audio files recursively, resamples to target rate, randomly crops or
pads to a fixed segment length, and peak-normalises to [-peak_target, peak_target].

Supports .wav, .flac, .mp3, .ogg, .m4a formats.
"""

import logging
import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


class AudioDataset(Dataset):
    """Recursive audio dataset with peak normalisation.

    Args:
        audio_dir:        Root directory (searched recursively).
        segment_length:   Samples per training segment (default 32000 = 2 s @ 16 kHz).
        sample_rate:      Target sample rate in Hz.
        use_rms_norm:     If True, normalise by RMS; otherwise normalise by peak.
        rms_threshold:    Minimum RMS to accept a segment (used both for filtering
                          and as denominator floor in RMS normalisation).
        peak_target:      Target peak amplitude when use_rms_norm=False.
    """

    def __init__(
        self,
        audio_dir: str,
        segment_length: int = 32000,
        sample_rate: int = 16000,
        use_rms_norm: bool = False,
        rms_threshold: float = 0.005,
        peak_target: float = 0.99,
    ):
        self.audio_dir = audio_dir
        self.segment_length = int(segment_length)
        self.sample_rate = int(sample_rate)
        self.use_rms_norm = bool(use_rms_norm)
        self.rms_threshold = float(rms_threshold)
        self.peak_target = float(peak_target)

        self.audio_files: list[str] = []
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.lower().endswith(AUDIO_EXTENSIONS):
                    self.audio_files.append(os.path.join(root, f))

        if not self.audio_files:
            raise ValueError(f"No audio files found in {audio_dir}")

        logger.info(
            "AudioDataset: %d files found in %s "
            "(segment=%d samples, sr=%d Hz, peak_target=%.2f)",
            len(self.audio_files), audio_dir, segment_length, sample_rate, peak_target,
        )

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a normalised waveform of shape (segment_length,)."""
        max_retries = 5
        for _ in range(max_retries):
            path = self.audio_files[idx]
            try:
                wav, sr = torchaudio.load(path)

                # Mix-down to mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                # Resample if needed
                if sr != self.sample_rate:
                    wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)

                # Random crop or zero-pad
                if wav.shape[1] >= self.segment_length:
                    start = random.randint(0, wav.shape[1] - self.segment_length)
                    wav = wav[:, start : start + self.segment_length]
                else:
                    wav = F.pad(wav, (0, self.segment_length - wav.shape[1]))

                # Silence gate
                if wav.pow(2).mean() < 1e-8:
                    idx = random.randint(0, len(self) - 1)
                    continue

                # Normalisation
                if self.use_rms_norm:
                    rms = wav.pow(2).mean().sqrt().clamp(min=self.rms_threshold)
                    wav = (wav / (rms + 1e-8) * 0.1).clamp(-1.0, 1.0)
                else:
                    peak = wav.abs().max()
                    if peak > 1e-6:
                        wav = wav / (peak + 1e-8) * self.peak_target

                # Post-normalisation silence gate
                if self.use_rms_norm and wav.pow(2).mean().sqrt() < self.rms_threshold:
                    idx = random.randint(0, len(self) - 1)
                    continue

                wav = wav.clamp(-1.0, 1.0).float().squeeze(0)  # (T,)
                return wav

            except Exception as exc:
                logger.warning("Failed to load %s: %s — retrying", path, exc)
                idx = random.randint(0, len(self) - 1)

        raise RuntimeError(
            f"Could not load a valid audio sample after {max_retries} retries."
        )


def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    """Stack waveforms into a batch tensor of shape (B, T)."""
    waveforms = torch.stack(batch, dim=0)
    assert torch.isfinite(waveforms).all(), "Non-finite values detected in batch"
    return waveforms


class ValidationDataset(AudioDataset):
    """Deterministic version of AudioDataset: always takes the start of each file."""

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.audio_files[idx]
        wav, sr = torchaudio.load(path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)

        if wav.shape[1] >= self.segment_length:
            wav = wav[:, : self.segment_length]
        else:
            wav = F.pad(wav, (0, self.segment_length - wav.shape[1]))

        if self.use_rms_norm:
            rms = wav.pow(2).mean().sqrt().clamp(min=self.rms_threshold)
            wav = (wav / (rms + 1e-8) * 0.1).clamp(-1.0, 1.0)
        else:
            peak = wav.abs().max()
            if peak > 1e-6:
                wav = wav / (peak + 1e-8) * self.peak_target

        return wav.clamp(-1.0, 1.0).float().squeeze(0)
