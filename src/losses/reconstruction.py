"""Spectral and time-domain losses for audio reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """Time-domain L1 loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class MultiScaleSTFTLoss(nn.Module):
    """Multi-Scale STFT Loss: spectral convergence + log-magnitude L1."""

    def __init__(
        self,
        fft_sizes: list | None = None,
        hop_sizes: list | None = None,
        win_sizes: list | None = None,
        factor_sc: float = 0.5,
        factor_mag: float = 0.5,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes or [2048, 1024, 512, 256, 128]
        self.hop_sizes = hop_sizes or [512, 256, 128, 64, 32]
        self.win_sizes = win_sizes or [2048, 1024, 512, 256, 128]
        assert len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_sizes)
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes, strict=True
        ):
            window = torch.hann_window(win_size).to(pred.device)
            stft_kwargs = dict(
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                window=window,
                return_complex=True,
            )
            mag_pred = torch.abs(torch.stft(pred, **stft_kwargs)) + self.eps
            mag_target = torch.abs(torch.stft(target, **stft_kwargs)) + self.eps

            sc_loss = torch.norm(mag_target - mag_pred, p="fro") / (
                torch.norm(mag_target, p="fro") + self.eps
            )
            mag_loss = F.l1_loss(torch.log(mag_pred), torch.log(mag_target))
            total_loss += self.factor_sc * sc_loss + self.factor_mag * mag_loss

        return total_loss / len(self.fft_sizes)
