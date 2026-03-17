"""Training losses for the baseline (no-GAN) vocoder.

Combined loss:
    L = λ_l1 · L1(ŷ, y) + λ_stft · MultiScaleSTFT(ŷ, y)

MultiScaleSTFT averages over several STFT resolutions:
    per scale: 0.5 · SpectralConvergence + 0.5 · LogMagnitudeL1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (linear STFT, log-magnitude).

    Args:
        fft_sizes:  List of FFT sizes for each scale.
        hop_sizes:  Corresponding hop lengths.
        win_sizes:  Corresponding window lengths.
        factor_sc:  Weight for spectral convergence term.
        factor_mag: Weight for log-magnitude L1 term.
        eps:        Epsilon added before log to avoid log(0).
    """

    def __init__(
        self,
        fft_sizes: tuple = (2048, 1024, 512, 256, 128),
        hop_sizes: tuple = (512, 256, 128, 64, 32),
        win_sizes: tuple = (2048, 1024, 512, 256, 128),
        factor_sc: float = 0.5,
        factor_mag: float = 0.5,
        eps: float = 1e-7,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.fft_sizes  = fft_sizes
        self.hop_sizes  = hop_sizes
        self.win_sizes  = win_sizes
        self.factor_sc  = factor_sc
        self.factor_mag = factor_mag
        self.eps        = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (B, T) predicted waveform
            y_true: (B, T) target waveform
        Returns:
            Scalar loss averaged over scales.
        """
        total = torch.zeros(1, device=y_pred.device)

        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            window = torch.hann_window(win, device=y_pred.device)

            S_pred = torch.stft(y_pred, fft, hop, win, window, return_complex=True)
            S_true = torch.stft(y_true, fft, hop, win, window, return_complex=True)

            mag_pred = S_pred.abs() + self.eps
            mag_true = S_true.abs() + self.eps

            # Spectral convergence
            sc = torch.norm(mag_true - mag_pred, p="fro") / (
                torch.norm(mag_true, p="fro") + self.eps
            )

            # Log-magnitude L1
            mag = F.l1_loss(mag_pred.log(), mag_true.log())

            total = total + self.factor_sc * sc + self.factor_mag * mag

        return total / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """L1 waveform loss + multi-scale STFT loss.

    Args:
        stft_loss_weight: Weight applied to the STFT term.
        l1_loss_weight:   Weight applied to the L1 term.
    """

    def __init__(self, stft_loss_weight: float = 1.0, l1_loss_weight: float = 0.5):
        super().__init__()
        self.stft_weight = stft_loss_weight
        self.l1_weight   = l1_loss_weight
        self.stft_loss   = MultiScaleSTFTLoss()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            loss_dict:  {"loss", "l1", "stft"}
        """
        l1   = F.l1_loss(y_pred, y_true)
        stft = self.stft_loss(y_pred, y_true)
        total = self.l1_weight * l1 + self.stft_weight * stft

        return total, {"loss": total.item(), "l1": l1.item(), "stft": stft.item()}
