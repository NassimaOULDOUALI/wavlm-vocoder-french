"""
Reconstruction Losses
=====================

Spectral and time-domain losses for audio reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """Simple L1 time-domain loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T) predicted waveform
            target: (B, T) target waveform
            
        Returns:
            Scalar loss
        """
        return F.l1_loss(pred, target)


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-Scale STFT Loss.
    
    Computes spectral convergence and log-magnitude L1 loss
    across multiple FFT scales.
    """
    
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512, 256, 128],
        hop_sizes=[512, 256, 128, 64, 32],
        win_sizes=[2048, 1024, 512, 256, 128],
        factor_sc=0.5,
        factor_mag=0.5,
        eps=1e-7
    ):
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T) predicted waveform
            target: (B, T) target waveform
            
        Returns:
            Scalar loss
        """
        total_loss = 0.0
        
        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            # Create window
            window = torch.hann_window(win_size).to(pred.device)
            
            # Compute STFT
            S_pred = torch.stft(
                pred,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                window=window,
                return_complex=True
            )
            
            S_target = torch.stft(
                target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                window=window,
                return_complex=True
            )
            
            # Magnitude
            mag_pred = torch.abs(S_pred) + self.eps
            mag_target = torch.abs(S_target) + self.eps
            
            # Spectral Convergence Loss
            sc_loss = torch.norm(mag_target - mag_pred, p='fro') / (
                torch.norm(mag_target, p='fro') + self.eps
            )
            
            # Log-Magnitude L1 Loss
            log_mag_pred = torch.log(mag_pred)
            log_mag_target = torch.log(mag_target)
            mag_loss = F.l1_loss(log_mag_pred, log_mag_target)
            
            # Weighted combination
            scale_loss = self.factor_sc * sc_loss + self.factor_mag * mag_loss
            total_loss += scale_loss
        
        # Average across scales
        total_loss = total_loss / len(self.fft_sizes)
        
        return total_loss


class MelSpectrogramLoss(nn.Module):
    """
    Mel-Spectrogram L1 Loss.
    
    Useful for perceptual quality.
    """
    
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000
    ):
        super().__init__()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T) predicted waveform
            target: (B, T) target waveform
            
        Returns:
            Scalar loss
        """
        mel_pred = self.mel_transform(pred)
        mel_target = self.mel_transform(target)
        
        # Log scale
        log_mel_pred = torch.log(mel_pred + 1e-5)
        log_mel_target = torch.log(mel_target + 1e-5)
        
        return F.l1_loss(log_mel_pred, log_mel_target)
