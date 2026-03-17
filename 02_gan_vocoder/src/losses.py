"""Losses pour entraînement GAN du synthétiseur vocal.

Inclut:
- Adversarial losses (hinge loss)
- Feature matching loss
- Mel-spectrogram loss
- Combined loss avec tous les composants

Usage:
    from losses_gan import GANLoss, MelSpectrogramLoss, CombinedGANLoss
    
    gan_loss = GANLoss()
    mel_loss = MelSpectrogramLoss()
    
    # Discriminator training
    loss_d = gan_loss.discriminator_loss(d_real, d_fake)
    
    # Generator training
    loss_g_adv = gan_loss.generator_loss(d_fake)
    loss_fm = gan_loss.feature_matching_loss(d_real, d_fake)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class GANLoss(nn.Module):
    """Adversarial losses pour GAN training.
    
    Supporte:
    - Hinge loss (défaut, recommandé)
    - Least-squares loss
    """
    def __init__(self, loss_type='hinge'):
        super().__init__()
        self.loss_type = loss_type
    
    def discriminator_loss(self, disc_real_outputs, disc_fake_outputs):
        """
        Loss pour le discriminateur.
        
        Args:
            disc_real_outputs: List[(output, fmaps)] du discriminateur sur vrais échantillons
            disc_fake_outputs: List[(output, fmaps)] du discriminateur sur faux échantillons
        Returns:
            loss: scalar
        """
        loss = 0
        for (real_out, _), (fake_out, _) in zip(disc_real_outputs, disc_fake_outputs):
            if self.loss_type == 'hinge':
                real_loss = torch.mean(F.relu(1 - real_out))
                fake_loss = torch.mean(F.relu(1 + fake_out))
            elif self.loss_type == 'lsgan':
                real_loss = torch.mean((real_out - 1) ** 2)
                fake_loss = torch.mean(fake_out ** 2)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            loss += (real_loss + fake_loss)
        
        return loss
    
    def generator_loss(self, disc_fake_outputs):
        """
        Adversarial loss pour le générateur.
        
        Args:
            disc_fake_outputs: List[(output, fmaps)] du discriminateur sur faux échantillons
        Returns:
            loss: scalar
        """
        loss = 0
        for fake_out, _ in disc_fake_outputs:
            if self.loss_type == 'hinge':
                loss += torch.mean(-fake_out)
            elif self.loss_type == 'lsgan':
                loss += torch.mean((fake_out - 1) ** 2)
        
        return loss
    
    def feature_matching_loss(self, disc_real_outputs, disc_fake_outputs):
        """
        Feature matching loss: L1 entre feature maps du discriminateur.
        
        Force le générateur à produire des représentations intermédiaires
        similaires aux vrais signaux. Très efficace pour la stabilité.
        
        Args:
            disc_real_outputs: List[(output, fmaps)]
            disc_fake_outputs: List[(output, fmaps)]
        Returns:
            loss: scalar
        """
        loss = 0
        for (_, real_fmaps), (_, fake_fmaps) in zip(disc_real_outputs, disc_fake_outputs):
            for real_fmap, fake_fmap in zip(real_fmaps, fake_fmaps):
                loss += F.l1_loss(fake_fmap, real_fmap.detach())
        
        return loss * 2  # Coefficient par défaut de HiFi-GAN


class MelSpectrogramLoss(nn.Module):
    """Mel-Spectrogram Loss.
    
    Calcule la L1 loss entre les log-mel spectrogrammes.
    Plus perceptuellement pertinente que la STFT linéaire car
    suit l'échelle de perception humaine (mel scale).
    """
    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=8000,
        center=False,
    ):
        super().__init__()
        self.hop_length = hop_length
        
        # Utiliser torchaudio pour le mel spectrogram
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=1.0,  # Magnitude, pas power
            center=center,
        )
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [B, T] predicted waveform
            y_true: [B, T] target waveform
        Returns:
            loss: scalar
        """
        # Move mel_spec to same device as input
        self.mel_spec = self.mel_spec.to(y_pred.device)
        
        # Compute mel spectrograms
        mel_pred = self.mel_spec(y_pred)
        mel_true = self.mel_spec(y_true)
        
        # Log-mel L1 loss
        log_mel_pred = torch.log(mel_pred.clamp(min=1e-5))
        log_mel_true = torch.log(mel_true.clamp(min=1e-5))
        
        return F.l1_loss(log_mel_pred, log_mel_true)


class MultiScaleSTFTLoss(nn.Module):
    """Multi-Scale STFT Loss améliorée.
    
    Version optimisée avec:
    - Epsilon stable
    - Spectral convergence + log-magnitude loss
    """
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512, 256],
        hop_sizes=[512, 256, 128, 64],
        win_sizes=[2048, 1024, 512, 256],
        eps=1e-7,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.eps = eps
    
    def forward(self, y_pred, y_true):
        total_loss = 0
        
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Compute STFT
            window = torch.hann_window(win_size, device=y_pred.device)
            
            S_pred = torch.stft(
                y_pred, n_fft=fft_size, hop_length=hop_size,
                win_length=win_size, window=window, return_complex=True
            )
            S_true = torch.stft(
                y_true, n_fft=fft_size, hop_length=hop_size,
                win_length=win_size, window=window, return_complex=True
            )
            
            # Magnitudes
            mag_pred = torch.abs(S_pred) + self.eps
            mag_true = torch.abs(S_true) + self.eps
            
            # Spectral convergence
            sc_loss = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + self.eps)
            
            # Log-magnitude L1
            mag_loss = F.l1_loss(torch.log(mag_pred), torch.log(mag_true))
            
            total_loss += 0.5 * sc_loss + 0.5 * mag_loss
        
        return total_loss / len(self.fft_sizes)


class CombinedGANLoss(nn.Module):
    """Loss combinée pour entraînement GAN complet.
    
    Combine:
    - L1 waveform loss
    - Mel-spectrogram loss
    - Multi-scale STFT loss
    - Adversarial loss
    - Feature matching loss
    
    Poids recommandés (basés sur HiFi-GAN):
    - mel_weight: 45.0 (très important pour la qualité perceptuelle)
    - fm_weight: 2.0
    - l1_weight: 1.0
    - stft_weight: 1.0
    """
    def __init__(
        self,
        l1_weight=1.0,
        mel_weight=45.0,
        stft_weight=1.0,
        fm_weight=2.0,
        sample_rate=16000,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.mel_weight = mel_weight
        self.stft_weight = stft_weight
        self.fm_weight = fm_weight
        
        self.gan_loss = GANLoss(loss_type='hinge')
        self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)
        self.stft_loss = MultiScaleSTFTLoss()
    
    def discriminator_step(self, disc_real_outputs, disc_fake_outputs):
        """
        Calcule la loss pour le discriminateur.
        
        Returns:
            loss: scalar
            loss_dict: dict avec les composants
        """
        loss_d = self.gan_loss.discriminator_loss(disc_real_outputs, disc_fake_outputs)
        return loss_d, {'loss_d': loss_d.item()}
    
    def generator_step(
        self,
        y_pred,
        y_true,
        mpd_real=None,
        mpd_fake=None,
        msd_real=None,
        msd_fake=None,
    ):
        """
        Calcule toutes les losses pour le générateur.
        
        Args:
            y_pred: [B, T] predicted waveform
            y_true: [B, T] target waveform
            mpd_real/fake: sorties MPD (optionnel pour pure reconstruction)
            msd_real/fake: sorties MSD (optionnel pour pure reconstruction)
        
        Returns:
            total_loss: scalar
            loss_dict: dict avec tous les composants
        """
        loss_dict = {}
        total_loss = 0
        
        # Reconstruction losses
        loss_l1 = F.l1_loss(y_pred, y_true)
        loss_dict['loss_l1'] = loss_l1.item()
        total_loss += self.l1_weight * loss_l1
        
        loss_mel = self.mel_loss(y_pred, y_true)
        loss_dict['loss_mel'] = loss_mel.item()
        total_loss += self.mel_weight * loss_mel
        
        loss_stft = self.stft_loss(y_pred, y_true)
        loss_dict['loss_stft'] = loss_stft.item()
        total_loss += self.stft_weight * loss_stft
        
        # Adversarial losses (si discriminateurs fournis)
        if mpd_fake is not None and msd_fake is not None:
            # Generator adversarial loss
            loss_gen_mpd = self.gan_loss.generator_loss(mpd_fake)
            loss_gen_msd = self.gan_loss.generator_loss(msd_fake)
            loss_dict['loss_gen_mpd'] = loss_gen_mpd.item()
            loss_dict['loss_gen_msd'] = loss_gen_msd.item()
            total_loss += loss_gen_mpd + loss_gen_msd
            
            # Feature matching loss
            if mpd_real is not None and msd_real is not None:
                loss_fm_mpd = self.gan_loss.feature_matching_loss(mpd_real, mpd_fake)
                loss_fm_msd = self.gan_loss.feature_matching_loss(msd_real, msd_fake)
                loss_dict['loss_fm_mpd'] = loss_fm_mpd.item()
                loss_dict['loss_fm_msd'] = loss_fm_msd.item()
                total_loss += self.fm_weight * (loss_fm_mpd + loss_fm_msd)
        
        loss_dict['loss_total'] = total_loss.item()
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test
    print("="*60)
    print("Test des losses GAN")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test data
    batch_size = 4
    seq_len = 16000
    
    y_true = torch.randn(batch_size, seq_len, device=device).clamp(-1, 1)
    y_pred = y_true + torch.randn_like(y_true) * 0.1
    y_pred = y_pred.clamp(-1, 1)
    
    print(f"\nInput shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
    
    # Test individual losses
    mel_loss = MelSpectrogramLoss().to(device)
    stft_loss = MultiScaleSTFTLoss().to(device)
    
    loss_mel = mel_loss(y_pred, y_true)
    loss_stft = stft_loss(y_pred, y_true)
    
    print(f"\nMel loss: {loss_mel.item():.6f}")
    print(f"STFT loss: {loss_stft.item():.6f}")
    
    # Test combined loss (sans discriminateurs)
    combined = CombinedGANLoss().to(device)
    total_loss, loss_dict = combined.generator_step(y_pred, y_true)
    
    print(f"\nCombined loss (sans GAN):")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n✅ Test réussi!")
