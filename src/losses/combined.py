"""
Combined Loss
=============

Combines reconstruction and GAN losses.
"""

import torch
import torch.nn as nn
from .reconstruction import L1Loss, MultiScaleSTFTLoss
from .gan import GeneratorAdversarialLoss, FeatureMatchingLoss


class CombinedLoss(nn.Module):
    """
    Combined loss for training WavLM vocoder.
    
    Supports two modes:
        - No GAN: L1 + Multi-Scale STFT
        - With GAN: L1 + Multi-Scale STFT + Adversarial + Feature Matching
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Reconstruction losses (always used)
        self.l1_loss = L1Loss()
        self.stft_loss = MultiScaleSTFTLoss()
        
        # GAN losses (optional)
        self.use_gan = config.loss.use_gan
        if self.use_gan:
            self.adv_loss = GeneratorAdversarialLoss()
            self.fm_loss = FeatureMatchingLoss()
        
        # Weights
        self.l1_weight = config.loss.l1_weight
        self.stft_weight = config.loss.stft_weight
        
        if self.use_gan:
            self.adv_weight = config.loss.adv_weight
            self.fm_weight = config.loss.fm_weight
    
    def forward(self, pred, target, disc_outputs=None, disc_features=None):
        """
        Compute combined loss.
        
        Args:
            pred: (B, T) predicted waveform
            target: (B, T) target waveform
            disc_outputs: Optional list of discriminator outputs (for GAN)
            disc_features: Optional tuple of (real_features, fake_features) (for GAN)
            
        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction losses
        l1 = self.l1_loss(pred, target)
        stft = self.stft_loss(pred, target)
        
        total_loss = self.l1_weight * l1 + self.stft_weight * stft
        
        loss_dict = {
            'l1_loss': l1.item(),
            'stft_loss': stft.item(),
        }
        
        # GAN losses (if enabled)
        if self.use_gan and disc_outputs is not None:
            # Adversarial loss
            adv = self.adv_loss(disc_outputs)
            total_loss += self.adv_weight * adv
            loss_dict['adv_loss'] = adv.item()
            
            # Feature matching loss
            if disc_features is not None:
                real_feats, fake_feats = disc_features
                fm = self.fm_loss(real_feats, fake_feats)
                total_loss += self.fm_weight * fm
                loss_dict['fm_loss'] = fm.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
