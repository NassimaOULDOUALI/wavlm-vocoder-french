"""
GAN Losses
==========

Adversarial and feature matching losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorAdversarialLoss(nn.Module):
    """
    Generator adversarial loss.
    
    Encourages generator to fool discriminators.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, disc_outputs):
        """
        Args:
            disc_outputs: List of discriminator outputs for fake samples
            
        Returns:
            Scalar loss
        """
        loss = 0
        for disc_out in disc_outputs:
            # We want discriminator to output 1 (real) for fake samples
            loss += F.mse_loss(disc_out, torch.ones_like(disc_out))
        
        return loss / len(disc_outputs)


class DiscriminatorAdversarialLoss(nn.Module):
    """
    Discriminator adversarial loss.
    
    Encourages discriminator to classify real/fake correctly.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, disc_real_outputs, disc_fake_outputs):
        """
        Args:
            disc_real_outputs: List of discriminator outputs for real samples
            disc_fake_outputs: List of discriminator outputs for fake samples
            
        Returns:
            Scalar loss
        """
        loss = 0
        
        for real_out, fake_out in zip(disc_real_outputs, disc_fake_outputs):
            # Real samples should be classified as 1
            real_loss = F.mse_loss(real_out, torch.ones_like(real_out))
            
            # Fake samples should be classified as 0
            fake_loss = F.mse_loss(fake_out, torch.zeros_like(fake_out))
            
            loss += (real_loss + fake_loss)
        
        return loss / len(disc_real_outputs)


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss.
    
    Matches intermediate features between real and fake samples.
    This helps stabilize GAN training.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, disc_real_features, disc_fake_features):
        """
        Args:
            disc_real_features: List of lists of intermediate features for real
            disc_fake_features: List of lists of intermediate features for fake
            
        Returns:
            Scalar loss
        """
        loss = 0
        num_features = 0
        
        for real_feats, fake_feats in zip(disc_real_features, disc_fake_features):
            for real_feat, fake_feat in zip(real_feats, fake_feats):
                loss += F.l1_loss(real_feat, fake_feat)
                num_features += 1
        
        return loss / num_features if num_features > 0 else 0
