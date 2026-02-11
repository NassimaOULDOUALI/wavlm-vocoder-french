"""
Discriminators for GAN Training
================================

Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):
    """
    Single period discriminator.
    
    Processes 1D signal as 2D with specific period.
    """
    
    def __init__(self, period):
        super().__init__()
        
        self.period = period
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
    
    def forward(self, x):
        """
        Args:
            x: (B, T) waveform
            
        Returns:
            output: (B, 1, ...)
            features: List of intermediate features
        """
        features = []
        
        # Reshape to 2D
        B, T = x.shape
        if T % self.period != 0:
            padding = self.period - (T % self.period)
            x = F.pad(x, (0, padding), mode='reflect')
            T = T + padding
        
        x = x.view(B, 1, T // self.period, self.period)
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator with multiple periods."""
    
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in periods
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, T) waveform
            
        Returns:
            outputs: List of discriminator outputs
            features: List of feature lists
        """
        outputs = []
        features = []
        
        for disc in self.discriminators:
            output, feat = disc(x)
            outputs.append(output)
            features.append(feat)
        
        return outputs, features


class ScaleDiscriminator(nn.Module):
    """Single scale discriminator."""
    
    def __init__(self):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 16, 15, 1, padding=7),
            nn.Conv1d(16, 64, 41, 4, groups=4, padding=20),
            nn.Conv1d(64, 256, 41, 4, groups=16, padding=20),
            nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
            nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ])
        
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, T) waveform
            
        Returns:
            output: (B, 1, T')
            features: List of intermediate features
        """
        features = []
        
        x = x.unsqueeze(1)  # (B, 1, T)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        
        x = torch.flatten(x, 1, -1)
        
        return x, features


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator with average pooling."""
    
    def __init__(self, num_scales=3):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(num_scales)
        ])
        
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2) for _ in range(num_scales - 1)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, T) waveform
            
        Returns:
            outputs: List of discriminator outputs
            features: List of feature lists
        """
        outputs = []
        features = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling[i - 1](x)
            
            output, feat = disc(x)
            outputs.append(output)
            features.append(feat)
        
        return outputs, features
