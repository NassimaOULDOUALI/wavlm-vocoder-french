"""
HiFi-GAN Generator
==================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""
    
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.norm1 = nn.BatchNorm1d(channels)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size-1)//2)
        self.norm2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.norm1(x)
        
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.norm2(x)
        
        return x + residual


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN style generator with progressive upsampling.
    
    Total upsampling: 8 × 5 × 4 × 2 = 320
    """
    
    def __init__(
        self,
        hidden_dim=256,
        upsample_rates=[8, 5, 4, 2],
        upsample_kernel_sizes=[16, 10, 8, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    ):
        super().__init__()
        
        # Input convolution
        self.input_conv = nn.Conv1d(hidden_dim, 512, kernel_size=7, padding=3)
        
        # Upsampling blocks
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        channels = 512
        for rate, kernel in zip(upsample_rates, upsample_kernel_sizes):
            out_channels = channels // 2
            
            # Transposed convolution for upsampling
            self.ups.append(
                nn.ConvTranspose1d(
                    channels, out_channels,
                    kernel_size=kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2
                )
            )
            
            # Multi-receptive field ResBlocks
            resblock_list = nn.ModuleList()
            for k_size, dilations in zip(resblock_kernel_sizes, resblock_dilations):
                for dil in dilations:
                    resblock_list.append(ResBlock(out_channels, k_size, dil))
            self.resblocks.append(resblock_list)
            
            channels = out_channels
        
        # Output convolution
        self.output_conv = nn.Conv1d(channels, 1, kernel_size=7, padding=3)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, hidden_dim, T)
            
        Returns:
            (B, 1, T×320)
        """
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        
        for up, resblocks in zip(self.ups, self.resblocks):
            x = up(x)
            x = F.leaky_relu(x, 0.2)
            
            # Apply all resblocks and average
            xs = None
            for resblock in resblocks:
                if xs is None:
                    xs = resblock(x)
                else:
                    xs = xs + resblock(x)
            x = xs / len(resblocks)
        
        x = self.output_conv(x)
        
        # Adaptive normalization
        peak = x.abs().max(dim=-1, keepdim=True)[0]
        x = x / (peak + 1e-8) * 0.95
        
        return x
