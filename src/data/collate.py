"""
Collate Functions
=================
"""

import torch


def collate_fn(batch):
    """
    Simple collate function for audio batches.
    
    Args:
        batch: List of (T,) audio tensors
        
    Returns:
        (B, T) batched tensor
    """
    return torch.stack(batch, dim=0)
