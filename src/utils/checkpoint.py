"""
Checkpoint Utilities
====================
"""

import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    step,
    config,
    save_dir,
    filename='checkpoint.pt',
    is_best=False
):
    """
    Save training checkpoint.
    
    Args:
        model: Model (or DDP model)
        optimizer: Optimizer
        scaler: GradScaler for AMP
        epoch: Current epoch
        step: Global step
        config: Configuration
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract state dict from DDP if needed
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config,
    }
    
    # Save checkpoint
    save_path = save_dir / filename
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")
    
    # Save as latest
    latest_path = save_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)
    
    # Save as best if applicable
    if is_best:
        best_path = save_dir / 'checkpoint_best.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Best checkpoint saved: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """
    Load checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scaler: Optional scaler to load state
        
    Returns:
        epoch, step from checkpoint
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler state
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    
    logger.info(f"Checkpoint loaded: epoch={epoch}, step={step}")
    
    return epoch, step
