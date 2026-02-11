"""
Logging Utilities
=================
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir, rank=0, level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        rank: Process rank (for distributed training)
        level: Logging level
        
    Returns:
        Logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        f'[%(asctime)s][Rank {rank}][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if rank == 0:
        fh = logging.FileHandler(log_dir / 'train.log')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Console handler (only for rank 0)
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger
