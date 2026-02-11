"""
Training Script
===============

Main entry point for training WavLM Vocoder.

Usage:
    # Single GPU
    python scripts/train.py --config configs/base.yaml
    
    # Multi-GPU with torchrun
    torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/base.yaml
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.trainers.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train WavLM Vocoder')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(
        log_dir=Path(config.training.output_dir) / 'logs',
        rank=0  # Will be updated in Trainer
    )
    
    logger.info("=" * 80)
    logger.info("WavLM Vocoder Training")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info("")
    
    # Create trainer and run
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
