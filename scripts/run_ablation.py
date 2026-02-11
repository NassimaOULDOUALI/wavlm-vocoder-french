"""
Layer Ablation Study
====================

Run systematic ablation of WavLM layers.

Tests N=1, 2, 3, ..., 12 layers and evaluates reconstruction quality.

Usage:
    python scripts/run_ablation.py \
        --base_config configs/experiments/ablation_layers.yaml \
        --output_dir ./outputs/ablation
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from omegaconf import OmegaConf


def run_training(config_path, num_gpus=4):
    """
    Run training with torchrun.
    
    Args:
        config_path: Path to config file
        num_gpus: Number of GPUs
    """
    cmd = [
        'torchrun',
        '--standalone',
        f'--nproc_per_node={num_gpus}',
        'scripts/train.py',
        '--config', str(config_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Run layer ablation study')
    parser.add_argument('--base_config', type=str, required=True, help='Base config file')
    parser.add_argument('--output_dir', type=str, default='./outputs/ablation', help='Output directory')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--layers', type=str, default='1,2,3,4,6,9,12', help='Comma-separated layer counts to test')
    args = parser.parse_args()
    
    # Parse layer counts
    layer_counts = [int(x) for x in args.layers.split(',')]
    
    print(f"\n{'='*80}")
    print(f"WavLM Layer Ablation Study")
    print(f"{'='*80}")
    print(f"Base config: {args.base_config}")
    print(f"Testing layers: {layer_counts}")
    print(f"Output dir: {args.output_dir}")
    print("")
    
    # Load base config
    base_config = load_config(args.base_config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ablation for each layer count
    for num_layers in layer_counts:
        print(f"\n{'='*80}")
        print(f"Running ablation: {num_layers} layers")
        print(f"{'='*80}\n")
        
        # Create config for this ablation
        config = OmegaConf.merge(base_config, OmegaConf.create({
            'model': {'num_layers': num_layers},
            'training': {'output_dir': str(output_dir / f'layers_{num_layers}')}
        }))
        
        # Save config
        config_path = output_dir / f'config_layers_{num_layers}.yaml'
        save_config(config, config_path)
        
        # Run training
        try:
            run_training(config_path, args.num_gpus)
            print(f"\n✓ Completed ablation for {num_layers} layers\n")
        except subprocess.CalledProcessError as e:
            print(f"\n⚠️ Failed ablation for {num_layers} layers: {e}\n")
            continue
    
    print(f"\n{'='*80}")
    print(f"✓ Ablation study complete!")
    print(f"✓ Results in: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
