"""
Evaluation Script
=================

Evaluate WavLM Vocoder on test set.

Computes metrics:
    - MCD (Mel-Cepstral Distortion)
    - PESQ (Perceptual Evaluation of Speech Quality)
    - STOI (Short-Time Objective Intelligibility)
    - F0 RMSE, F0 Correlation
    - V/UV F1

Usage:
    python scripts/eval.py \
        --checkpoint outputs/checkpoints/checkpoint_best.pt \
        --test_dir /path/to/test/audio \
        --output_dir ./eval_results
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.wavlm_vocoder import WavLM2Audio
from src.utils.audio import load_audio, process_long_audio


def compute_metrics(pred, target, sr=16000):
    """
    Compute evaluation metrics.
    
    Note: This is a placeholder. Actual implementation requires:
        - librosa for MCD
        - pesq library
        - pystoi library
        - pyworld for F0
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Placeholder - implement actual metrics
    # For now, just compute MSE
    mse = np.mean((pred - target) ** 2)
    metrics['mse'] = float(mse)
    
    # TODO: Implement real metrics
    # metrics['mcd'] = compute_mcd(pred, target, sr)
    # metrics['pesq'] = compute_pesq(pred, target, sr)
    # metrics['stoi'] = compute_stoi(pred, target, sr)
    # metrics['f0_rmse'] = compute_f0_rmse(pred, target, sr)
    # metrics['f0_corr'] = compute_f0_correlation(pred, target, sr)
    # metrics['vuv_f1'] = compute_vuv_f1(pred, target, sr)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate WavLM Vocoder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_samples', type=int, default=None, help='Max samples to evaluate')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"WavLM Vocoder Evaluation")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dir: {args.test_dir}")
    print("")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    model = WavLM2Audio(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")
    
    # Get test files
    test_dir = Path(args.test_dir)
    audio_files = sorted(test_dir.glob('**/*.wav'))
    if args.num_samples:
        audio_files = audio_files[:args.num_samples]
    
    print(f"Evaluating on {len(audio_files)} files...\n")
    
    # Evaluate
    all_metrics = []
    
    for audio_path in tqdm(audio_files):
        try:
            # Load
            audio, sr = load_audio(audio_path, target_sr=config.data.sample_rate)
            
            # Generate
            with torch.no_grad():
                output = process_long_audio(model, audio, device=device)
            
            # Match length
            min_len = min(len(audio), len(output))
            audio = audio[:min_len].numpy()
            output = output[:min_len].numpy()
            
            # Compute metrics
            metrics = compute_metrics(output, audio, sr)
            metrics['filename'] = audio_path.name
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"\n⚠️ Error: {audio_path.name}: {e}")
            continue
    
    # Aggregate results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Compute averages
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key != 'filename':
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f'{key}_std'] = float(np.std(values))
    
    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    for key, value in avg_metrics.items():
        if not key.endswith('_std'):
            std_key = f'{key}_std'
            if std_key in avg_metrics:
                print(f"{key:20s}: {value:.4f} ± {avg_metrics[std_key]:.4f}")
            else:
                print(f"{key:20s}: {value:.4f}")
    print(f"{'='*80}\n")
    
    print(f"✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
