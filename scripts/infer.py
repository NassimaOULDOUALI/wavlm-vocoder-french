"""
Inference Script
================

Generate audio from trained WavLM Vocoder.

Usage:
    python scripts/infer.py \
        --checkpoint outputs/checkpoints/checkpoint_best.pt \
        --input_dir /path/to/audio \
        --output_dir ./generated \
        --num_samples 10
"""

import sys
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.wavlm_vocoder import WavLM2Audio
from src.utils.audio import load_audio, save_audio, process_long_audio


def main():
    parser = argparse.ArgumentParser(description='WavLM Vocoder Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--input_dir', type=str, default=None, help='Input directory')
    parser.add_argument('--input_file', type=str, default=None, help='Single input file')
    parser.add_argument('--output_dir', type=str, default='./generated', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--chunk_size', type=int, default=320000, help='Chunk size for long audio')
    args = parser.parse_args()
    
    if args.input_dir is None and args.input_file is None:
        raise ValueError("Must provide either --input_dir or --input_file")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"WavLM Vocoder Inference")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = WavLM2Audio(config).to(device)
    
    # Load weights
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    print("✓ Model loaded\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    if args.input_file:
        input_files = [Path(args.input_file)]
    else:
        input_dir = Path(args.input_dir)
        audio_extensions = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
        input_files = []
        for ext in audio_extensions:
            input_files.extend(input_dir.glob(f'*{ext}'))
        input_files = sorted(input_files)[:args.num_samples]
    
    print(f"Processing {len(input_files)} files...\n")
    
    # Process each file
    for input_path in tqdm(input_files):
        try:
            # Load audio
            audio, sr = load_audio(input_path, target_sr=config.data.sample_rate)
            
            # Process
            output = process_long_audio(
                model,
                audio,
                chunk_size=args.chunk_size,
                device=device
            )
            
            # Match length
            if len(output) > len(audio):
                output = output[:len(audio)]
            elif len(output) < len(audio):
                pad_size = len(audio) - len(output)
                output = torch.nn.functional.pad(output, (0, pad_size))
            
            # Normalize
            peak = output.abs().max()
            if peak > 1e-6:
                output = output / peak * 0.95
            output = torch.clamp(output, -1.0, 1.0)
            
            # Save
            stem = input_path.stem
            save_audio(audio, output_dir / f'{stem}_input.wav', sr)
            save_audio(output, output_dir / f'{stem}_output.wav', sr)
            
        except Exception as e:
            print(f"\n⚠️ Error processing {input_path.name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"✓ Generated {len(input_files)} samples")
    print(f"✓ Saved to: {output_dir.resolve()}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
