#!/usr/bin/env python3
"""Inference script — reconstruct audio from a trained baseline checkpoint.

Usage:
    python inference.py \\
        --checkpoint outputs/checkpoints/checkpoint_latest.pt \\
        --input_dir  /path/to/audio_files \\
        --output_dir generated_samples \\
        --num_samples 10
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from dataset import AudioDataset
from model import WavLM2Audio


def load_model(ckpt_path: str, device: torch.device) -> tuple[WavLM2Audio, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]
    mc   = cfg["model"]

    model = WavLM2Audio(
        wavlm_model_name   = mc["wavlm_model_name"],
        hidden_dim         = int(mc["hidden_dim"]),
        num_adapter_layers = int(mc.get("num_adapter_layers", 6)),
        kernel_size        = int(mc.get("kernel_size", 7)),
        freeze_wavlm       = bool(mc.get("freeze_wavlm", True)),
        dropout            = float(mc.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    print(f"Loaded checkpoint: step={ckpt.get('step', '?')}  epoch={ckpt.get('epoch', '?')}")
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--output_dir",  default="generated_samples")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)

    dataset = AudioDataset(
        audio_dir      = args.input_dir,
        segment_length = int(cfg["data"]["segment_length"]),
        sample_rate    = int(cfg["data"]["sample_rate"]),
        peak_target    = float(cfg["data"]["peak_target"]),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = int(cfg["data"]["sample_rate"])

    print(f"Generating {args.num_samples} samples → {out_dir}")
    with torch.no_grad():
        for i in range(min(args.num_samples, len(dataset))):
            audio = dataset[i].unsqueeze(0).to(device)  # (1, T)
            out   = model(audio)                         # (1, T)
            torchaudio.save(str(out_dir / f"sample_{i+1:02d}_input.wav"),  audio.cpu(), sr)
            torchaudio.save(str(out_dir / f"sample_{i+1:02d}_output.wav"), out.cpu(),   sr)
            print(f"  [{i+1}/{args.num_samples}] written")

    print("Done.")


if __name__ == "__main__":
    main()
