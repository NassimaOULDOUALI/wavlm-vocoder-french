# 01 — Baseline vocoder (no GAN)

This module trains a WavLM-to-audio vocoder using only spectral and waveform losses (no adversarial supervision). It serves as the **Stage 1 foundation** for the voice conversion pipeline and produces intelligible but perceptually synthetic ("robotic") speech.

## Architecture

```
Audio (B, T)
    │
    ▼
WavLM-Base+ (frozen)
    │  last_hidden_state (B, T', 768)
    ▼
WavLMAdapter
    6 × [Conv1d + BatchNorm + GELU + Dropout + residual]
    (B, 256, T')
    │
    ▼
HiFiGenerator  (320× upsampling: 8 × 5 × 4 × 2)
    Conv → [ConvTranspose + multi-dilation ResBlocks] × 4 → Conv → tanh
    (B, T)
```

**WavLM frame rate** ≈ 50 Hz (stride 320 samples at 16 kHz), so `T' ≈ T / 320`.

## Training objective

```
L = 0.5 · L1(ŷ, y)  +  1.0 · MultiScaleSTFT(ŷ, y)
```

MultiScaleSTFT averages over 5 STFT resolutions (FFT 2048→128). Per scale:
- Spectral Convergence: `‖|S_y| − |S_ŷ|‖_F / ‖|S_y|‖_F`
- Log-Magnitude L1: `‖log|S_ŷ| − log|S_y|‖₁`

## Files

| File | Role |
|---|---|
| `model.py` | `WavLM2Audio` — full model (`WavLMAdapter` + `HiFiGenerator`) |
| `dataset.py` | `AudioDataset` — recursive loader, crop/pad, peak normalisation |
| `losses.py` | `CombinedLoss` (L1 + MultiScaleSTFT) |
| `train.py` | DDP training loop (torchrun) |
| `inference.py` | Reconstruct audio from a checkpoint |
| `config.yaml` | Default hyperparameters |

## Quick start

**1. Install dependencies**
```bash
pip install torch torchaudio transformers pyyaml tqdm
```

**2. Edit `config.yaml`**
```yaml
data:
  train_dir: "/path/to/your/french_corpus"
model:
  wavlm_model_name: "microsoft/wavlm-base-plus"  # or local path
```

**3. Train (4 GPUs)**
```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

**Single GPU:**
```bash
python train.py --config config.yaml
```

**4. Generate samples**
```bash
python inference.py \
    --checkpoint outputs/checkpoints/checkpoint_latest.pt \
    --input_dir  /path/to/test_audio \
    --num_samples 10
```

## Expected behaviour

- Loss converges in ~50k steps on 4 GPUs with 238h of French speech.
- Reconstructed audio is **intelligible but synthetic** — the spectrogram is smooth and lacks fine harmonic texture. This is expected without GAN supervision.
- For better perceptual quality, see `../02_gan_vocoder/`.

## Checkpoints

Each checkpoint contains:

```python
{
    "step":                 int,
    "epoch":                int,
    "model_state_dict":     ...,
    "optimizer_state_dict": ...,
    "scaler_state_dict":    ...,
    "config":               dict,   # full config embedded for reproducibility
}
```

To resume training, set `training.resume: true` in `config.yaml`.
