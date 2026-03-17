# 02 — GAN vocoder

This module adds adversarial supervision (MPD + MSD + Feature Matching) on top of the baseline architecture. It produces the results reported in the paper.

## Architecture

Same encoder-adapter-generator stack as `01_baseline_no_gan`, plus:

- **Learned layer fusion** — instead of using only `last_hidden_state`, the model learns a softmax-weighted sum over selected WavLM transformer layers.
- **Multi-Period Discriminator (MPD)** — analyses the waveform at periods {2, 3, 5, 7, 11} to capture periodic structure (pitch).
- **Multi-Scale Discriminator (MSD)** — analyses the waveform at 3 temporal scales (original, ×2, ×4 downsample) to capture texture at different resolutions.
- **Feature Matching loss** — L1 distance between discriminator intermediate activations.

```
Audio (B, T)
    │
    ▼
WavLM-Base+ (frozen)
    │  hidden_states[i], i ∈ {last N layers}
    │  weighted sum with learnable softmax weights α_i
    ▼
WavLMAdapter  (B, 256, T')
    │
    ▼
HiFiGeneratorImproved  →  ŷ (B, T)
    │
    ├──[train]── MPD(ŷ), MSD(ŷ)  ←→  MPD(y), MSD(y)
    └──[train]── Feature Matching + Mel + L1 + STFT
```

## Training objective

```
L_G = λ_l1·L1(ŷ,y) + λ_mel·Mel(ŷ,y) + λ_stft·STFT(ŷ,y)
    + Ladv(MPD,MSD) + λ_fm·LFM(MPD,MSD)

L_D = Hinge(MPD, y, ŷ) + Hinge(MSD, y, ŷ)
```

Default weights: `λ_l1=1, λ_mel=45, λ_stft=1, λ_fm=2`.  
The first 10,000 steps use spectral losses only (no adversarial) to stabilise the generator.

## Results (from paper)

| Model | MCD↓ | Mel-L1↓ | PESQ↑ | STOI↑ | V/UV F1↑ | F0 RMSE↓ | F0 Corr↑ |
|---|---|---|---|---|---|---|---|
| No GAN | 9.72 | 1.55 | 1.11 | 0.74 | 0.878 | 10.1 | 0.83 |
| **+MPD/MSD+FM** | **8.43** | **1.17** | **1.28** | **0.86** | **0.932** | **7.7** | **0.96** |

Evaluated on 15 stratified test samples (unseen speakers, 1.5–5 s).

## Best layer configuration

From the ablation study (`../03_ablation_study/`):

```yaml
feature_mode: "weighted_last_n"
wavlm_last_n: 7    # layers 6-12
```

Layers 7–12 carry most of the phonetic/prosodic information useful for reconstruction.

## Files

| File | Role |
|---|---|
| `src/model.py` | `WavLM2AudioImproved` — full model with layer fusion |
| `src/discriminators.py` | `MultiPeriodDiscriminator`, `MultiScaleDiscriminator` |
| `src/losses.py` | `CombinedGANLoss`, `GANLoss`, `MelSpectrogramLoss` |
| `train.py` | DDP training loop with G/D alternation |
| `config.yaml` | Default hyperparameters |

Dataset code is shared with `../01_baseline_no_gan/dataset.py`.

## Quick start

**1. Install**
```bash
pip install torch torchaudio transformers pyyaml tqdm
```

**2. Edit `config.yaml`** — set `data.train_dir` and `model.wavlm_model_name`.

**3. Train (4 GPUs)**
```bash
torchrun --nproc_per_node=4 train.py --config config.yaml
```

**Override layer config at launch (e.g. for SLURM array):**
```bash
torchrun --nproc_per_node=4 train.py \
    --config config.yaml \
    --feature_mode weighted_last_n \
    --wavlm_last_n 7 \
    --output_dir ./outputs_N7
```
