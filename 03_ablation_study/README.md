# 03 — Ablation study

Layer ablation and adversarial supervision analysis from the paper:

> **"WavLM-to-Audio Vocoding in French: Layer Ablation Study and Adversarial Supervision for Continuous Voice Conversion"**

## What was studied

**Question 1 — Which WavLM layers are most useful?**  
We trained N=1…7 configurations using the N last transformer layers of WavLM-Base+ (layers 12, 11–12, 10–12, …, 6–12) with both uniform averaging and learned weighted fusion.

**Question 2 — How much does GAN supervision help?**  
Direct comparison of spectral-only vs MPD/MSD+FM training on a fixed configuration.

## Key findings

### GAN supervision

| Model | MCD↓ | Mel-L1↓ | PESQ↑ | STOI↑ | F0 RMSE↓ | F0 Corr↑ |
|---|---|---|---|---|---|---|
| No GAN | 9.72 | 1.55 | 1.11 | 0.74 | 10.1 | 0.83 |
| +MPD/MSD+FM | **8.43** | **1.17** | **1.28** | **0.86** | **7.7** | **0.96** |
| Δ (relative) | −13.3% | −24.5% | +15.3% | +16.2% | −23.8% | +15.7% |

### Layer importance (learned weights)

From the weighted fusion experiments, layer 6 dominates when only 1 layer is used (α=0.80), while for N=7 (layers 6–12) the distribution becomes more uniform. The upper transformer layers (7–12) collectively carry the most phonetic/prosodic information.

## Figures

| File | Description |
|---|---|
| `figures/ablation_losses.pdf` | Training loss curves per N configuration |
| `figures/ablation_weights.pdf` | Learned layer weights across configurations |
| `figures/heatmap_weights.pdf` | Heatmap of α weights by layer and N |
| `figures/layers.pdf` | Metric scores as a function of N |
| `figures/barplot.png` | Bar chart comparing GAN vs no-GAN |

## Results

| File | Description |
|---|---|
| `results/results_FINAL.csv` | Final metrics for all ablation runs |
| `results/results_ablation_N1to6.csv` | Per-checkpoint metrics for N=1…6 |

## Source code

| File | Description |
|---|---|
| `src/model.py` | `WavLM2AudioImproved` — supports `last`, `last_n_mean`, `weighted_all`, `weighted_last_n` |
| `src/discriminators.py` | MPD + MSD |
| `src/losses.py` | GAN + FM + Mel + STFT losses |
| `src/train.py` | Training loop (used for all N configurations) |
| `src/evaluate.py` | Evaluation pipeline (MCD, PESQ, STOI, F0, V/UV) |
| `config.yaml` | Base config (overridden per run via CLI) |

## Reproducing the ablation

Each configuration was launched as a SLURM array job:

```bash
torchrun --nproc_per_node=4 src/train.py \
    --config config.yaml \
    --feature_mode weighted_last_n \
    --wavlm_last_n N \
    --output_dir ./runs/N${N}_layers$(( 13-N ))-12
```

For N ∈ {1, 2, 3, 4, 5, 6, 7}.

## Evaluation

```bash
python src/evaluate.py \
    --runs_dir ./runs \
    --test_dir  /path/to/test_audio \
    --output_dir ./eval_results
```

Metrics: MCD (WORLD-MCEP), Log-Mel L1, PESQ, STOI, F0 RMSE, F0 Correlation, V/UV F1.
