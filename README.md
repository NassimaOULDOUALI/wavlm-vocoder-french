# WavLM Vocoder for French 🎙️🇫🇷

![CI Tests](https://github.com/NassimaOULDOUALI/wavlm-vocoder-french/actions/workflows/ci.yml/badge.svg)
![Quick Checks](https://github.com/NassimaOULDOUALI/wavlm-vocoder-french/actions/workflows/quick-checks.yml/badge.svg)
![Python Versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **Vocodage WavLM vers audio en français : Ablation des couches et supervision adversariale**  
> Neural vocoder for reconstructing high-quality French speech from WavLM representations

📄 **Paper**: [Vocodage WavLM vers audio en français](docs/paper/JEP_2026_WavLM_Vocodeur.pdf)  
🎯 **Goal**: Stage 1 foundation for continuous voice conversion in WavLM latent space

---

## 🎯 Overview

This repository implements a neural vocoder that reconstructs audio from frozen **WavLM-Base+** representations, specifically trained and evaluated on French speech corpora. 

### Key Features

- ✅ **WavLM-Base+ Integration**: Frozen 12-layer transformer encoder (768-dim)
- ✅ **HiFi-GAN Generator**: Progressive upsampling (×320) with multi-receptive field residual blocks
- ✅ **Layer Ablation Study**: Systematic evaluation of N last layers (N=1...12)
- ✅ **Learned Layer Fusion**: Weighted combination vs. simple averaging
- ✅ **Adversarial Training**: MPD/MSD discriminators + Feature Matching
- ✅ **French Corpora**: SIWIS (10.9h) + M-AILABS (160.7h) + Common Voice (66.7h) = 238.3h
- ✅ **Comprehensive Metrics**: MCD, PESQ, STOI, F0-RMSE, V/UV F1

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│  Audio (16kHz) → WavLM-Base+ (frozen) → Layer Selection        │
│       ↓                                                          │
│  Learned Fusion (α₁h₁ + ... + αₙhₙ) → Adapter (768→256)       │
│       ↓                                                          │
│  HiFi-GAN Generator (×320 upsampling) → Reconstructed Audio    │
│       ↓                                                          │
│  [Optional] MPD/MSD Discriminators + Feature Matching          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results Summary

| Configuration | MCD↓ | Mel-L1↓ | PESQ↑ | STOI↑ | V/UV F1↑ | F0 RMSE↓ | F0 Corr↑ |
|--------------|------|---------|-------|-------|----------|----------|----------|
| **No GAN** | 9.72 | 1.55 | 1.11 | 0.74 | 0.878 | 10.1 | 0.83 |
| **+MPD/MSD+FM** | **8.43** | **1.17** | **1.28** | **0.86** | **0.932** | **7.7** | **0.96** |
| **Gain** | -13.3% | -24.5% | +15.3% | +16.2% | +6.1% | -23.8% | +15.7% |

> **Key Findings**:
> - Adversarial supervision (GAN) provides **consistent gains** across all metrics
> - Layers 7-12 capture most phonetic-prosodic information
> - Learned layer fusion outperforms fixed single-layer extraction

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/NassimaOULDOUALI/wavlm-vocoder-french.git
cd wavlm-vocoder-french

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed setup instructions.

### 2. Data Preparation
```bash
# Download datasets
bash scripts/download_data.sh

# Preprocess audio (normalize, filter, detect silence)
python scripts/preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --sample_rate 16000
```

### 3. Training
```bash
# Train without GAN (baseline)
bash scripts/train_model.sh --config configs/no_gan_config.yaml

# Train with GAN supervision
bash scripts/train_model.sh --config configs/with_gan_config.yaml

# Layer ablation experiments
bash scripts/train_model.sh --config configs/layer_ablation_config.yaml --num_layers 9
```

See [docs/TRAINING.md](docs/TRAINING.md) for advanced training options.

### 4. Evaluation
```bash
# Evaluate on test set
python scripts/evaluate_model.sh \
    --checkpoint outputs/checkpoints/best_model.pt \
    --test_dir data/processed/test \
    --output_dir outputs/evaluation
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for metric details.

### 5. Inference
```bash
# Reconstruct audio from WavLM representations
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --input_audio examples/sample.wav \
    --output_audio outputs/samples/reconstructed.wav \
    --num_layers 9
```

---

## 📁 Repository Structure
```
wavlm-vocoder-french/
├── src/
│   ├── models/
│   │   ├── adapter.py              # WavLM adapter (768→256)
│   │   ├── generator.py            # HiFi-GAN generator
│   │   ├── discriminator.py        # MPD/MSD discriminators
│   │   ├── wavlm_vocoder.py        # Main vocoder class
│   │   └── layer_fusion.py         # Learned layer weighting
│   ├── dataset/
│   │   ├── audio_dataset.py        # PyTorch dataset
│   │   └── preprocessing.py        # Audio cleaning pipeline
│   ├── training/
│   │   ├── train.py                # Training script
│   │   ├── losses.py               # L1, Mel, STFT, Adv, FM losses
│   │   └── trainer.py              # DDP/AMP trainer
│   ├── evaluation/
│   │   ├── metrics.py              # MCD, PESQ, STOI, F0, V/UV
│   │   └── evaluate.py             # Evaluation pipeline
│   └── utils/
│       ├── audio_processing.py     # Chunk inference, windowing
│       └── logger.py               # TensorBoard logging
├── configs/
│   ├── base_config.yaml            # Shared hyperparameters
│   ├── no_gan_config.yaml          # Baseline (spectral losses only)
│   ├── with_gan_config.yaml        # Full model (MPD/MSD+FM)
│   └── layer_ablation_config.yaml  # Layer sweep experiments
├── scripts/
│   ├── download_data.sh            # Download SIWIS, M-AILABS, CommonVoice
│   ├── preprocess_data.py          # Audio cleaning & filtering
│   ├── train_model.sh              # Multi-GPU training launcher
│   ├── evaluate_model.sh           # Compute all metrics
│   └── inference.py                # Reconstruct from checkpoint
├── notebooks/
│   ├── 1_data_exploration.ipynb    # Corpus statistics
│   ├── 2_model_architecture.ipynb  # Visualize layers, weights
│   └── 3_results_analysis.ipynb    # Plot metrics, ablations
├── tests/
│   ├── test_models.py              # Unit tests for architectures
│   ├── test_dataset.py             # Dataset loading tests
│   └── test_losses.py              # Loss function tests
├── docs/
│   ├── INSTALL.md                  # Installation guide
│   ├── TRAINING.md                 # Training instructions
│   ├── EVALUATION.md               # Metrics & evaluation
│   ├── REPRODUCTION.md             # Reproduce paper results
│   └── paper/
│       └── JEP_2026_WavLM_Vocodeur.pdf  # Accepted paper
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # This file
├── LICENSE                         # MIT License
└── CITATION.bib                    # BibTeX citation
```

---

## 📖 Documentation

- **[Installation Guide](docs/INSTALL.md)**: Detailed setup (CUDA, dependencies, data)
- **[Training Guide](docs/TRAINING.md)**: Hyperparameters, DDP/AMP, early stopping
- **[Evaluation Guide](docs/EVALUATION.md)**: Metric calculation, test protocols
- **[Reproduction Guide](docs/REPRODUCTION.md)**: Exact commands to reproduce paper results

---

## 🔬 Key Experiments

### Experiment 1: GAN vs. No-GAN
```bash
# Baseline (spectral losses only)
python src/training/train.py --config configs/no_gan_config.yaml

# With adversarial supervision
python src/training/train.py --config configs/with_gan_config.yaml
```

**Result**: GAN provides 13-24% improvement in spectral fidelity and 16% in perceptual quality.

### Experiment 2: Layer Ablation
```bash
# Test N=1,2,...,12 last layers
for N in {1..12}; do
    python src/training/train.py \
        --config configs/layer_ablation_config.yaml \
        --num_layers $N
done
```

**Result**: Layers 9-12 are optimal (sufficient information without redundancy).

### Experiment 3: Learned Fusion vs. Averaging
```yaml
# In config file:
layer_fusion:
  mode: "learned"  # or "average"
  num_layers: 9
```

**Result**: Learned fusion (weighted α) outperforms simple averaging.

---

## 📦 Pretrained Checkpoints

| Model | Layers | GAN | MCD | PESQ | Download |
|-------|--------|-----|-----|------|----------|
| Baseline | 12 | ❌ | 9.72 | 1.11 | [Link](#) |
| Best (N=9) | 9 | ✅ | **8.43** | **1.28** | [Download](https://huggingface.co/nassimaODL/wavlm-vocoder-french/blob/main/checkpoint_step180000.pt) |
| Lightweight (N=6) | 6 | ✅ | 8.89 | 1.21 | [Link](#) |

---

## 🎓 Citation

If you use this work, please cite our paper:
```bibtex
@inproceedings{wavlm_vocoder_french_2026,
  title={Vocodage WavLM vers audio en français : Ablation des couches et supervision adversariale comme fondation pour la conversion de voix continue},
  author={Nassima OULD OUALI, Awais Hussein Sani, Reda Dehak, Eric Moulines},
  booktitle={Journées d'Études sur la Parole (JEP)},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **WavLM**: Microsoft Research ([Chen et al., 2022](https://arxiv.org/abs/2110.13900))
- **HiFi-GAN**: [Kong et al., 2020](https://arxiv.org/abs/2010.05646)
- **Datasets**: SIWIS, M-AILABS, Common Voice
- **Metrics**: WORLD vocoder, CREPE pitch tracker

---

## 🛠️ Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions, please open an issue or contact: **nassima.ould-ouali@ip-paris.fr**

---

## 🗺️ Roadmap

- [x] Stage 1: Reconstruction vocoder (this work)
- [ ] Stage 2: Voice conversion in WavLM latent space
- [ ] Stage 3: Diffusion/Flow-based manipulation
- [ ] Real-time inference optimization

---


**Star ⭐ this repo if you find it useful!**
