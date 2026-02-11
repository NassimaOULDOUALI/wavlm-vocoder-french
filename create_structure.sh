#!/bin/bash

# Script pour créer l'arborescence complète du projet WavLM Vocoder

set -e

echo "🚀 Création de l'arborescence du projet WavLM Vocoder..."
echo ""

# Aller dans le repo
cd ~/wavlm-vocoder-french

# Créer tous les dossiers
echo "📁 Création des dossiers..."

mkdir -p configs/experiments
mkdir -p src/{data,models,losses,trainers,utils}
mkdir -p scripts
mkdir -p notebooks
mkdir -p data/{raw,processed}
mkdir -p outputs/{checkpoints,logs,samples}
mkdir -p tests
mkdir -p docs

echo "✓ Dossiers créés"
echo ""

# Créer les fichiers racine
echo "📄 Création des fichiers racine..."

touch .gitignore
touch README.md
touch requirements.txt
touch setup.py
touch CITATION.bib
touch LICENSE

echo "✓ Fichiers racine créés"
echo ""

# Configs
echo "⚙️  Création des configs..."

touch configs/base.yaml
touch configs/experiments/no_gan.yaml
touch configs/experiments/gan.yaml
touch configs/experiments/ablation_layers.yaml

echo "✓ Configs créés"
echo ""

# Source code - __init__.py
echo "🐍 Création des __init__.py..."

touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/losses/__init__.py
touch src/trainers/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py

echo "✓ __init__.py créés"
echo ""

# Source code - Data
echo "📊 Création des modules data..."

touch src/data/dataset.py
touch src/data/collate.py
touch src/data/preprocessing.py

echo "✓ Modules data créés"
echo ""

# Source code - Models
echo "🧠 Création des modules models..."

touch src/models/wavlm_vocoder.py
touch src/models/adapter.py
touch src/models/generator.py
touch src/models/discriminator.py

echo "✓ Modules models créés"
echo ""

# Source code - Losses
echo "📉 Création des modules losses..."

touch src/losses/reconstruction.py
touch src/losses/gan.py
touch src/losses/combined.py

echo "✓ Modules losses créés"
echo ""

# Source code - Trainers
echo "🏋️  Création des trainers..."

touch src/trainers/trainer.py

echo "✓ Trainers créés"
echo ""

# Source code - Utils
echo "🛠️  Création des utils..."

touch src/utils/config.py
touch src/utils/audio.py
touch src/utils/logging.py
touch src/utils/checkpoint.py

echo "✓ Utils créés"
echo ""

# Scripts
echo "📜 Création des scripts..."

touch scripts/train.py
touch scripts/infer.py
touch scripts/eval.py
touch scripts/run_ablation.py
touch scripts/train.slurm

echo "✓ Scripts créés"
echo ""

# Notebooks
echo "📓 Création des notebooks..."

touch notebooks/1_data_exploration.ipynb
touch notebooks/2_model_architecture.ipynb
touch notebooks/3_results_analysis.ipynb

echo "✓ Notebooks créés"
echo ""

# Tests
echo "🧪 Création des tests..."

touch tests/test_models.py
touch tests/test_losses.py
touch tests/test_data.py

echo "✓ Tests créés"
echo ""

# Docs
echo "📚 Création de la documentation..."

touch docs/INSTALL.md
touch docs/TRAINING.md
touch docs/EVALUATION.md

echo "✓ Documentation créée"
echo ""

# Afficher l'arborescence
echo "=========================================="
echo "🌳 ARBORESCENCE CRÉÉE"
echo "=========================================="
echo ""

# Afficher avec tree si disponible, sinon avec find
if command -v tree &> /dev/null; then
    tree -L 3 -I '__pycache__|*.pyc|.git' --dirsfirst
else
    echo "📂 wavlm-vocoder-french/"
    find . -maxdepth 4 -not -path '*/\.*' -not -path '*/__pycache__/*' | sort | sed 's|^\./||' | sed 's|[^/]*/|  |g'
fi

echo ""
echo "=========================================="
echo "✅ STRUCTURE COMPLÈTE CRÉÉE !"
echo "=========================================="
echo ""
echo "📊 Statistiques:"
echo "  - Dossiers: $(find . -type d -not -path '*/\.*' | wc -l)"
echo "  - Fichiers: $(find . -type f -not -path '*/\.*' -not -path '*/__pycache__/*' | wc -l)"
echo ""
echo "📍 Localisation: $(pwd)"
echo ""

