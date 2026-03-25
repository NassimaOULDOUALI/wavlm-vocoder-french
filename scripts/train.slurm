#!/bin/bash
#SBATCH --job-name=wavlm_vocoder
#SBATCH --account=lsq@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err

set -euo pipefail

# ============================================================================
# WavLM Vocoder Training - Jean Zay
# ============================================================================

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=========================================="
echo "  WavLM Vocoder Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: 4"
echo "Start: $(date)"
echo ""

# Working directory (UPDATE THIS)
WORK_DIR="$HOME/wavlm-vocoder-french"
cd "$WORK_DIR"

# Environment setup
module purge
ENV_PREFIX="/lustre/fsn1/projects/rech/lsq/umv83if/conda_envs/torch_env"
export PATH="$ENV_PREFIX/bin:$PATH"

echo "Working dir: $(pwd)"
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__)"
echo ""

# Configuration file
CONFIG_FILE="configs/base.yaml"

echo "Config: $CONFIG_FILE"
echo ""

# Launch training
echo "Starting training..."
srun --ntasks=1 bash -lc \
    "torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     scripts/train.py --config $CONFIG_FILE"

echo ""
echo "Training completed: $(date)"
