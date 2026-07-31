#!/bin/bash
# SLURM batch script for SAM 3 full fine-tuning on PACE-ICE.
#
# Submit from the repository root:
#   mkdir -p slurm_outs
#   sbatch scripts/train_full.sbatch
#
# Pass a different config when needed:
#   sbatch scripts/train_full.sbatch \
#     configs/experiment/sam3_full_kvasir_full.yaml
#
# Watch:
#   squeue -u $USER
#   tail -f slurm_outs/slurm_<job_id>.out

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J sam3_full
#SBATCH -o slurm_outs/slurm_%j.out

set -e

echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# Use Sean's existing SAM 3 environment.
module load anaconda3/2023.03
module load cuda/12.9.1
PYTHON=/storage/ice1/4/1/skang411/envs/sam-medseg/bin/python

# Keep the gated SAM 3 weights in scratch storage.
export HF_HOME=/storage/ice1/4/1/skang411/hf_cache

# Relative config and source paths require the repository root.
cd "$SLURM_SUBMIT_DIR"

# BUSI is the default. A Kvasir config can be supplied to sbatch.
CONFIG="${1:-configs/experiment/sam3_full_busi_full.yaml}"
echo "config: $CONFIG"

"$PYTHON" -u scripts/train_full.py "$CONFIG"

echo "done"