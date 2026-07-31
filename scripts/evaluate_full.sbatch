#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J sam3_eval
#SBATCH -o slurm_outs/slurm_%j.out

set -e

module load anaconda3/2023.03
module load cuda/12.9.1

PYTHON=/storage/ice1/4/1/skang411/envs/sam-medseg/bin/python
export HF_HOME=/storage/ice1/4/1/skang411/hf_cache

cd "$SLURM_SUBMIT_DIR"

CONFIG="${1:-configs/experiment/sam3_full_busi_full.yaml}"

echo "job: $SLURM_JOB_ID"
echo "config: $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv

"$PYTHON" -u scripts/evaluate_full.py "$CONFIG"

echo "done"