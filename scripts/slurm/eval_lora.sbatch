#!/bin/bash
# Score the LoRA checkpoints with the shared 6-metric evaluator (scripts/eval_model.py).
# Evaluation only - no training, no checkpoints written.
#
# Submit from the repo root:
#   sbatch scripts/slurm/eval_lora.sbatch              # all three datasets
#   sbatch scripts/slurm/eval_lora.sbatch monusac      # just one
# Watch:  squeue -u $USER    then    tail -f slurm_outs/lora/slurm_<jobid>.out
#
# Results are appended to results/metrics/lora_eval.csv, which notebook 02 reads
# for its results table.

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8                               # CPU cores (dataloader workers)
#SBATCH --gres=gpu:H100:1                  # team standard; NOT V100 (cu128 torch dropped CC 7.0)
#SBATCH --mem=32G
#SBATCH -t 01:00:00                        # 3 backbone builds + ~430 forward passes; minutes on an H100
#SBATCH -J eval_lora
#SBATCH -o slurm_outs/lora/slurm_%j.out    # Slurm does NOT create this dir; keep the .gitkeep

set -e
echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

module load anaconda3/2023.03
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/sam3-cluster

# gated SAM3 weights (6.9GB) live in scratch, not the 15GB home quota
export HF_HOME=~/scratch/hf

# eval_lora.py does sys.path.insert(0,"src") and opens configs/... relatively,
# so cwd must be the repo root
cd "$SLURM_SUBMIT_DIR"

echo "datasets: ${*:-all}"
python scripts/eval_lora.py "$@"
echo "done"
