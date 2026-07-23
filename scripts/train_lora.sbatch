#!/bin/bash
# SLURM batch script for the SAM3 LoRA arm on PACE-ICE.
#
# IMPORTANT: run everything from ~/scratch, not home. Home is only 15GB; the
# conda env + 6.9GB SAM3 weights + checkpoints will not fit. Scratch is 300GB.
# So: clone this repo into ~/scratch, put the HF cache in scratch (HF_HOME below),
# and the relative checkpoints/ dir then also lives in scratch.
#
# Submit from the repo root (on a login node is fine):
#   mkdir -p slurm_outs
#   sbatch scripts/train_lora.sbatch
# Watch:  squeue -u $USER    then    tail -f slurm_outs/slurm_<jobid>.out
# Cancel: scancel <jobid>

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8                               # CPU cores (dataloader workers)
#SBATCH --gres=gpu:A100:1                  # A100 (CC 8.0). NOT V100: cu128 torch dropped V100's CC 7.0.
#SBATCH --mem=32G
#SBATCH -t 03:00:00                        # walltime (max 8h; our run is ~1-2h)
#SBATCH -J lora_busi
#SBATCH -o slurm_outs/slurm_%j.out         # stdout+stderr (%j = job id); mkdir this first

set -e
echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# --- software ---
# versions differ by node; confirm with `module spider anaconda` / `module spider cuda`
module load anaconda3/2023.03              # 2023+ (present on A100 nodes; provides conda)
module load cuda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/sam3-cluster     # the env you created in scratch (see docs/cluster.md)

# keep the gated SAM3 weights (6.9GB) in scratch, not the 15GB home quota
export HF_HOME=~/scratch/hf

# run from the repo root: the script does sys.path.insert(0,"src") and opens
# configs/... as relative paths, so cwd must be the repo root (submit from there)
cd "$SLURM_SUBMIT_DIR"

# config path can be passed to sbatch (e.g. a SLURM test with the smoke config):
#   sbatch scripts/train_lora.sbatch configs/experiment/sam3_lora_busi.yaml
# defaults to the full run if none given
CONFIG="${1:-configs/experiment/sam3_lora_busi_full.yaml}"
echo "config: $CONFIG"
python scripts/train_lora.py "$CONFIG"
echo "done"
