#!/bin/bash
# =============================================================================
# Prereqs: on GT VPN, then  
# ssh <gtid>@login-ice.pace.gatech.edu
# Then:    bash setup_pace_env.sh
# =============================================================================
set -euo pipefail

# --- make sure scratch exists (the guide's procedure) ---
if [ ! -d "$HOME/scratch" ]; then
    echo "no ~/scratch link; creating one from pace-quota..."
    SCRATCH_REAL=$(pace-quota | awk '/Scratch:/ {print $1}' | sed 's|Scratch:||')
    mkdir -p "$HOME/scratch" 2>/dev/null || true
    [ -d "$HOME/scratch" ] || ln -s "$SCRATCH_REAL" "$HOME/scratch"
fi

ENV_PREFIX="$HOME/scratch/sam-medseg"

module load anaconda3            # guide notes versions vary by node; bare name picks a default
# keep conda's package cache in scratch too, or it fills home
conda config --add pkgs_dirs "$HOME/scratch/conda_pkgs" 2>/dev/null || true

if [ ! -d "$ENV_PREFIX" ]; then
    conda create --yes --prefix "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"

# guide warns conda-install is flaky on ICE; use pip3, and confirm it points to scratch
which pip3
case "$(which pip3)" in
  *"$HOME/scratch"*) : ;;   # good
  *) echo "WARNING: pip3 is NOT in scratch env -- see guide's troubleshooting before continuing" >&2 ;;
esac

# --- CUDA-matched torch. Check the node's CUDA, then pick the wheel. ---
# The guide's method: `module load cuda; nvcc --version` and match the whl url.
# cu121 works across the V100/A100 nodes; adjust if nvcc shows older.
module load cuda 2>/dev/null || true
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- project deps for the YOLO arm ---
pip3 install ultralytics pycocotools scipy opencv-python-headless medsegbench

python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
echo "env ready at: $ENV_PREFIX"
echo "activate later with:  conda activate $ENV_PREFIX"
