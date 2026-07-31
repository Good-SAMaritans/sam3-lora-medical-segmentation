#!/usr/bin/env bash
# Run the three 512px sweeps back to back (~5.3h measured). One CSV per sweep.
# Each sweep is resumable, so re-running skips whatever is already in its CSV,
# and a sweep that dies does not stop the next one.
#
#   ./scripts/run_sweeps.sh                        # uses the active python
#   PYTHON=~/miniconda3/envs/sam-medseg/bin/python ./scripts/run_sweeps.sh
set -u
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
mkdir -p results/logs

for name in busi512 kvasir512 abdomenus512; do
    echo "=== $name  started $(date +%H:%M)"
    "$PYTHON" scripts/sweep_unet.py "configs/experiment/unet_sweep_${name}.yaml" \
        2>&1 | tee "results/logs/unet_sweep_${name}.log"
done

echo "=== all sweeps done $(date +%H:%M) -> results/metrics/"
