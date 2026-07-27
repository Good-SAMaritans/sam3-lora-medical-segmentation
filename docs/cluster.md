# Running on PACE-ICE

How to run the SAM3 training on the ICE cluster (CUDA GPUs). Batch jobs go through
SLURM; the batch script is [scripts/slrum/train_lora.sbatch](../scripts/slurm/train_lora.sbatch).

## The one storage rule that matters

Home (`~`) is only **15 GB**. Scratch (`~/scratch`) is **300 GB**. The conda env,
the 6.9 GB gated SAM3 weights, and the ~1.8 GB checkpoints will blow the home
quota. So put **everything in scratch**: clone the repo there, install the env
there, point the HF cache there. Scratch is not backed up and is purged after
120 days of no access, which is fine for this.

Find your scratch path with `pace-quota` if `~/scratch` does not exist (see the
class PACE guide).

## One-time setup (on a login node)

```bash
# 1. connect: GT VPN (GlobalProtect) on, then
ssh <gtid>@login-ice.pace.gatech.edu

# 2. work entirely in scratch
cd ~/scratch
git clone <repo-url> group_project
cd group_project
git checkout lora-adaptation

# 3. conda env IN SCRATCH (--prefix keeps env files off the home quota)
module load anaconda3/2022.05.0.1        # confirm version: module spider anaconda
module load cuda                          # then: nvcc --version  (note the CUDA version)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --prefix ~/scratch/sam3-cluster python=3.12 -y
conda activate ~/scratch/sam3-cluster

# 4. torch matching the node's CUDA (check nvcc --version; example for cu121)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. the native sam3 package, CLEAN official clone (NOT the Mac vendor/ copy;
#    no triton stub and no device="cuda" patches are needed on CUDA)
cd ~/scratch
git clone https://github.com/facebookresearch/sam3.git
pip install -e sam3
pip install "setuptools<81"               # sam3 uses pkg_resources

# 6. the project's own deps
cd ~/scratch/group_project
pip install -r requirements.txt

# 7. gated weights: log in, and keep the cache in scratch
export HF_HOME=~/scratch/hf               # also set in the sbatch script
hf auth login                             # paste a read token
```

## Submit a training job

```bash
cd ~/scratch/group_project
mkdir -p slurm_outs
sbatch scripts/slurm/train_lora.sbatch          # queues the job, prints a job id
squeue -u $USER                           # watch it (PD = pending, R = running)
tail -f slurm_outs/slurm_<jobid>.out      # live output once it starts
scancel <jobid>                           # cancel if needed
```

The job writes checkpoints to `checkpoints/sam3_lora_busi_full/` (inside the repo,
which is in scratch, so no quota problem). `best.pt` is updated after every epoch,
so a job that hits the walltime still leaves a usable best-so-far model.

## Notes

- **GPU choice:** the sbatch requests an H100 (`--gres=gpu:H100:1`) — team standard
  as of 2026-07-27 (a BUSI full run is ~36 min on H100 vs ~80 min on A100). Do NOT
  use V100s: the cu128 torch build dropped their compute capability. The model only
  needs ~9 GB, so any H100/A100 has plenty of memory.
- **Limits:** max 8 h walltime and 16 GPU-hours per job. This run is ~1-2 h.
- **Shared storage:** `/storage/ice-shared/cs7643/shared-group-project-data`
  (50 GB/team) is a good place to stage the SAM3 weights and datasets once so the
  team does not each re-download. Name the folder after the team.
- **Env per node type:** anaconda/cuda versions can differ between GPU node types.
  If activation misbehaves on a new node, `module spider anaconda` / `module spider
  cuda` and adjust, or make a second env for that node type.
