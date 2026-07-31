"""Train the U-Net baseline: python scripts/train_unet.py [config.yaml]"""
import os, sys, yaml
sys.path.insert(0, "src")

from medseg.engine.experiment import run_experiment
from medseg.utils.tracking import log_run

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment/unet_busi.yaml"
cfg = yaml.safe_load(open(cfg_path))
run = os.path.splitext(os.path.basename(cfg_path))[0]

row = run_experiment(cfg, run)
log_run(os.path.join(cfg["train"].get("out_dir", "results"), "metrics", "runs.csv"), row)
print(f"done - {row}")
