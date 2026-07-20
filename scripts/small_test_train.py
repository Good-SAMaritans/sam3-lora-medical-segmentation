"""Tiny end-to-end small subset run: build data + model + trainer, run 2 epochs on 8 images."""
import sys, yaml
sys.path.insert(0, "src")

import torch
from medseg.data.medseg_data import DataConfig, build_dataloaders
from medseg.models import Sam3MedSeg
from medseg.engine import Trainer

cfg = yaml.safe_load(open("configs/experiment/sam3_full_busi.yaml"))

data_cfg = DataConfig.from_dict(cfg["data"])
loaders = build_dataloaders(data_cfg)
print("splits:", {k: len(v.dataset) for k, v in loaders.items()})

model = Sam3MedSeg(**cfg["model"])
print("trainable params:", sum(p.numel() for p in model.trainable_parameters()) / 1e6, "M")

trainer = Trainer(model, loaders,
                  ckpt_dir=cfg["train"]["ckpt_dir"],
                  backbone_lr=cfg["train"]["backbone_lr"],
                  head_lr=cfg["train"]["head_lr"])

torch.cuda.reset_peak_memory_stats()
best = trainer.fit(cfg["train"]["epochs"])
print(f"SMOKE OK — best_dice={best:.4f}  peak_VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")