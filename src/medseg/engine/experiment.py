"""
experiment.py - one run end to end: data -> model -> fit -> test metrics -> figure

`run` names every output, so a run's checkpoint, figure and CSV row line up:
    {out_dir}/checkpoints/{run}/best.pt
    {out_dir}/figures/{run}/predictions.png
"""

import os
import time

import torch

from ..data.medseg_data import DataConfig, build_dataloaders
from ..models import UNet
from ..utils.device import get_device
from ..utils.viz import save_predictions
from .metrics import mean_score, resize_mask, seg_scores
from .trainer import Trainer




@torch.no_grad()
def evaluate(model, loader, device: str, ignore_index: int | None = None) -> tuple[float, float]:
    model.eval()
    dices, nsds = [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        mask = resize_mask(batch["mask"].to(device), logits.shape[-2:])
        dsc, nsd_v = seg_scores(logits, mask, tol=2, ignore_index=ignore_index)
        dices.append(dsc)
        nsds.append(nsd_v)
    return mean_score(dices), mean_score(nsds)


def run_experiment(cfg: dict, run: str) -> dict:
    out_dir = cfg["train"].get("out_dir", "results")
    ckpt_dir = os.path.join(out_dir, "checkpoints", run)
    device = get_device()

    data_cfg = DataConfig.from_dict(cfg["data"])
    loaders = build_dataloaders(data_cfg)
    model = UNet(**cfg["model"])

    trainer = Trainer(model, loaders, ckpt_dir=ckpt_dir,
                      backbone_lr=cfg["train"]["backbone_lr"],
                      head_lr=cfg["train"]["head_lr"],
                      weight_decay=cfg["train"].get("weight_decay", 0.01),
                      device=device,
                      num_classes=cfg["model"]["num_classes"])

    start = time.time()
    val_dice = trainer.fit(cfg["train"]["epochs"])
    minutes = (time.time() - start) / 60

    # report test metrics + figure from the best checkpoint,, not the last epoch
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    model.load_state_dict(torch.load(ckpt_path)["model"])
    test_dice, test_nsd = evaluate(model, loaders["test"], device,
                                   ignore_index=cfg["train"].get("ignore_index"))
    save_predictions(model, loaders["test"], os.path.join(out_dir, "figures", run, "predictions.png"),
                     n=5, device=device, normalize=data_cfg.normalize)

    # sweeps keep the fgiure + CSV row only; 100+ checkpoints is tens of GB
    if not cfg["train"].get("keep_checkpoint", True):
        os.remove(ckpt_path)

    return {
        "run": run,
        "dataset": data_cfg.name,
        "img_size": data_cfg.effective_img_size(),
        "encoder": cfg["model"]["encoder_name"],
        "pretrained": cfg["model"]["encoder_weights"] or "none",
        "batch_size": data_cfg.batch_size,
        "aug": data_cfg.aug_preset,
        "backbone_lr": cfg["train"]["backbone_lr"],
        "head_lr": cfg["train"]["head_lr"],
        "epochs": cfg["train"]["epochs"],
        "val_dice": round(val_dice, 4),
        "test_dice": round(test_dice, 4),
        "test_nsd": round(test_nsd, 4),
        "minutes": round(minutes, 1),
    }
