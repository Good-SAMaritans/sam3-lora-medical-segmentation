"""Tiny end-to-end small subset run: build data + model + trainer, run 2 epochs on 8 images."""
import sys, yaml
sys.path.insert(0, "src")

import torch
from medseg.data.medseg_data import DataConfig, build_dataloaders
from medseg.utils.device import get_device
from medseg.models import Sam3MedSeg
from medseg.models.sam3_medseg import SAM3_INPUT_SIZE
from medseg.engine import Trainer


def main():
    #use cuda or mps when available
    device = get_device()
    print("device:", device)

    #config path from the command line, defaulting to the smoke config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment/lora/smoke/sam3_lora_busi.yaml"
    print("config:", config_path)
    cfg = yaml.safe_load(open(config_path))

    data_cfg = DataConfig.from_dict(cfg["data"])
    loaders = build_dataloaders(data_cfg)
    print("splits:", {k: len(v.dataset) for k, v in loaders.items()})

    model = Sam3MedSeg(**cfg["model"])
    print("trainable params:", sum(p.numel() for p in model.trainable_parameters()) / 1e6, "M")

    #log the resolved resolutions: the config path alone doesn't say what a run trained at, and a
    #256-vs-512 mix-up is invisible in the loss curve. head_out < img_size means the GT is being
    #downsampled to score, head_out > img_size means it's being upsampled (see Trainer._step_batch).
    print(f"resolution: img_size={data_cfg.effective_img_size()}  backbone_in={SAM3_INPUT_SIZE}  "
          f"head_out={model.head.out_size}")

    #num_classes selects the loss: 1 -> BceDiceLoss, >1 -> CeDiceLoss (see engine/losses.build_loss).
    #Without it the Trainer default of 1 would train a multi-class run against a binary loss.
    trainer = Trainer(model, loaders,
                      ckpt_dir=cfg["train"]["ckpt_dir"],
                      backbone_lr=cfg["train"]["backbone_lr"],
                      head_lr=cfg["train"]["head_lr"],
                      device=device,
                      num_classes=cfg["model"].get("num_classes", 1))

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best = trainer.fit(cfg["train"]["epochs"])
    if device == "cuda":
        print(f"SMOKE OK — best_dice={best:.4f}  peak_VRAM={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    else:
        print(f"SMOKE OK — best_dice={best:.4f}  (device={device})")


#macOS DataLoader workers re-import this file; the guard stops them
#re-running the whole script (which crashed with a nested-spawn error)
if __name__ == "__main__":
    main()
