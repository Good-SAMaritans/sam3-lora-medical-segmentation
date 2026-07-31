"""Run the SAM 3 full-fine-tuning arm from an experiment config."""

import sys
from typing import Any

import torch
import yaml

sys.path.insert(0, "src")

from medseg.data.medseg_data import DataConfig, build_dataloaders
from medseg.engine import Trainer
from medseg.models import Sam3MedSeg
from medseg.utils.device import get_device


DEFAULT_CONFIG_PATH = "configs/experiment/sam3_full_busi.yaml"


def load_config(path: str) -> dict[str, Any]:
    """Load and validate the experiment configuration."""
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    required_sections = {"data", "model", "train"}
    missing_sections = required_sections - config.keys()
    if missing_sections:
        raise KeyError(
            f"Missing config sections: {sorted(missing_sections)}"
        )

    if config["model"].get("finetune_mode") != "full":
        raise ValueError(
            "train_full.py requires model.finetune_mode: full"
        )

    return config


def main() -> None:
    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_CONFIG_PATH
    )
    config = load_config(config_path)
    device = get_device()

    print(f"device: {device}")
    print(f"config: {config_path}")

    data_config = DataConfig.from_dict(config["data"])
    loaders = build_dataloaders(data_config)
    split_sizes = {
        name: len(loader.dataset)
        for name, loader in loaders.items()
    }
    print(f"splits: {split_sizes}")

    model = Sam3MedSeg(**config["model"])
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.trainable_parameters()
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_percent = 100 * trainable_parameters / total_parameters

    print(
        f"parameters: {trainable_parameters:,} trainable / "
        f"{total_parameters:,} total "
        f"({trainable_percent:.4f}%)"
    )

    trainer = Trainer(
        model,
        loaders,
        ckpt_dir=config["train"]["ckpt_dir"],
        backbone_lr=config["train"]["backbone_lr"],
        head_lr=config["train"]["head_lr"],
        device=device,
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_dice = trainer.fit(config["train"]["epochs"])
    summary = f"complete: best_val_dice={best_dice:.4f}"

    if device == "cuda":
        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / 1e9
        )
        summary += f"  peak_vram={peak_vram_gb:.1f}GB"

    print(summary)


if __name__ == "__main__":
    main()