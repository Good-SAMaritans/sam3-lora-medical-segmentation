"""
medseg_data.py
==============
Config-driven ingestion for MedSegBench: turns the on-disk .npz files into
model-ready (image, mask) batches from a single DataConfig

Downloading the .npz from Zenodo is MedSegBench's job (the `download` flag).
This module only does bytes-on-disk -> batch dict:
    resolve_dataset_class -> build_dataset -> MedSegWrapper -> build_dataloaders

The wrapper fixes three awkward things about the raw MedSegBench sample:
  1. transform / target_transform are applied to image and mask independently,
     so it can't do synchronized spatial augmentation. We take the raw sample
     and augment image + mask jointly via albumentations (transforms.py)
  2. Channel count varies by modality; force_rgb expands grayscale to 3 channels
     for a consistent tensor shape
  3. Masks come out as {0,255} (binary) or raw class indices (multi-class); we
     normalize them to one format

Every dataset yields the same batch dict:
    {image, mask, task, num_classes, dataset, index}

    from medseg.data.medseg_data import DataConfig, build_dataloaders
    cfg = DataConfig.from_yaml("configs/data/busi.yaml")
    loaders = build_dataloaders(cfg)          # {'train','val','test'}
"""

from __future__ import annotations

import os
import inspect
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import medsegbench
from medsegbench.dataset import MedSegBench as _MedSegBase

from .transforms import build_transforms


# ImageNet normalization stats (used when cfg.normalize == "imagenet")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --- dataset registry    ---
# Built by introspection so it can't drift from the installed package. Each
# dataset is reachable by three lowercase keys — class name, Zenodo flag, and
# short name — e.g. KvasirMSBench -> {"kvasirmsbench", <flag>, "kvasir"}
def _build_registry() -> Dict[str, type]:
    registry: Dict[str, type] = {}
    for class_name, cls in inspect.getmembers(medsegbench, inspect.isclass):
        if not (issubclass(cls, _MedSegBase) and cls is not _MedSegBase):
            continue
        keys = {class_name.lower(), cls.flag.lower()}
        if class_name.lower().endswith("msbench"):
            keys.add(class_name.lower().removesuffix("msbench"))
        for key in keys:
            registry[key] = cls
    return registry


DATASET_REGISTRY = _build_registry()


def resolve_dataset_class(name: str) -> type:
    """Map a friendly name / flag / class name to its MedSegBench class"""
    key = name.lower().replace(" ", "").replace("-", "")
    if key not in DATASET_REGISTRY and name.lower() in DATASET_REGISTRY:
        key = name.lower()                              # allow flags like 'bkai-igh'
    if key not in DATASET_REGISTRY:
        options = sorted({v.__name__ for v in DATASET_REGISTRY.values()})
        raise KeyError(f"Unknown dataset '{name}'. Available: {options}")
    return DATASET_REGISTRY[key]


# --- config ---
@dataclass
class DataConfig:
    # dataset identity / acquisition
    name: str                                   # dataset key, e.g. "busi"
    size: int = 256                             # source resolution: 128 | 256 | 512
    root: str = "~/.medsegbench"               # where .npz files live / download to
    download: bool = True                       # fetch from Zenodo if missing
    category: Optional[str] = None              # sub-category for BUSI/WBC, e.g. "C1"

    # data-efficiency subset (None = full train split)
    n_train: Optional[int] = None
    subset_seed: int = 0

    # preprocessing
    force_rgb: bool = True                       # expand grayscale -> 3 channels
    img_size: Optional[int] = None               # target square size (falls back to size)
    normalize: Optional[str] = None              # None (/255) | "imagenet"

    # augmentation (shared across arms)
    aug_preset: str = "medseg_default"           # "medseg_default" | "none"

    # loader
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    def effective_img_size(self) -> int:
        return self.img_size or self.size

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        known = set(cls.__dataclass_fields__)       # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown config keys for DataConfig: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        if "data" in d and isinstance(d["data"], dict):   # tolerate a top-level 'data:' block
            d = d["data"]
        return cls.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- wrapper ---
class MedSegWrapper(Dataset):
    """Normalizes a raw MedSegBench dataset's output for training.

    Takes an already-built base dataset plus its info dict, so it can be unit
    tested against a mock with no download
    """

    def __init__(self, base: Dataset, info: Dict[str, Any], cfg: DataConfig, is_train: bool):
        self.base = base
        self.info = info
        self.cfg = cfg
        self.is_train = is_train
        self.task = info["task"]                          # 'binary' | 'multi-class'
        self.num_classes = len(info["pixel_labels"])      # includes background
        self.name = info.get("python_class", cfg.name)
        self.tf = build_transforms(cfg.effective_img_size(), is_train, cfg.aug_preset)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # One fixed pipeline, read top to bottom:
        #   load -> joint augment -> format mask -> normalize
        image, mask = self._load_sample(index)
        image, mask = self._augment(image, mask)
        mask = self._format_mask(mask)
        return {
            "image": self._normalize_image(image),
            "mask": mask,
            "task": self.task,
            "num_classes": self.num_classes,
            "dataset": self.name,
            "index": index,
        }

    def _load_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Raw base sample -> (image HWC uint8, mask HW uint8) numpy arrays

        MedSegBench hands back a bare (PIL image, mask). We only reshape here:
        grayscale is expanded to 3 channels and a stray channel axis is dropped
        from the mask. Augmentation/normalization come later
        """
        img_pil, target = self.base[index]
        if self.cfg.force_rgb and img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
        image = np.asarray(img_pil)
        if image.ndim == 2:
            image = image[:, :, None]
        mask = np.asarray(target)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return image, mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the shared albumentations policy jointly to image and mask,
        yielding uint8 tensors: image (3,H,W), mask (H,W)"""
        out = self.tf(image=image, mask=mask)
        return out["image"], out["mask"]

    def _format_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Binary -> (1,H,W) float {0,1}; multi-class -> (H,W) long indices"""
        if self.task == "binary":
            return (mask > 127).float().unsqueeze(0)
        return mask.long()

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """uint8 image tensor -> float in [0,1], optionally ImageNet-normalize"""
        image = image.float() / 255.0                     # (3,H,W)
        if self.cfg.normalize == "imagenet":
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return image


# --- builders ---
def build_dataset(cfg: DataConfig, split: str) -> Dataset:
    """Build the raw MedSegBench dataset for a split and wrap it.

    For train, an optional deterministic subset (cfg.n_train) supprots the
    data-efficiency experiment; val/test are left whole so runs stay comparable
    """
    cls = resolve_dataset_class(cfg.name)
    root = os.path.expanduser(cfg.root)
    os.makedirs(root, exist_ok=True)
    base = cls(
        split=split,
        transform=None,          # image handling happens in the wrapper
        target_transform=None,   # mask handling happens in the wrapper
        download=cfg.download,
        root=root,
        size=cfg.size,
        category=cfg.category,
    )
    ds: Dataset = MedSegWrapper(base, base.info, cfg, is_train=(split == "train"))

    if split == "train" and cfg.n_train is not None and cfg.n_train < len(ds):
        rng = np.random.default_rng(cfg.subset_seed)
        idx = rng.permutation(len(ds))[: cfg.n_train]
        ds = Subset(ds, idx.tolist())
    return ds


def build_dataloaders(cfg: DataConfig) -> Dict[str, DataLoader]:
    """train/val/test DataLoaders from one config"""
    g = torch.Generator().manual_seed(cfg.seed)
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = build_dataset(cfg, split)
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=(split == "train"),
            generator=g if split == "train" else None,
        )
    return loaders


def list_datasets() -> List[Tuple[str, str, int, int]]:
    """(class_name, task, img_channels, n_classes) for every dataset"""
    seen = {}
    for v in DATASET_REGISTRY.values():
        info = medsegbench.info.INFO[v.flag]
        seen[v.__name__] = (v.__name__, info["task"], info["n_channels_im"],
                            len(info["pixel_labels"]))
    return sorted(seen.values(), key=lambda r: (r[1], r[0]))
