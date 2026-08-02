from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import inspect
import medsegbench
from medsegbench.dataset import MedSegBench as _Base
import numpy as np

from .geometry import (
    mask_to_polygons,
    polygons_to_mask,
    poly_bbox_area,
    mask_dice,
)

# Backwards-compatible aliases: These names are kept so existing
# imports and tests continue to work.
_poly_bbox_area = poly_bbox_area
_dice = mask_dice


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class ExportConfig:
    name: str                                   # medsegbench key, e.g. "kvasir"
    out_root: str                               # where to write the COCO dataset
    size: int = 512                             # source resolution: 128|256|512
    root: str = "~/.medsegbench"                # .npz cache
    download: bool = True
    category: Optional[str] = None              # BUSI: None|"C1"|"C2"
    splits: Tuple[str, ...] = ("train", "val", "test")
    class_name: str = "lesion"                  # single foreground class label
    min_area: float = 5.0
    eps_frac: float = 0.004
    #for MoNuSAC implementation
    class_values: Optional[List[int]] = None
    class_names: Optional[List[str]] = None

def _categories(cfg: "ExportConfig") -> List[Dict]:
    #if we are dealing with multi-class, treats diferently
    if cfg.class_values is None:
        return [{"id": 1, "name": cfg.class_name, "supercategory": "medical"}]
    names = cfg.class_names or [f"class{v}" for v in cfg.class_values]
    return [{"id": i, "name": n, "supercategory": "medical"}
            for i, n in enumerate(names, start=1)]
def export_split(cfg: ExportConfig, split: str) -> Dict[str, float]:
    """Export one split to images/{split}/ + instances_{split}.json.

    Returns summary stats including the mean reconstruction Dice for the split.
    """
    cls = _resolve_class(cfg.name)
    base = cls(
        split=split,
        transform=None,
        target_transform=None,
        download=cfg.download,
        root=os.path.expanduser(cfg.root),
        size=cfg.size,
        category=cfg.category,
    )

    img_dir = os.path.join(cfg.out_root, "images", split)
    ann_dir = os.path.join(cfg.out_root, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    coco = {
        "info": {"description": f"MedSegBench:{cfg.name} split={split}"},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": cfg.class_name, "supercategory": "medical"}],
    }
    #instantiation
    annot_id = 1
    recon_dices: List[float] = []
    n_empty = 0

    for i in range(len(base)):
        img_pil, target = base[i]
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")           # match force_rgb convention
        W, H = img_pil.size

        stem = f"{cfg.name}_{split}_{i:05d}"
        fname = f"{stem}.png"
        img_pil.save(os.path.join(img_dir, fname))     # PNG = lossless

        coco["images"].append(
            {"id": i, "file_name": fname, "width": W, "height": H}
        )

        mask = np.asarray(target)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if cfg.class_values is None:
            #binary implementation
            polys = mask_to_polygons(mask, cfg.min_area, cfg.eps_frac)
            polys_with_cat = [(p, 1) for p in polys]
        else: #treatment of multiclass
            polys, polys_with_cat = [], []
            for cat_id, value in enumerate(cfg.class_values, start=1):
                binary = (mask == value).astype(np.uint8)
                if not binary.any():
                    continue
                cls_polys = mask_to_polygons(binary, cfg.min_area, cfg.eps_frac)
                polys.extend(cls_polys)
                polys_with_cat.extend((p, cat_id) for p in cls_polys)

        if not polys:
            n_empty += 1                               # valid: background image
        else:
            recon = polygons_to_mask(polys, H, W)
            if cfg.class_values is None: #treatment of binary vs multi-class
                target = mask
            else:
                target = np.isin(mask, cfg.class_values).astype(np.uint8)
            recon_dices.append(mask_dice(mask, recon))
            for poly, cat_id in polys_with_cat:
                bbox, area = poly_bbox_area(poly)
                coco["annotations"].append({
                    "id": annot_id,
                    "image_id": i,
                    "category_id": cat_id,
                    "segmentation": [poly],            # COCO polygon: list of lists
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                annot_id += 1

    with open(os.path.join(ann_dir, f"instances_{split}.json"), "w") as f:
        json.dump(coco, f)

    return {
        "split": split,
        "n_images": len(base),
        "n_instances": annot_id - 1,
        "n_empty": n_empty,
        "recon_dice_mean": float(np.mean(recon_dices)) if recon_dices else 1.0,
        "recon_dice_min": float(np.min(recon_dices)) if recon_dices else 1.0,
    }


def export_dataset(cfg: ExportConfig) -> List[Dict[str, float]]:
    """Export all splits. Prints a per-split summary; the reconstruction Dice
    is the number to sanity-check before trusting any downstream YOLO result."""
    os.makedirs(cfg.out_root, exist_ok=True)
    stats = [export_split(cfg, s) for s in cfg.splits]
    print(f"\n=== COCO export: {cfg.name} -> {cfg.out_root} ===")
    for s in stats:
        flag = "" if s["recon_dice_mean"] > 0.97 else "  <-- CHECK eps_frac"
        print(f"  {s['split']:5s}  imgs={s['n_images']:4d}  "
              f"inst={s['n_instances']:4d}  empty={s['n_empty']:3d}  "
              f"recon_dice mean={s['recon_dice_mean']:.4f} "
              f"min={s['recon_dice_min']:.4f}{flag}")
    #itll throw a flag especially for monusac. it just doesn't flow great with it.
    return stats


def _resolve_class(name: str):

    key = name.lower().replace(" ", "").replace("-", "")
    for cname, cls in inspect.getmembers(medsegbench, inspect.isclass):
        if not (issubclass(cls, _Base) and cls is not _Base):
            continue
        keys = {cname.lower(), cls.flag.lower()}
        if cname.lower().endswith("msbench"):
            keys.add(cname.lower().removesuffix("msbench"))
        if key in keys or name.lower() in keys:
            return cls
    raise KeyError(f"Unknown dataset '{name}'")
