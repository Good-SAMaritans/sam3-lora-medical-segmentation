"""
evaluate.py - evaluate a trained YOLO-seg model with the PROJECT's own metrics.
used mostly in the yolo_train file, called at end

Usage (after training):
    from medseg.models.yolo.evaluate import evaluate_yolo
    evaluate_yolo("runs/segment/kvasir/weights/best.pt",
                  coco_root="data/yolo/kvasir/coco", split="test",
                  out_json="results/metrics/yolo_kvasir.json")
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional
import numpy as np
from medseg.engine.metrics import dice_score, nsd
from ultralytics import YOLO
from .geometry import polygons_to_mask, to_bchw

_rasterize_polys = polygons_to_mask #this is pulled later into test_yolo_eval for conciseness in that file
_to_bchw = to_bchw


def _load_gt_mask(image_id: int, height: int, width: int,
                  annot_index: Dict[int, list]) -> np.ndarray:
    polys = [poly
             for annotation in annot_index.get(image_id, [])
             for poly in annotation["segmentation"]]
    return polygons_to_mask(polys, height, width)


def _pred_mask_from_result(res, height: int, width: int) -> np.ndarray:
    """YOLO Results -> semantic foreground mask at ORIGINAL image resolution.

    Uses `masks.xy` (original-image pixel coords) rather than `masks.data`
    (letterboxed model space) -- see module docstring.
    """
    if getattr(res, "masks", None) is None or len(res.masks) == 0:
        return np.zeros((height, width), np.uint8)
    return polygons_to_mask(res.masks.xy, height, width)


def evaluate_yolo(weights: str,
                  coco_root: str,
                  split: str = "test",
                  imgsz: int = 640,
                  conf: float = 0.25,
                  tol: float = 2.0,
                  out_json: Optional[str] = None,
                  device: Optional[str] = None,
                  model=None) -> Dict[str, float]:
    """Run YOLO-seg over a split and score it with the project's Dice/NSD.
    Metrics accumulate PER IMAGE (not per batch), so partial batches cannot skew
    the mean
    """
    ann_path = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(
            f"missing COCO annotations: {ann_path}\n"
            f"run scripts/yolo_export.py first"
        )
    with open(ann_path) as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    if not images:
        raise ValueError(f"no images in {ann_path}")

    annot_index: Dict[int, list] = {}
    for annotation in coco["annotations"]:
        annot_index.setdefault(annotation["image_id"], []).append(annotation)

    img_dir = os.path.join(coco_root, "images", split)
    if model is None:
        model = YOLO(weights)

    per_image: List[Dict] = []
    n_both_empty = 0

    for im in sorted(images.values(), key=lambda d: d["id"]):
        path = os.path.join(img_dir, im["file_name"])
        H, W = int(im["height"]), int(im["width"])

        res = model.predict(path, imgsz=imgsz, conf=conf,
                            device=device, verbose=False)[0]

        pred_np = _pred_mask_from_result(res, H, W)
        gt_np = _load_gt_mask(im["id"], H, W, annot_index)

        pred_t, gt_t = to_bchw(pred_np), to_bchw(gt_np)
        both_empty = (pred_np.sum() == 0 and gt_np.sum() == 0)
        if both_empty:
            n_both_empty += 1

        per_image.append({
            "image_id": im["id"],
            "file_name": im["file_name"],
            "dice": dice_score(pred_t, gt_t),
            "nsd": nsd(pred_t, gt_t, tol=int(tol)),
            "gt_empty": bool(gt_np.sum() == 0),
            "pred_empty": bool(pred_np.sum() == 0),
        })

    def _mean(key, records):
        vals = [r[key] for r in records]
        return float(np.mean(vals)) if vals else float("nan")

    scored = [r for r in per_image if not (r["gt_empty"] and r["pred_empty"])]

    result = {
        "arm": "yolo_seg",
        "weights": weights,
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "tol": tol,
        "n_images": len(per_image),
        "n_both_empty": n_both_empty,
        "dice": _mean("dice", per_image),
        "nsd": _mean("nsd", per_image),
        # same metrics excluding free-1.0 background images
        "dice_excl_empty": _mean("dice", scored),
        "per_image": per_image,
    }

    if out_json:
        parent = os.path.dirname(out_json)
        if parent:                       # bare filename -> dirname "" -> skip
            os.makedirs(parent, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_json}")

    print(f"YOLO-seg {split}: dice={result['dice']:.4f}  "
          f"nsd={result['nsd']:.4f}  "
          f"(n={result['n_images']}, both_empty={n_both_empty})")
    return result
