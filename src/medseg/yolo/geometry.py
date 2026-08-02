"""
geometry.py - mask <-> polygon conversion shared by export and evaluate.

Both halves of the YOLO arm need the same two primitives:

  export.py    mask -> polygons          (writing COCO)
               polygons -> mask          (the reconstruction-fidelity check)
  evaluate.py  polygons -> mask          (rasterizing GT and YOLO predictions)
Polygon representation: everything here accepts EITHER the flat COCO form
[x1,y1,x2,y2,...] or an (N,2) array, since COCO stores the former and
Ultralytics' `masks.xy` yields the latter. Coordinates are always ABSOLUTE
pixels; normalization to [0,1] is Ultralytics' job during convert_coco.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from medseg.engine.metrics import dice_score

PolygonLike = Sequence[float] | np.ndarray


def as_xy(poly: PolygonLike) -> np.ndarray:
    """Any accepted polygon form -> (N,2) float array of pixel coordinates."""
    return np.asarray(poly, np.float64).reshape(-1, 2)


def mask_to_polygons(mask: np.ndarray,
                     min_area: float = 10.0,
                     eps_frac: float = 0.004) -> List[List[float]]:
    """Binary mask (H,W) -> list of flat COCO polygons [x1,y1,x2,y2,...].

    Accepts {0,1} or {0,255} masks. Disconnected components become separate
    polygons (i.e. separate COCO instances). Components below `min_area` px are
    dropped as speckle. `eps_frac` is the Douglas-Peucker tolerance as a
    fraction of each contour's perimeter: larger = fewer vertices = more
    boundary loss, which `reconstruction_dice` is there to catch.

    RETR_EXTERNAL ignores interior holes. Fine for polyps and lesions; verify
    before reusing on a modality where targets are annular.
    """
    m = (mask > 127) if mask.max() > 1 else (mask > 0)
    m = m.astype(np.uint8)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[float]] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:   # a polygon cant be less than 3 sides, duh
            continue
        polys.append(pts.astype(np.float64).flatten().tolist())
    return polys
def polygons_to_mask(polys: Iterable[PolygonLike],
                     height: int, width: int) -> np.ndarray:
    """Rasterize polygons into a binary {0,1} uint8 mask (H,W).
    The single rasterizer for the whole YOLO arm: ground truth, predictions,
    and the export reconstruction check all go through here, so they cannot
    disagree.
    """
    m = np.zeros((height, width), np.uint8)
    for poly in polys:
        pts = as_xy(poly)
        if pts.shape[0] < 3: #skips polygons with less than 3 points (they would be lines or points)
            continue
        cv2.fillPoly(m, [pts.round().astype(np.int32)], 1)
    return m
def poly_bbox_area(poly: PolygonLike) -> Tuple[List[float], float]:
    """COCO `bbox` [x,y,w,h] and `area` (shoelace) for one polygon."""
    pts = as_xy(poly)
    x, y = pts[:, 0], pts[:, 1]
    x0, y0, x1, y1 = x.min(), y.min(), x.max(), y.max()
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], float(area)
def to_bchw(mask: np.ndarray) -> torch.Tensor:
    """(H,W) mask -> (1,1,H,W) float CPU tensor, ready for engine.metrics."""
    return torch.from_numpy(np.ascontiguousarray(mask).astype(np.float32))[None, None]
def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    """just dice it up"""
    return dice_score(to_bchw(a), to_bchw(b))
