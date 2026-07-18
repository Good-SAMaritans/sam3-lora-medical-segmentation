"""
metrics.py - evaluation metrics.

dice_score: region overlap (DSC). nsd: Normalized Surface Distance (boundary agreement at a tolerance in pixels).
Inputs are binary masks (B,1,H,W) with values in {0,1} (predictions already thresholded or as probabilities).
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()

@torch.no_grad()
def _boundary(mask: torch.Tensor) -> torch.Tensor:
    # mask (B,1,H,W) {0,1}; boundary = mask minus its eroded version (via min-pool)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (mask - eroded).clamp(0, 1)

@torch.no_grad()
def nsd(pred: torch.Tensor, target: torch.Tensor, tol: int = 2, eps: float = 1e-6) -> float:
    # Normalized Surface Distance at tolerance `tol` pixels.
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    pb = _boundary(pred)
    tb = _boundary(target)
    # dilate each boundary by tol (via max-pool) to form tolerance bands
    k = 2 * tol + 1
    pb_d = F.max_pool2d(pb, k, stride=1, padding=tol)
    tb_d = F.max_pool2d(tb, k, stride=1, padding=tol)
    dims = (1, 2, 3)
    # fraction of each boundary within tol of the other
    pred_ok = (pb * tb_d).sum(dims)
    tgt_ok = (tb * pb_d).sum(dims)
    denom = pb.sum(dims) + tb.sum(dims)
    score = (pred_ok + tgt_ok + eps) / (denom + eps)
    return score.mean().item()