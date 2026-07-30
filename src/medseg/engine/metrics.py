"""
metrics.py - evaluation metrics.

dice_score: region overlap (DSC). nsd: Normalized Surface Distance (boundary agreement at a tolerance in pixels)
Inputs are binary masks (B,1,H,W) with values in {0,1} (predictions already thresholded or as probabilities)
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

def resize_mask(mask: torch.Tensor, size) -> torch.Tensor:
    """Nearest-resize a target to `size`, preserving its layout.

    Binary masks arrive as (B,1,H,W) float, multi-class ones as (B,H,W) long
    """
    if mask.ndim == 3:
        return F.interpolate(mask[:, None].float(), size=size, mode="nearest")[:, 0].long()
    return F.interpolate(mask, size=size, mode="nearest")

@torch.no_grad()
def multiclass_scores(logits: torch.Tensor, target: torch.Tensor, tol: int = 2,
                      ignore_index: int | None = None) -> tuple[float, float]:
    """Macro-averaged (Dice, NSD) over foreground classes, one-vs-rest on the argmax...

    Basicallym, each class is scored only on the images that actually contain it. Most MoNuSAC
    images hold two or three of the five cell types, and scoring an absent class
    would hand back a free 1.0 for predicting nothing.

    ignore_index marks a "don't care" label (e.g. MoNuSAC's Ambiguous class, which
    has no training examples). Pixels of that class are dropped from the prediction
    so they count as neither hit nor false positive for any real class
    """
    pred = logits.argmax(1)
    valid = None if ignore_index is None else (target != ignore_index)
    dices, nsds = [], []
    for c in range(1, logits.shape[1]):
        t = (target == c).float().unsqueeze(1)
        keep = t.flatten(1).sum(1) > 0
        if not keep.any():
            continue
        pc = (pred == c) if valid is None else ((pred == c) & valid)
        p = pc.float().unsqueeze(1)[keep]
        dices.append(dice_score(p, t[keep]))
        nsds.append(nsd(p, t[keep], tol))
    if not dices:
        return float("nan"), float("nan")
    return sum(dices) / len(dices), sum(nsds) / len(nsds)

def mean_score(values: list[float]) -> float:
    """Mean over scored batches, dropping the NaNs from batches with no foreground."""
    scored = [v for v in values if v == v]
    return sum(scored) / max(1, len(scored))

@torch.no_grad()
def seg_scores(logits: torch.Tensor, target: torch.Tensor, tol: int = 2,
               ignore_index: int | None = None) -> tuple[float, float]:
    """(Dice, NSD) for either task, dispatched on the logit channel count."""
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        return dice_score(probs, target), nsd(probs, target, tol)
    return multiclass_scores(logits, target, tol, ignore_index)