"""eval_model.py - precision, recall, F1, IoU, Dice and NSD 
    import sys; sys.path.insert(0, "scripts")
    from eval_model import evaluate

    model = ...                       # your nn.Module, weights already loaded
    evaluate(model, "busi")           # prints val + test, returns the numbers
c = 1 for busi & kvasir, c = 5 for monusac (# classe)
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

from medseg.data.medseg_data import DataConfig, build_dataloaders
from medseg.utils.device import get_device

EPS = 1e-6
METRICS = ("precision", "recall", "f1", "iou", "dice", "nsd")


def default_tol(eval_size):
    """NSD tolerance in px, anchored at 2 px for a 512 raster so that a model
    scored at any resolution forgives the same real boundary displacement."""
    return max(1, round(2 * eval_size / 512))


# --- metrics -----------------------------------------------------------------

def _boundary(mask):
    """1-px inner boundary of a binary mask, via a 3x3 erosion."""
    eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (mask - eroded).clamp(0, 1)


def _nsd(pred, target, tol):
    """(B,) fraction of boundary pixels lying within `tol` px of the other boundary."""
    pb, tb = _boundary(pred), _boundary(target)
    k = 2 * tol + 1
    pb_dilated = F.max_pool2d(pb, k, stride=1, padding=tol)
    tb_dilated = F.max_pool2d(tb, k, stride=1, padding=tol)
    dims = (1, 2, 3)
    hits = (pb * tb_dilated).sum(dims) + (tb * pb_dilated).sum(dims)
    return (hits + EPS) / (pb.sum(dims) + tb.sum(dims) + EPS)


def score_binary(pred, target, tol):
    """pred/target: (B,1,H,W) in {0,1}. -> dict of (B,) per-image tensors."""
    dims = (1, 2, 3)
    tp = (pred * target).sum(dims)
    fp = (pred * (1 - target)).sum(dims)
    fn = ((1 - pred) * target).sum(dims)
    f1 = 2 * tp / (2 * tp + fp + fn + EPS)
    return {
        "precision": tp / (tp + fp + EPS),
        "recall": tp / (tp + fn + EPS),
        "f1": f1,
        "iou": tp / (tp + fp + fn + EPS),
        "dice": f1,
        "nsd": _nsd(pred, target, tol),
    }


def score_multiclass(pred, target, num_classes, tol, ignore_index):
    """pred/target: (B,H,W) long. -> dict of per-image lists, averaged over
    the foreground classes actually present in each image"""

    valid = None if ignore_index is None else (target != ignore_index)
    out = {m: [] for m in METRICS}
    for b in range(target.shape[0]):
        per_class = {m: [] for m in METRICS}
        for c in range(1, num_classes):
            t = (target[b] == c)
            if not t.any():
                continue
            p = (pred[b] == c)
            if valid is not None:
                p = p & valid[b]              # don't-care pixels are never a hit or a false positive
            s = score_binary(p[None, None].float(), t[None, None].float(), tol)
            for m in METRICS:
                per_class[m].append(s[m].item())
        if not per_class["iou"]:
            continue                          # background-only image
        for m in METRICS:
            out[m].append(sum(per_class[m]) / len(per_class[m]))
    return out


# --- evaluation --------------------------------------------------------------

def _match(x, size):
    """Nearest-resize labels to a square `size` (no-op if already there)."""
    if x.shape[-2:] == (size, size):
        return x
    return F.interpolate(x, size=(size, size), mode="nearest")


@torch.no_grad()
def _run_split(model, loader, tol, ignore_index, device):
    acc = {m: [] for m in METRICS}
    eval_size = None
    for batch in loader:
        logits = model(batch["image"].to(device))
        mask = batch["mask"].to(device)

        if eval_size is None:                 # the model's own output resolution sets the scale
            eval_size = logits.shape[-1]
            tol = default_tol(eval_size) if tol is None else tol

        if logits.shape[1] == 1:              # binary: (B,1,H,W) float mask binary treatment
            pred = _match((torch.sigmoid(logits) > 0.5).float(), eval_size)
            target = _match(mask, eval_size)
            scores = {m: v.tolist() for m, v in score_binary(pred, target, tol).items()}
        else:                                 # multi-class: (B,H,W) long mask
            pred = _match(logits.argmax(1, keepdim=True).float(), eval_size)
            target = _match(mask[:, None].float(), eval_size)
            scores = score_multiclass(pred[:, 0].long(), target[:, 0].long(),
                                      logits.shape[1], tol, ignore_index)
        for m in METRICS:
            acc[m] += scores[m]

    n = len(acc["iou"])
    if n == 0:
        raise RuntimeError("no image was scored -- every mask was background-only")
    return {m: sum(v) / n for m, v in acc.items()}, n, eval_size, tol


def evaluate(model, dataset, splits=("val", "test"), tol=None, ignore_index="auto",
             size=512, normalize="imagenet", batch_size=4, num_workers=4,
             download=False, csv_path=None, device=None, verbose=True):
    """Score `model` on `dataset` and return {split: {metric: value}}.

    dataset      "busi", "kvasir", or "monusac"
    tol          NSD tolerance (should be 2)
    ignore_index "auto" -> 5 on MoNuSAC (Ambiguous), None elsewhere
    size         resolution the images are loaded at; must match training
    normalize    "imagenet" or None; must match training # double check
    csv_path     optional file for csv
    """

    if ignore_index == "auto":
        ignore_index = 5 if dataset.lower() == "monusac" else None
    device = device or get_device() # if you are using cuda, mps, or cpu...
    model = model.to(device).eval()

    loaders = build_dataloaders(DataConfig(
        name=dataset, size=size, img_size=size, normalize=normalize,
        download=download, batch_size=batch_size, num_workers=num_workers,
    ))

    results, rows = {}, []
    for split in splits:
        scores, n, eval_size, used_tol = _run_split(model, loaders[split], tol, ignore_index, device)
        results[split] = scores
        rows.append(dict(dataset=dataset, split=split, images=n, eval_size=eval_size,
                         tol=used_tol, **{m: round(scores[m], 4) for m in METRICS}))

    if verbose:
        print(f"\ndataset={dataset}  scored_at={rows[0]['eval_size']}  tol={rows[0]['tol']}  "
              f"ignore_index={ignore_index}  normalize={normalize}  device={device}")
        header = f"{'split':<6}{'images':>8}" + "".join(f"{m:>11}" for m in METRICS)
        print(header + "\n" + "-" * len(header))
        for row in rows:
            print(f"{row['split']:<6}{row['images']:>8}" + "".join(f"{row[m]:>11.4f}" for m in METRICS))
        print()

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        if verbose:
            print(f"wrote {csv_path}")

    return results
