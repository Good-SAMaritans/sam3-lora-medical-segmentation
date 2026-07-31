"""
qualitative figure: input | ground truth | prediction
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ..data.medseg_data import IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def save_predictions(model, loader, out_path: str, n: int = 5,
                     device: str = "cuda", normalize: str | None = None):
    model.eval()
    imgs, gts, outs = [], [], []
    for batch in loader:
        outs.append(model(batch["image"].to(device)).cpu())
        imgs.append(batch["image"])
        gts.append(batch["mask"])
        if sum(len(x) for x in imgs) >= n:
            break

    img, gt, logits = torch.cat(imgs)[:n], torch.cat(gts)[:n], torch.cat(outs)[:n]

    # both tasks end up as (N,H,W) label imageonly the colour scale differs
    if logits.shape[1] == 1:
        gt, pred = gt[:, 0], (torch.sigmoid(logits)[:, 0] > 0.5).float()
        cmap, vmax = "gray", 1
    else:
        gt, pred = gt.float(), logits.argmax(1).float()
        cmap, vmax = plt.get_cmap("tab10", logits.shape[1]), logits.shape[1] - 1
    pred = F.interpolate(pred[:, None], size=gt.shape[-2:], mode="nearest")[:, 0]

    if normalize == "imagenet":
        img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(0, 2, 3, 1).numpy()

    n = len(img)
    fig, axes = plt.subplots(n, 3, figsize=(6.5, 2.2 * n), squeeze=False)
    for r in range(n):
        panels = [(img[r], None, 1), (gt[r], cmap, vmax), (pred[r], cmap, vmax)]
        for c, (data, cm, hi) in enumerate(panels):
            axes[r, c].imshow(data, cmap=cm, vmin=0, vmax=hi)
            axes[r, c].set_axis_off()
    for c, title in enumerate(["Image", "Ground truth", "Prediction"]):
        axes[0, c].set_title(title, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
