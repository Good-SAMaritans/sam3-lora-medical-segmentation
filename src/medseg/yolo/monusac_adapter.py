"""
monusac_adapter.py - make a trained YOLO-seg model look like an nn.Module that
`scripts/eval_model.py:evaluate()` and we can reuse code.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from .geometry import polygons_to_mask

N_CLASSES = 5          # 0=background, 1..4 nucleus types (5=Ambiguous never predicted)
#ACKGROUND_LOGIT = 0.5  # any instance with conf > this wins the argmax over background
NEG, POS = -10.0, 10.0

class YoloSemanticAdapter(nn.Module):
    """Wraps a trained YOLO-seg checkpoint as a semantic-segmentation module.

    forward(x): (B,3,H,W) float image batch -> (B,5,H,W) pseudo-logits whose
    argmax is the predicted class index, at the SAME H,W as the input. Keeping
    the output at the input resolution means `evaluate()` scores at 512 with
    tol=2, matching the other arms.
    """

    def __init__(self, weights: str, conf: float = 0.25,
                 imgsz: int = 512, n_classes: int = N_CLASSES,
                 device: Optional[str] = None):
        super().__init__()
        

        self.yolo= {"model": YOLO(weights)}
        self.conf = conf
        self.imgsz = imgsz
        self.n_classes = n_classes
        self._device_hint = device
        # Gives .to(device) something real to move, so nn.Module bookkeeping and
        # `next(model.parameters()).device` behave normally.
        self.register_buffer("_anchor", torch.zeros(1))

    def train(self, mode: bool = True):
        """Adapter is inference-only; .eval() from the evaluator is a no-op."""
        self.training = False
        return self

    # -- the actual conversion --------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        dev = x.device
 
        results = self.yolo["model"].predict(
            x, imgsz=self.imgsz, conf=self.conf, verbose=False,
            device=0 if dev.type == "cuda" else "cpu",
        )
 
        if self.n_classes == 1:                  # binary: one foreground channel
            logits = torch.full((B, 1, H, W), NEG, device=dev, dtype=torch.float32)
        else:                                    # multi-class: ch0 = background
            logits = torch.full((B, self.n_classes, H, W), NEG,
                                device=dev, dtype=torch.float32)
            logits[:, 0] = 0.0                   # background wins unless an
                                                 # instance channel goes positive
 
        for b, res in enumerate(results):
            masks = getattr(res, "masks", None)
            if masks is None or len(masks) == 0:
                continue                          # all background: valid prediction
 
            # masks.xy is in ORIGINAL-image pixel coords (un-letterboxed by
            # ultralytics), i.e. the H,W we were handed. Never use masks.data,
            # which is still in the padded model space.
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
 
            for poly, cls_id, conf in zip(masks.xy, cls_ids, confs):
                # binary: single channel 0. multi-class: YOLO 0..3 -> channel 1..4
                ch = 0 if self.n_classes == 1 else int(cls_id) + 1
                if not (0 <= ch < self.n_classes):
                    continue                      # defensive: unexpected class id
                inst = polygons_to_mask([poly], H, W)
                if inst.sum() == 0:
                    continue
                inst_t = torch.from_numpy(inst.astype(np.float32)).to(dev)
                # max-accumulate: where instances of different classes overlap,
                # the more confident one wins the argmax.
                # POS scaled by confidence: overlapping instances of different
                # classes resolve to the more confident one under argmax.
                val = torch.where(inst_t > 0, POS * float(conf),
                                  torch.full_like(inst_t, NEG))
                logits[b, ch] = torch.maximum(logits[b, ch], val)
 
        return logits