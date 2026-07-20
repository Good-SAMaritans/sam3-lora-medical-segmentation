"""
losses.py - BCE + Dice loss (MedSAM recipe, unweighted sum).

Binary segmentation. Inputs are raw logits (B,1,H,W) and float targets {0,1} of the same shape.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BceDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * target).sum(dims)
        union = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return bce + dice_loss