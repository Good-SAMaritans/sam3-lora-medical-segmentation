"""
losses.py - BCE + Dice loss (MedSAM recipe, unweighted sum).

Binary segmentation. Inputs are raw logits (B,1,H,W) and float targets {0,1} of the same shape.
CeDiceLoss is the multi-class counterpart; build_loss picks between them.
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

class CeDiceLoss(nn.Module):
    """Cross-entropy + soft Dice, logits (B,C,H,W) and long targets (B,H,W) of class indices...
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        probs = logits.softmax(1)
        onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)[1:]
        union = (probs.sum(dims) + onehot.sum(dims))[1:]
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return ce + (1 - dice.mean())

def build_loss(num_classes: int) -> nn.Module:
    return BceDiceLoss() if num_classes == 1 else CeDiceLoss()