"""
seg_head.py — FPN segmentation decoder.

Turns SAM 3's 3-level feature pyramid into a dense mask. 
Input feature maps are all 256-channel at strides 4/7/14 of a 1008 input: 288, 144, 72. 
We fuse them top-down (coarse upsampled + added to finer), smooth with 3x3 convs, 
then upsample to the target size and project to 'num_classes' logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FpnSegHead(nn.Module):
    def __init__(self, in_channels: int = 256, mid_channels: int = 128, num_classes: int = 1, out_size: int = 1008):
        super().__init__()
        self.out_size = out_size
        # lateral 1x1 convs to a common mid width (one per FPN level)
        self.lat_fine = nn.Conv2d(in_channels, mid_channels, 1)    # 288
        self.lat_med = nn.Conv2d(in_channels, mid_channels, 1)     # 144
        self.lat_coarse = nn.Conv2d(in_channels, mid_channels, 1)  # 72
        # 3x3 smoothing after each fusion
        self.smooth_med = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.smooth_fine = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        # final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats ordered fine->coarse: [288, 144, 72]
        fine, med, coarse = feats
        c = self.lat_coarse(coarse) # (B,mid,72,72)
        m = self.lat_med(med) + F.interpolate(c, size=med.shape[-2:], mode="bilinear", align_corners=False)
        m = self.smooth_med(m) # (B,mid,144,144)
        f = self.lat_fine(fine) + F.interpolate(m, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        f = self.smooth_fine(f) # (B,mid,288,288)
        logits = self.classifier(f) # (B,num_classes,288,288)
        logits = F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return logits