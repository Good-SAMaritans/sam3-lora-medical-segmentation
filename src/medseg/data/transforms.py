"""
transforms.py
=============
Image augmentation. build_transforms() returns an albumentations pipeline that
transforms the image and its mask together, so they stay aligned....

Two presets:
  * "medseg_default" (training) — flips, small rotations/shifts, intensity
    tweaks (brightness/contrast, gamma, CLAHE), and elastic deformation....
    ... we can edit this medseg_default or use create new pipelines other whatever name
  * "none" — resize + pad + to-tensor only

Validation/test always get resize + pad + to-tensor only (no randomness), so
evaluation is deterministic
"""

from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _resize_and_pad(img_size: int):
    """Resize longest side to img_size, then pad to a square (keeps aspect ratio)"""
    return [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    ]


def build_transforms(img_size: int, is_train: bool,
                     preset: str = "medseg_default") -> A.Compose:
    """Albumentations Compose applied jointly to image and mask

    Augmentation runs only when is_train; val/test get resize+pad+tensor only,
    so evaluation is deterministic. Output after ToTensorV2 is a uint8 image
    tensor (C,H,W) — normalized downstream — and a uint8 mask (H,W)
    """
    tail = _resize_and_pad(img_size) + [ToTensorV2()]

    if not is_train or preset == "none":
        return A.Compose(tail)

    if preset == "medseg_default":
        aug = [
            # geometric (image + mask)
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05),
                     rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT,
                     fill=0, fill_mask=0, p=0.5),
            A.ElasticTransform(alpha=30, sigma=5, p=0.25),
            # intensity (image only)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.15),
        ]
        return A.Compose(aug + tail)

    raise ValueError(f"Unknown augmentation preset: {preset!r} "
                     f"(expected 'medseg_default' or 'none').")
