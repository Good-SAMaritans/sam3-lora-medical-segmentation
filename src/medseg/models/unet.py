"""
unet.py - U-Net baseline via segmentation_models_pytorch
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", encoder_weights: str | None = "imagenet",
                 in_channels: int = 3, num_classes: int = 1, finetune_mode: str = "full"):
        super().__init__()
        self.finetune_mode = finetune_mode
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights= encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
        self._apply_finetune_mode(finetune_mode)

    @property
    def backbone(self) -> nn.Module:
        return self.net.encoder

    @property
    def head(self) -> nn.Module:
        return nn.ModuleList([self.net.decoder, self.net.segmentation_head])

    def _apply_finetune_mode(self, mode: str):
        if mode == "full":
            for p in self.parameters():
                p.requires_grad = True
        elif mode == "decoder_only":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.head.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"unknown finetune_mode: {mode}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
