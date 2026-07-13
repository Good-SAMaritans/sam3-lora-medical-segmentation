"""
sam3_medseg.py - SAM 3 medical segmentation model.

Pipeline: (B,3,256,256) in [0,1]  -> resize 1008 + normalize(0.5)  ->
SAM 3 vision backbone (bf16 autocast) -> FpnSegHead -> (B,1,1008,1008) logits.

The input adapter (resize + normalize) lives HERE on purpose, 
so this arm does not depend on the shared data layer being changed to 1008 yet.
When the team standardizes the data pipeline to 1008, drop '_prep_input'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam3_backbone import Sam3Backbone
from .seg_head import FpnSegHead

SAM3_SIZE = 1008
SAM3_MEAN = 0.5
SAM3_STD = 0.5

class Sam3MedSeg(nn.Module):
    def __init__(self, finetune_mode: str = "full", num_classes: int = 1):
        super().__init__()
        self.finetune_mode = finetune_mode
        self.backbone = Sam3Backbone()
        self.head = FpnSegHead(in_channels=self.backbone.out_channels, num_classes=num_classes, out_size=SAM3_SIZE)
        self._apply_finetune_mode(finetune_mode)

    # --- input adapter (decoupled from shared data layer) ---
    def _prep_input(self, image: torch.Tensor) -> torch.Tensor:
        # image: (B,3,H,W) in [0,1]. Resize to 1008, normalize with mean/std 0.5.
        if image.shape[-2:] != (SAM3_SIZE, SAM3_SIZE):
            image = F.interpolate(image, size=(SAM3_SIZE, SAM3_SIZE), mode="bilinear", align_corners=False)
        return (image - SAM3_MEAN) / SAM3_STD

    # finetune strategy dispatch
    def _apply_finetune_mode(self, mode: str):
        if mode == "full":
            for p in self.parameters():
                p.requires_grad = True
        elif mode == "decoder_only":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.head.parameters():
                p.requires_grad = True
        elif mode == "lora":
            # Owned by Nik. Freeze backbone; LoRA adapters injected + head trained.
            raise NotImplementedError(
                "lora mode: Nik's arm. See spec section 3. Plug LoRA injection here.")
        else:
            raise ValueError(f"unknown finetune_mode: {mode}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self._prep_input(image)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = self.backbone(image)
        feats = [f.float() for f in feats] # head runs in fp32 for stable logits
        return self.head(feats) # (B,num_classes,1008,1008)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]