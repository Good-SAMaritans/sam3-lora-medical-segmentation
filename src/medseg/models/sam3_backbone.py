"""
sam3_backbone.py - SAM 3 vision backbone wrapper.

Loads SAM 3 via build_sam3_image_model(), keeps only the vision path (backbone.forward_image), 
and returns the 3-level FPN as an ordered fine->coarse list. 
The language_backbone is deleted to free memory and params. 
Must run under bf16 autocast (SAM 3 requirement) handled by the caller (Sam3MedSeg).
"""
import torch
import torch.nn as nn

class Sam3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        from sam3.model_builder import build_sam3_image_model
        full = build_sam3_image_model()
        self.backbone = full.backbone
        # Drop the language encoder. We only do image-only dense segmentation.
        if hasattr(self.backbone, "language_backbone"):
            del self.backbone.language_backbone
        self.out_channels = 256

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        # image: (B,3,1008,1008) float. Autocast handled by caller/model.
        feats = self.backbone.forward_image(image)
        fpn = feats["backbone_fpn"] # [(B,256,288,288),(B,256,144,144),(B,256,72,72)]
        return list(fpn)