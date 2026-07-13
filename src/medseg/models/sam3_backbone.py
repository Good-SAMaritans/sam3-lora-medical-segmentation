"""
sam3_backbone.py - SAM 3 vision backbone wrapper.

Loads SAM 3 via build_sam3_image_model(), keeps only the vision path (backbone.forward_image), 
and returns the 3-level FPN as an ordered fine->coarse list. The language_backbone is deleted. 
Must run under bf16 autocast (SAM 3 requirement) handled by the caller (Sam3MedSeg).

FINE-TUNING FIX (see below): SAM 3's ViT MLP hardcodes an inference-only fused op (sam3.perflib.fused.addmm_act) 
that raises when grad is enabled and detaches weights. It CANNOT be trained through.
Patch it with a differentiable Linear+activation equivalent so backprop reaches fc1. 
Affects ANY arm that fine-tunes the backbone (full / LoRA / decoder only if unfrozen).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Differentiable replacement for SAM3's inference-only fused MLP op
def _differentiable_addmm_act(activation, linear, mat1):
    """Drop-in for sam3.perflib.fused.addmm_act that supports autograd.

    Original detaches weights + refuses grad (inference-only). 
    This uses the SAME weights via F.linear so gradients flow, then applies the activation.
    """
    x = F.linear(mat1, linear.weight, linear.bias)
    if activation in (torch.nn.functional.relu, torch.nn.ReLU):
        return F.relu(x)
    if activation in (torch.nn.functional.gelu, torch.nn.GELU):
        return F.gelu(x)
    raise ValueError(f"Unexpected activation {activation}")

def _patch_sam3_for_training():
    """Replace the fused inference-only MLP op in vitdet with a grad-friendly one."""
    import sam3.model.vitdet as vitdet
    vitdet.addmm_act = _differentiable_addmm_act

class Sam3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        _patch_sam3_for_training() # must run before/after load. patches the module-level name
        from sam3.model_builder import build_sam3_image_model
        full = build_sam3_image_model()
        self.backbone = full.backbone
        # Drop the language encoder. we only do image only dense segmentation.
        if hasattr(self.backbone, "language_backbone"):
            del self.backbone.language_backbone
        self.out_channels = 256

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        # image: (B,3,1008,1008) float. Autocast handled by caller/model.
        feats = self.backbone.forward_image(image)
        fpn = feats["backbone_fpn"] # [(B,256,288,288),(B,256,144,144),(B,256,72,72)]
        return list(fpn)