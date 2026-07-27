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
from .lora import LoRALayer, LoRA_qkv

SAM3_SIZE = 1008
SAM3_MEAN = 0.5
SAM3_STD = 0.5

class Sam3MedSeg(nn.Module):
    def __init__(self, finetune_mode: str = "full", num_classes: int = 1, lora_r: int = 4, lora_alpha: int = 4):
        super().__init__()
        self.finetune_mode = finetune_mode
        self.backbone = Sam3Backbone()
        self.head = FpnSegHead(in_channels=self.backbone.out_channels, num_classes=num_classes, out_size=SAM3_SIZE)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
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
            # freeze ALL backbone params. inject LoRA into each fused qkv,
            # head says trainable 
            for p in self.backbone.parameters():
                p.requires_grad = False

            #2) wrap each vitdet block's fused qkv with LoRA
            for block in self.backbone.backbone.vision_backbone.trunk.blocks :
                qkv = block.attn.qkv #1024 , read it
                dim = qkv.in_features
                lora_q = LoRALayer( dim, dim, r=self.lora_r, alpha=self.lora_alpha)
                lora_v = LoRALayer(dim, dim,  r=self.lora_r, alpha=self.lora_alpha)
                block.attn.qkv = LoRA_qkv(qkv, lora_q, lora_v)
            
        else:
            raise ValueError(f"unknown finetune_mode: {mode}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # torch 2.10.0+cu128 has a bf16-autocast backward cuBLAS bug
        # (CUBLAS_STATUS_INVALID_VALUE) - verified with a 3-line repro. Train in fp32 (weights are fp32 anyway). 
        # Revisit bf16/AMP if the torch build is upgraded.
        image = self._prep_input(image)
        feats = self.backbone(image) # fp32 throughout
        return self.head(feats)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]