import torch
from .lora import LoRALayer, LoRA_qkv

def inject_lora_into_sam(sam, r=4, alpha=4):
    """
    Freeze all of SAM, then wrap each encoder block's qkv 
    with LoRA on Q & V.

    Only the LoRA adapters stay trainable.
    
    """
    #1. freeze everything
    for p in sam.parameters():
        p.requires_grad_(False)
    
    #2. wrap each transfomer block's qkv
    for blk in sam.image_encoder.blocks:
        qkv = blk.attn.qkv #the pretrained fused linear
        dim = qkv.in_features #768 for ViT-B
        lora_q = LoRALayer(dim, dim, r, alpha)
        lora_v = LoRALayer(dim, dim, r, alpha)

        #swap in the wrapper
        blk.attn.qkv = LoRA_qkv(qkv, lora_q, lora_v) 

    return sam