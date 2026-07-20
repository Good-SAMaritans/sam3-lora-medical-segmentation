"""
LoRALayer

A small nn.Module for one projection's low-rank update. 
No SAM yet, just the reusable piece.

Without LoRA
y = Wx

With LoRA
y = Wx + ΔW·x     where ΔW = B·A
  = Wx + (B·A)·x


It computes x->B(A(x)). (aplha/r)

"""

import torch.nn as nn
import torch


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r, alpha):
        """
        in_dim = d = 768 on full lora, r = 4
        in_dim = out_dim typically
        r = reduced dimension for LoRA
        aplha = scaling constant 
        
        """
        
        #get parent class inheritence 
        super().__init__()

        #A project d->r
        #ex. (768, 4)layer 
        # (A weight shape-> 4, 768)
        self.A = nn.Linear(in_dim, r, bias = False) 

        #B projects r->d
        #ex. layer shape(4, 768) 
        # (B weight shape-> 768, 4)
        self.B = nn.Linear(r, out_dim, bias = False) 

        #zero-init B: 
        nn.init.zeros_(self.B.weight)

        self.scaling = alpha / r
    
    def forward(self, x):
        #Apply through the bottle next B(A(x)) 
        #better than forming the whole matrix B@A a 768x 768
        # A(x): 768->4
        #B(...): 4->768
        return self.B(self.A(x))* self.scaling


class LinearWithLoRA(nn.Module):

    def __init__(self, original_linear, r, alpha):
       # parent constructor
        super().__init__()

        #store the pretrained Linear as a submodule
        self.original = original_linear

        #build the adaptor. 
        # LoRALayer(in_dim, out_dim, r, alpha) 
        self.lora = LoRALayer(self.original.in_features, self.original.out_features, r, alpha)

    def forward(self, x):
    #two parallel paths through the same x, summed:
    #- the frozen orginal : self.orignal(x)
    #- the low-rank update self.lora(x)
        base = self.original(x)
        lora = self.lora(x)

        return base + lora


#For FUSED-qkv attention (SAM 1 / native sam3 vitdet: one Linear(dim, 3*dim)).
#This is the wrapper for the TEAM stack (Sean's native-sam3 branch).
#Restored from commit ea56381 after the native-vs-HF discovery (2026-07-17).
class LoRA_qkv(nn.Module):
    def __init__(self, qkv, lora_q, lora_v):
        super().__init__()

        #original fused linear (frozen)
        # receive an exisitng layer
        self.qkv = qkv

        #two LoRALayers
        self.lora_q, self.lora_v = lora_q, lora_v

        #two LoRALayers
        # = 768; width of each Q/K/V
        self.dim = qkv.in_features

    def forward(self, x):
        #dim = 16 but qkv = nn.Linear(16,48) because 3*dim =48
        #(2,5,48) = (batch, tokens, dim)
        #the last axis of 48 is the three blocks glued together
        base = self.qkv(x) #(2,5,48)


        #base[...,:16] = Q (2,5,16)
        #(2,5,16) + (2,5,16) -> (2,5,16)
        q = base[...,:self.dim] + self.lora_q(x)

        #base[..., 16:32] = K (2,5,16)
        #(2,5,16) untouched
        k = base[..., self.dim: 2*self.dim]

        #base[..., 32:48] = V (2,5,16)
        #(2,5,16) + (2,5,16) -> (2,5,16)
        v = base[..., 2*self.dim:] + self.lora_v(x)

        #glue them back into one tensor along last axis
        #(2,5, 48)
        return torch.cat([q,k,v],dim= -1)


def inject_lora_into_sam3(model, r=4, alpha=4):
    """
    HF-transformers edition ONLY (separate q_proj/v_proj) — does NOT work on the
    native `sam3` package the team uses (fused qkv; see LoRA_qkv). Kept as the
    verified HF reference until the team stack is confirmed with Sean, then rewrite.

    Freeze all of SAM 3, then wrap each vision-encoder layer's
    q_proj and v_proj with LoRA. Only the adapters stay trainable.
    """
    # 1. freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # 2. wrap q_proj and v_proj in every vision-encoder layer
    for layer in model.vision_encoder.backbone.layers:
        attention = layer.attention
        attention.q_proj = LinearWithLoRA(attention.q_proj, r, alpha)
        attention.v_proj = LinearWithLoRA(attention.v_proj, r, alpha)

    return model