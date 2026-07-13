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

import math
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
