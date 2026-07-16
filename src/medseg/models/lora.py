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
