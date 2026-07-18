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