import torch
from medseg.models.lora import LoRALayer, LinearWithLoRA
import torch.nn as nn

def test_output_shape():
    layer = LoRALayer(in_dim=16, out_dim=16, r=4, alpha=4)
    y = layer(torch.rand(2,16))
    assert y.shape == (2,16)

def test_zero_at_init():
    layer = LoRALayer(in_dim=16, out_dim=16, r = 4, alpha=4)
    y = layer(torch.randn(2,16))
    assert torch.allclose(y , torch.zeros_like(y))

def test_param_count():
    layer = LoRALayer(in_dim=16, out_dim=16, r=4, alpha=4)
    # .numel() Returns the total number of elements in the input tensor.
    n = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    #number of parameters = 2*r*d
    assert n == 2*4*16


def test_linear_with_lora():
    original = nn.Linear(64, 64)  # bias=True by default, like SAM 3
    wrapped = LinearWithLoRA(original, r=4, alpha=4)
    x = torch.randn(2, 5, 64)      # (batch, tokens, dim)

    # 1. exact equality at init — why does this hold EXACTLY? 
    assert torch.equal(wrapped(x), original(x))

    # 2. randomize B, assert the outputs now DIFFER
    # (you did nn.init.normal_ on a B weight in your LoRALayer tests)
    nn.init.normal_(wrapped.lora.B.weight)
    assert not torch.equal(wrapped(x), original(x))

    # 3. shape preserved
    assert wrapped(x).shape == original(x).shape