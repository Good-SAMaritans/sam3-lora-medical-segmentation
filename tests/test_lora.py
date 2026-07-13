import torch
from medseg.models.lora import LoRALayer

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