import torch
from medseg.models.lora import LoRALayer, LinearWithLoRA, LoRA_qkv
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


def test_lora_qkv_slices():
    # fused attention like SAM3 vitdet: one Linear producing [ Q | K | V ]
    dim = 16
    qkv = nn.Linear(dim, 3 * dim)
    wrapped = LoRA_qkv(qkv,
                       LoRALayer(dim, dim, r=4, alpha=4),
                       LoRALayer(dim, dim, r=4, alpha=4))

    # 1. adapters off (B=0): perfect drop-in, bit-for-bit — on the usual
    # (batch, tokens, dim) AND the vitdet 4D (batch, H, W, dim) input
    for x in (torch.randn(2, 5, dim), torch.randn(2, 3, 3, dim)):
        assert torch.equal(wrapped(x), qkv(x))

    x = torch.randn(2, 3, 3, dim)          # keep the 4D case for the slice checks
    base = qkv(x)

    # 2. turn on ONLY lora_q: Q third moves; K and V thirds must be
    # bit-identical (torch.equal, not allclose — K is supposed to BE the
    # original numbers, not merely close to them)
    nn.init.normal_(wrapped.lora_q.B.weight)
    out = wrapped(x)
    assert not torch.equal(out[..., :dim], base[..., :dim])
    assert torch.equal(out[..., dim:2*dim], base[..., dim:2*dim])
    assert torch.equal(out[..., 2*dim:], base[..., 2*dim:])

    # 3. turn on lora_v too: V third moves, K STILL untouched
    nn.init.normal_(wrapped.lora_v.B.weight)
    out = wrapped(x)
    assert not torch.equal(out[..., 2*dim:], base[..., 2*dim:])
    assert torch.equal(out[..., dim:2*dim], base[..., dim:2*dim])