import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SAM3 needs a CUDA GPU")

def test_sam3_medseg_forward_shape():
    from medseg.models import Sam3MedSeg
    model = Sam3MedSeg(finetune_mode="full").cuda().eval()
    x = torch.rand(1, 3, 256, 256, device="cuda") # simulate a data-layer batch
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 1008, 1008), out.shape
    assert torch.isfinite(out).all()

def test_full_mode_all_trainable():
    from medseg.models import Sam3MedSeg
    model = Sam3MedSeg(finetune_mode="full")
    assert all(p.requires_grad for p in model.parameters())

def test_decoder_only_freezes_backbone():
    from medseg.models import Sam3MedSeg
    model = Sam3MedSeg(finetune_mode="decoder_only")
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())