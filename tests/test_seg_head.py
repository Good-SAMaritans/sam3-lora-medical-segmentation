import torch
from medseg.models.seg_head import FpnSegHead

def test_fpn_seg_head_output_shape():
    head = FpnSegHead(in_channels=256, num_classes=1, out_size=1008)
    feats = [
        torch.randn(2, 256, 288, 288),
        torch.randn(2, 256, 144, 144),
        torch.randn(2, 256, 72, 72),
    ]
    out = head(feats)
    assert out.shape == (2, 1, 1008, 1008), out.shape
    assert torch.isfinite(out).all()