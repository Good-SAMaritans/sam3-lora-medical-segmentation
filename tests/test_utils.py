import torch
from medseg.utils.device import get_device

def test_get_device():
    device = get_device()
    assert device in ("cuda", "mps", "cpu")
    #the device actually accept tensors
    t = torch.ones(1).to(device)
    assert t.device.type == device