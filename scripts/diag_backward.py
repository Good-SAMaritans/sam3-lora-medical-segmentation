"""Get the FULL backward error + enable expandable segments to rule out fragmentation."""
import sys; sys.path.insert(0, "src")
import os, traceback
import torch
from medseg.models import Sam3MedSeg

print("torch:", torch.__version__, "| device:", torch.cuda.get_device_name(0))

model = Sam3MedSeg(finetune_mode="full").cuda().train()
x = torch.rand(1, 3, 256, 256, device="cuda")
target = torch.rand(1, 1, 1008, 1008, device="cuda")

# Test 1: forward only (does forward+grad graph build OK?)
try:
    out = model(x)
    torch.cuda.synchronize()
    print("forward OK:", tuple(out.shape), out.dtype, "requires_grad:", out.requires_grad)
except Exception:
    print("FORWARD FAILED:"); traceback.print_exc(); sys.exit()

# Test 2: backward. print the FULL error
try:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), target)
    loss.backward()
    torch.cuda.synchronize()
    print("BACKWARD OK  loss=", loss.item())
except Exception:
    print("BACKWARD FAILED (full trace):")
    traceback.print_exc()