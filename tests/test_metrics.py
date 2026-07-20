import torch
from medseg.engine.metrics import dice_score, nsd

def test_dice_identical_is_one():
    m = torch.zeros(1, 1, 32, 32); m[:, :, 8:20, 8:20] = 1.0
    assert dice_score(m, m) > 0.99

def test_dice_disjoint_is_zero():
    a = torch.zeros(1, 1, 32, 32); a[:, :, 0:8, 0:8] = 1.0
    b = torch.zeros(1, 1, 32, 32); b[:, :, 20:28, 20:28] = 1.0
    assert dice_score(a, b) < 0.01

def test_nsd_identical_is_one():
    m = torch.zeros(1, 1, 32, 32); m[:, :, 8:20, 8:20] = 1.0
    assert nsd(m, m, tol=2) > 0.99

def test_nsd_shifted_less_than_one():
    a = torch.zeros(1, 1, 32, 32); a[:, :, 8:20, 8:20] = 1.0
    b = torch.zeros(1, 1, 32, 32); b[:, :, 8:20, 18:30] = 1.0  # shifted far
    assert nsd(a, b, tol=2) < 0.9