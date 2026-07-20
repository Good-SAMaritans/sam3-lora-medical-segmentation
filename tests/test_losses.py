import torch
from medseg.engine.losses import BceDiceLoss

def test_perfect_prediction_low_loss():
    loss_fn = BceDiceLoss()
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0
    logits = torch.where(target > 0.5, 10.0, -10.0) # near perfect
    loss = loss_fn(logits, target)
    assert loss.item() < 0.05, loss.item()

def test_wrong_prediction_higher_loss():
    loss_fn = BceDiceLoss()
    target = torch.zeros(1, 1, 16, 16); target[:, :, 4:8, 4:8] = 1.0
    good = loss_fn(torch.where(target > 0.5, 10.0, -10.0), target)
    bad = loss_fn(torch.where(target > 0.5, -10.0, 10.0), target)
    assert bad.item() > good.item()