"""
trainer.py - training loop for the SAM 3 full-finetune arm.

bf16 autocast, AdamW with discriminative LR (backbone lr < head lr), BCE+Dice loss, 
per-epoch validation (Dice + NSD), best-Dice checkpointing to scratch.
This loop is Sean's; other arms may share losses/metrics but can write their own driver.
"""
import os
import torch
from torch.optim import AdamW

from .losses import build_loss
from .metrics import mean_score, resize_mask, seg_scores

class Trainer:
    def __init__(self, model, loaders: dict, ckpt_dir: str,
                 backbone_lr: float = 1e-5, head_lr: float = 1e-4,
                 weight_decay: float = 0.01, device: str = "cuda",
                 num_classes: int = 1):
        self.model = model.to(device)
        self.loaders = loaders
        self.device = device
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.loss_fn = build_loss(num_classes)
        # discriminative LR: separate param groups for backbone vs head
        bb = [p for p in self.model.backbone.parameters() if p.requires_grad]
        hd = [p for p in self.model.head.parameters() if p.requires_grad]
        groups = []
        if bb:
            groups.append({"params": bb, "lr": backbone_lr})
        if hd:
            groups.append({"params": hd, "lr": head_lr})
        self.opt = AdamW(groups, weight_decay=weight_decay)
        self.best_dice = -1.0

    def _step_batch(self, batch, train: bool):
        img = batch["image"].to(self.device)
        mask = batch["mask"].to(self.device) # (B,1,256,256) float {0,1}, or (B,256,256) long
        logits = self.model(img) # (B,C,1008,1008)
        # resize to match logits
        mask = resize_mask(mask, logits.shape[-2:])
        loss = self.loss_fn(logits, mask)
        if train:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return loss.item(), logits.detach(), mask

    def train_epoch(self):
        self.model.train()
        total = 0.0
        for batch in self.loaders["train"]:
            loss, _, _ = self._step_batch(batch, train=True)
            total += loss
        return total / max(1, len(self.loaders["train"]))

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        dices, nsds = [], []
        for batch in self.loaders["val"]:
            _, logits, mask = self._step_batch(batch, train=False)
            # NSD tol scales with output resolution (2px @512, 4px @1008) so every arm forgives the same real displacement. Matches eval_model.default_tol.
            tol = max(1, round(2 * logits.shape[-1] / 512))
            dsc, nsd_v = seg_scores(logits, mask, tol=tol)
            dices.append(dsc)
            nsds.append(nsd_v)
        return mean_score(dices), mean_score(nsds)

    def fit(self, epochs: int):
        for ep in range(1, epochs + 1):
            tr = self.train_epoch()
            dsc, nsd_v = self.validate()
            print(f"epoch {ep}: train_loss={tr:.4f}  val_dice={dsc:.4f}  val_nsd={nsd_v:.4f}")
            if dsc > self.best_dice:
                self.best_dice = dsc
                path = os.path.join(self.ckpt_dir, "best.pt")
                torch.save({"model": self.model.state_dict(),
                            "epoch": ep, "dice": dsc}, path)
                print(f"  saved {path} (dice={dsc:.4f})")
        return self.best_dice