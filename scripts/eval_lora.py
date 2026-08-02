"""Score the three LoRA checkpoints with the shared 6-metric evaluator.

    python scripts/eval_lora.py                # all three
    python scripts/eval_lora.py monusac        # just one

Calls scripts/eval_model.py, the same evaluator the full fine-tuning arm uses, so
the two arms produce comparable numbers. It scores at whatever resolution the
model outputs, derives the NSD tolerance from that (2px @512, 4px @1008), and
resizes ground truth to match.

Two arguments MUST match how each checkpoint was trained, or the numbers come out
quietly wrong rather than obviously broken:

  size       resolution the run loaded images at. All three trained at 512 as of
             2026-07-31 (BUSI/Kvasir were 256 before that); eval_model defaults to 512.
  normalize  None for every SAM3 run: our configs set `normalize: null` because
             Sam3MedSeg._prep_input applies SAM 3's own 0.5/0.5 internally.
             eval_model defaults to "imagenet", which would double-normalize.

Both are read out of each run's experiment config rather than hardcoded here, so
they cannot drift from the configs the runs actually used.
"""
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from eval_model import evaluate
from medseg.models import Sam3MedSeg
from medseg.utils.device import get_device

REPO = Path(__file__).resolve().parent.parent

# dataset -> experiment config. One LoRA checkpoint per dataset; the path comes out of
# train.ckpt_dir in the config, for the same reason size/normalize do -- the config is the
# only place a run's identity is recorded, so nothing here can drift away from it.
RUNS = {
    "busi":    "configs/experiment/lora/baseline/sam3_lora_busi_full.yaml",
    "kvasir":  "configs/experiment/lora/baseline/sam3_lora_kvasir_full.yaml",
    "monusac": "configs/experiment/lora/baseline/sam3_lora_monusac_full.yaml",
}
METRICS = ("precision", "recall", "f1", "iou", "dice", "nsd")


def score(dataset: str, device: str, csv_path: str | None = None):
    cfg = yaml.safe_load(open(REPO / RUNS[dataset]))
    ckpt_file = REPO / cfg["train"]["ckpt_dir"] / "best.pt"

    data, model_cfg = cfg["data"], cfg["model"]
    num_classes = model_cfg.get("num_classes", 1)
    # eval_model wants the loading resolution; img_size is what the data layer
    # produces and falls back to `size`, matching DataConfig.effective_img_size()
    size = data.get("img_size") or data["size"]
    normalize = data.get("normalize")          # null in every SAM3 config -> None

    print(f"\n{'=' * 72}\n{dataset}  |  size={size}  normalize={normalize}  "
          f"num_classes={num_classes}\n{'=' * 72}")

    if not ckpt_file.exists():
        print(f"  SKIP - no checkpoint at {ckpt_file}")
        return None

    model = Sam3MedSeg(finetune_mode=model_cfg["finetune_mode"], num_classes=num_classes)
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"  loaded epoch {ckpt['epoch']} (val dice {ckpt['dice']:.4f})")

    # ignore_index="auto" -> 5 on MoNuSAC (Ambiguous), None elsewhere.
    # tol=None -> derived from the model's output resolution.
    return evaluate(model, dataset, size=size, normalize=normalize,
                    tol=None, ignore_index="auto", device=device,
                    csv_path=csv_path, verbose=True)


def main():
    wanted = sys.argv[1:] or list(RUNS)
    unknown = [d for d in wanted if d not in RUNS]
    if unknown:
        sys.exit(f"unknown dataset(s) {unknown}; choose from {list(RUNS)}")

    device = get_device()
    print(f"device: {device}")
    out_csv = REPO / "results" / "metrics" / "lora_eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for dataset in wanted:
        r = score(dataset, device, csv_path=str(out_csv))
        if r:
            results[dataset] = r

    print(f"\n{'=' * 72}\nSUMMARY - LoRA arm, test split\n{'=' * 72}")
    print(f"{'dataset':10s} " + " ".join(f"{m:>10s}" for m in METRICS))
    for dataset, r in results.items():
        test = r.get("test", {})
        print(f"{dataset:10s} " + " ".join(f"{test.get(m, float('nan')):10.4f}" for m in METRICS))
    print(f"\nappended to {out_csv}")


if __name__ == "__main__":
    main()
