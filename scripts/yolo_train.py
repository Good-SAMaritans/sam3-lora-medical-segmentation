"""
yolo_train.py - fine-tune YOLO-seg on an exported dataset, then score it with
the project's own Dice/NSD (not Ultralytics' mAP).

    python scripts/yolo_train.py --data data/yolo/kvasir/yolo/data.yaml
        --coco-root data/yolo/kvasir/coco --model yolo11s-seg.pt
        --epochs 100 --imgsz 512
this is just a little wrapper, yolo implementation already follows under ultralytics
overall, main follows:
1. putting args in main
2. training the yolo model
3. evaluating the model (via evaluate_yolo.py) on the best weights found in training

YOLO-seg's role in the study is a middle-model between UNet and SAM3, while also having a totally different architecture by being a CNN instead of anything attention-based.
"""

import argparse
import os
import sys
import json
sys.path.insert(0, "src")
from eval_model import evaluate 
from ultralytics import YOLO
from medseg.models.yolo.evaluate import evaluate_yolo
from medseg.models.yolo.monusac_adapter import YoloSemanticAdapter
def build_parser():
    ap = argparse.ArgumentParser(description="Train + evaluate YOLO-seg.")
    # --- required paths ---
    ap.add_argument("--data", required=True) #path to data yaml
    ap.add_argument("--coco-root", required=True) #path to coco root
    # --- run identity / infra ---
    ap.add_argument("--model", default="yolo11s-seg.pt") #typically that
    ap.add_argument("--name", default="yolo_seg")
    ap.add_argument("--device", default=None) #0, cpu, or mps
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42069, help="controlled comparison -> keep fixed") #can be unfixed later, but defaulting to some value enables repeatability
    # --- core training schedule ---
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=-1) #-1 is automatic allocation, but can be set manually
    ap.add_argument("--patience", type=int, default=100) #early stop
    ap.add_argument("--close-mosaic", type=int, default=10) #stops mosaic augmentation 10 epochs before finish
    # --- optimizer ---
    ap.add_argument("--optimizer", default="auto") #typically using auto but can go from auto, sgd, adam, and adamw
    ap.add_argument("--lr0", type=float, default=None) #LR
    ap.add_argument("--lrf", type=float, default=None) #final LR
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--warmup-epochs", type=float, default=None)
    # --- augmentation (COCO defaults are aggressive for small medical sets) ---
    ap.add_argument("--mosaic", type=float, default=None) #typically trying 0.5 honestly
    ap.add_argument("--mixup", type=float, default=None)
    ap.add_argument("--hsv-h", type=float, default=None) #hue shifting
    ap.add_argument("--hsv-s", type=float, default=None) #saturation shifting
    ap.add_argument("--hsv-v", type=float, default=None) #value shifting
    ap.add_argument("--degrees", type=float, default=None) #rotation augmentation for data
    ap.add_argument("--fliplr", type=float, default=None) #horizontal flip prob
    ap.add_argument("--flipud", type=float, default=None) #vert flip probability
    # --- evaluation ---
    ap.add_argument("--conf", type=float, default=0.25) #confidence
    ap.add_argument("--tol", type=float, default=2.0) #nsd tol
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default=None) #(busi, kvasir, monusac)
    ap.add_argument("--shared-eval", action="store_true")
    ap.add_argument("--save-period", type=int, default=-1) #this is for saving each thing to generate sufficient dice graphs
    return ap
 
def _shared_evaluate(best, args, dataset):
    """Score via scripts/eval_model.py -- the SAME evaluator the SAM 3 and U-Net
    arms use, so the numbers belong in one table. The adapter turns YOLO's
    per-instance output into the (B,C,H,W) logits it expects; eval_model then
    applies its own argmax/sigmoid, resizes GT to the model's output size, sets
    NSD tol from that resolution, and (MoNuSAC) drops Ambiguous via
    ignore_index=5. None of that is reimplemented here."""
    sys.path.insert(0, "scripts") #imports the eval and adapter
    from eval_model import evaluate 
    nc = _n_classes(args.data)
    adapter = YoloSemanticAdapter(
        best, conf=args.conf, imgsz=args.imgsz,
        n_classes=(nc + 1) if nc > 1 else 1,   # +1 for the background channel
    )
    # normalize=None: the adapter hands raw [0,1] images straight to YOLO.
    return evaluate(
        adapter, dataset, splits=(args.split,), size=args.imgsz,
        normalize=None, csv_path=os.path.join("results", "metrics", "yolo_shared_eval.csv"),
    )
 
def _n_classes(data_yaml: str) -> int:
    """Read the class count from data.yaml so we know binary vs multi-class."""
    try:
        import yaml
        with open(data_yaml) as f:
            names = yaml.safe_load(f).get("names", {})
        return len(names) if names else 1
    except Exception:
        return 1
def _shared_evaluate(best, args, dataset):
    """Score via scripts/eval_model.py -- the SAME evaluator the SAM 3 and U-Net
    arms use, so the numbers belong in one table. The adapter turns YOLO's
    per-instance output into the (B,C,H,W) logits it expects; eval_model then
    applies its own argmax/sigmoid, resizes GT to the model's output size, sets
    NSD tol from that resolution, and (MoNuSAC) drops Ambiguous via
    ignore_index=5. None of that is reimplemented here."""
    sys.path.insert(0, "scripts")
    nc = _n_classes(args.data)
    adapter = YoloSemanticAdapter(
        best, conf=args.conf, imgsz=args.imgsz,
        n_classes=(nc + 1) if nc > 1 else 1,   # +1 for the background channel
    )
    # normalize=None: the adapter hands raw [0,1] images straight to YOLO.
    return evaluate(
        adapter, dataset, splits=(args.split,), size=args.imgsz,
        normalize=None, csv_path=os.path.join("results", "metrics", "yolo_shared_eval.csv"),
    )
 
def main():
    args = build_parser().parse_args()
 
    # Only pass hyperparameters the user actually set, so unset ones keep the
    # model's stock default (which Ultralytics records in args.yaml).
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        close_mosaic=args.close_mosaic,
        optimizer=args.optimizer,
        name=args.name,
        save_period=args.save_period,
        plots=True,
    )
    optional = {
        "lr0": args.lr0, "lrf": args.lrf, "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs, "mosaic": args.mosaic, "mixup": args.mixup,
        "hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v,
        "degrees": args.degrees, "fliplr": args.fliplr, "flipud": args.flipud,
    }
    train_kwargs.update({k: v for k, v in optional.items() if v is not None})
    model = YOLO(args.model)
    model.train(**train_kwargs)
 
    # Where Ultralytics actually saved (runs_dir is user-configurable).
    save_dir = getattr(model.trainer, "save_dir", None)
    save_dir = str(save_dir) if save_dir else os.path.join("runs", "segment", args.name)
    best = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.isfile(best):
        raise FileNotFoundError(f"trained checkpoint not found: {best}")
    print(f"\nbest checkpoint: {best}")
 
    # --- write the hyperparameter record ---
    os.makedirs(os.path.join("results", "metrics"), exist_ok=True)
    config_path = os.path.join("results", "metrics", f"{args.name}_config.json")
    record = {
        "name": args.name,
        "model": args.model,
        "data": args.data,
        "requested_cli": vars(args),               # exactly what you typed / defaulted
        "train_kwargs_sent": _jsonable(train_kwargs),
        "ultralytics_resolved": _read_resolved_args(save_dir),  # full args.yaml YOLO used
        "checkpoint": best,
    }
    with open(config_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote hyperparameter record: {config_path}")
 
    # --- evaluate ---
    # evaluate_yolo is BINARY (it unions all instances into one foreground mask).
    # On a multi-class dataset like MoNuSAC that silently produces a meaningless
    # number, so we stop and point at the adapter instead, which routes through
    # the SHARED scripts/eval_model.py -- identical metric, identical
    # ignore_index handling, so YOLO lands in the same table as SAM 3 / U-Net.
    nc = _n_classes(args.data)
    if nc > 1 or args.shared_eval:
        if not args.dataset:
            raise SystemExit(
                "--dataset is required to use the shared evaluator "
                "(busi|kvasir|monusac). It names the medsegbench set that "
                "scripts/eval_model.py loads ground truth from.")
        dataset = args.dataset
        print(f"\nscoring with the shared evaluator (scripts/eval_model.py): "
              f"dataset={dataset}, {nc} class(es)")
        _shared_evaluate(best, args, dataset)
        return
    evaluate_yolo(
        weights=best,
        coco_root=args.coco_root,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        tol=args.tol,
        device=args.device,
        out_json=os.path.join("results", "metrics", f"{args.name}_{args.split}.json"),
    )
 
 
def _read_resolved_args(save_dir):
    """Load the args.yaml Ultralytics wrote (the fully-resolved hyperparams)."""
    path = os.path.join(save_dir, "args.yaml")
    if not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        with open(path) as f:
            return f.read()
 
 
def _jsonable(d):
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in d.items()}
 
 
if __name__ == "__main__":
    main()