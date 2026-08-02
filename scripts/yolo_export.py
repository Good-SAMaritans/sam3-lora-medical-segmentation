"""
yolo_export.py - one-shot: MedSegBench -> COCO -> YOLO-txt + data.yaml.

    python scripts/yolo_export.py --name kvasir --out data/yolo/kvasir --size 512
    python scripts/yolo_export.py --name busi   --out data/yolo/busi   --size 512
    python scripts/yolo_export.py --name monusac   --out data/yolo/monusac   --size 512

Steps:
  1. Export COCO json + images (medseg.models.yolo.export). Prints reconstruction
     Dice -- READ IT: below ~0.97 means polygon simplification is eating the
     boundary and is capping YOLO's achievable NSD before training even starts.
     Note: this is occuring for monusac no matter what pretty much. not a great sign
  2. Convert COCO -> YOLO-txt via ultralytics.data.converter.convert_coco.
  3. Assemble the images/ + labels/ tree and write data.yaml.
  4. Verify the tree, and fail on a label/image mismatch.
"""

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, "src")

from medseg.models.yolo.export import ExportConfig, export_dataset
from ultralytics.data.converter import convert_coco
#new for monusac
PRESETS = {
    "monusac": dict(
        class_values=[1, 2, 3, 4],
        class_names=["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"],
    ),
}

def _find_label_dir(save_dir: str, split: str) -> str:
    """Locate the label dir convert_coco produced for split, across versions."""
    candidates = [
        os.path.join(save_dir, "labels", split),
        os.path.join(save_dir, "labels", f"instances_{split}"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # last resort: any labels/* dir whose name contains the split. suboptimal but it has to be done, i wish it was cleaner
    for d in glob.glob(os.path.join(save_dir, "labels", "*")):
        if os.path.isdir(d) and split in os.path.basename(d):
            return d
    raise FileNotFoundError(
        f"convert_coco produced no label dir for split '{split}'. "
        f"Looked in {os.path.join(save_dir, 'labels')}: "
        f"{os.listdir(os.path.join(save_dir, 'labels')) if os.path.isdir(os.path.join(save_dir,'labels')) else 'MISSING'}"
    )
def _names_block(cfg) -> str:
    if cfg.class_values is None: #binary return
        return f"names:\n  0: {cfg.class_name}\n"
    names = cfg.class_names or [f"class{v}" for v in cfg.class_values]
    body = "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    return "names:\n" + body

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)#medsegbench key, e.g. kvasir|busi
    ap.add_argument("--out", required=True)#"output dataset root"
    ap.add_argument("--size", type=int, default=512, choices=[128, 256, 512])
    ap.add_argument("--category", default=None) #BUSI: C1|C2, default full set
    ap.add_argument("--eps-frac", type=float, default=0.004) #polygon simplification; lower = higher boundary fidelity
    ap.add_argument("--class-name", default="lesion") #binary only 
    ap.add_argument("--class-values", default=None) #need comma separated names for vals and names
    ap.add_argument("--class-names", default=None)
    args = ap.parse_args()
    #multiclass stuff
    preset = PRESETS.get(args.name.lower(), {})
    if args.class_values is not None:
        class_values = [int(v) for v in args.class_values.split(",")]
    else:
        class_values = preset.get("class_values")
    if args.class_names is not None:
        class_names = args.class_names.split(",")
    else:
        class_names = preset.get("class_names")
 
    if class_values is not None:
        if class_names and len(class_names) != len(class_values):
            raise ValueError(
                f"--class-names has {len(class_names)} entries but --class-values "
                f"has {len(class_values)}; they must correspond one-to-one.")
        print(f"multi-class export: GT values {class_values} -> YOLO classes "
              f"{list(range(len(class_values)))}"
              + (f"  {class_names}" if class_names else ""))
        print("  (any GT value not listed is NOT exported -- e.g. MoNuSAC "
              "Ambiguous=5 and background=0)")
    coco_root = os.path.join(args.out, "coco")
    cfg = ExportConfig(
        name=args.name, out_root=coco_root, size=args.size,
        category=args.category, eps_frac=args.eps_frac, class_name=args.class_name,
        class_values=class_values, class_names=class_names #new line for multiclass
    )
    stats = export_dataset(cfg)
    worst = min(s["recon_dice_mean"] for s in stats)
    if worst < 0.97:
        print(f"\n!! reconstruction Dice {worst:.4f} < 0.97 -- polygon simplification "
              f"is losing boundary detail. Re-run with a smaller --eps-frac "
              f"(current {args.eps_frac}).")
    # --- COCO -> YOLO-txt ---
    staging = os.path.join(args.out, "_convert")
    shutil.rmtree(staging, ignore_errors=True)
    convert_coco(
        labels_dir=os.path.join(coco_root, "annotations"),
        save_dir=staging,
        use_segments=True,      # polygons, not boxes
        cls91to80=False,
    )

    # --- assemble images/ + labels/ tree ---
    yolo_root = os.path.join(args.out, "yolo")
    shutil.rmtree(yolo_root, ignore_errors=True)
    summary = []
    for split in cfg.splits:
        lbl_src = _find_label_dir(staging, split)
        lbl_dst = os.path.join(yolo_root, "labels", split)
        img_src = os.path.join(coco_root, "images", split)
        img_dst = os.path.join(yolo_root, "images", split)

        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        shutil.copytree(lbl_src, lbl_dst)
        shutil.copytree(img_src, img_dst)     # copy, not symlink: survives rsync to a cluster

        n_img = len(glob.glob(os.path.join(img_dst, "*.png")))
        n_lbl = len(glob.glob(os.path.join(lbl_dst, "*.txt")))
        summary.append((split, n_img, n_lbl))

        if n_img == 0:
            raise RuntimeError(f"split '{split}': 0 images -- export failed")
        if n_lbl == 0:
            raise RuntimeError(
                f"split '{split}': {n_img} images but 0 label files. "
                f"convert_coco layout probe failed -- training would silently "
                f"run on no labels."
            )

    shutil.rmtree(staging, ignore_errors=True)

    # --- data.yaml ---
    yaml_path = os.path.join(yolo_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"# auto-generated by scripts/yolo_export.py\n"
            f"path: {os.path.abspath(yolo_root)}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n" 
            + _names_block(cfg)
        )

    print("\n=== YOLO tree ===")
    for split, n_img, n_lbl in summary:
        note = "" if n_img == n_lbl else f"  ({n_img - n_lbl} background image(s), no .txt -- expected)"
        print(f"  {split:5s}  images={n_img:4d}  labels={n_lbl:4d}{note}")
    print(f"\nready to train: {yaml_path}")
    print(f"  python scripts/yolo_train.py --data {yaml_path} "
          f"--coco-root {coco_root} --imgsz {args.size}")


if __name__ == "__main__":
    main()
