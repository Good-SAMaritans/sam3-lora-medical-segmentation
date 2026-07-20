# SAM weights: download & use

The Segment Anything Model (SAM) checkpoint is the **frozen pretrained backbone**
every adaptation arm builds on (zero-shot, decoder-only, LoRA, full fine-tuning).
It is not stored in the repo — this doc is how you get it and use it.

---

## Why it isn't in the repo

The checkpoint is a ~375 MB binary. Committing it would bloat every clone and
every fetch, so it is **gitignored** (`.gitignore` has both `checkpoints/` and
`*.pth`). The repo stores the *instructions* to fetch it, not the file itself — so
a fresh `git clone` gives you the code but **not** the weights. You download them
once, per machine.

## Download (first-time setup)

From the repo root:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

- `-L` follows redirects (the URL bounces to a CDN); without it you'd save the
  redirect page instead of the file.
- `-o <path>` writes to `checkpoints/sam_vit_b_01ec64.pth` (gitignored).

## Verify it downloaded intact

```bash
ls -lh checkpoints/sam_vit_b_01ec64.pth        # expect ~375M
python -c "import torch; print('keys:', len(torch.load('checkpoints/sam_vit_b_01ec64.pth', map_location='cpu')))"
```

A healthy checkpoint prints a few hundred tensor keys. A few-KB file or a load
error means a partial download (usually a missing `-L`) — delete and retry.

## Which checkpoint, and why

SAM ships in three sizes. We use **ViT-B**.

| Variant | File | Size | Registry key |
|---|---|---|---|
| ViT-B (base) | `sam_vit_b_01ec64.pth` | ~375 MB | `vit_b` |
| ViT-L (large) | `sam_vit_l_0b3195.pth` | ~1.2 GB | `vit_l` |
| ViT-H (huge) | `sam_vit_h_4b8939.pth` | ~2.5 GB | `vit_h` |

ViT-B is the lightest to fine-tune, fits comfortably on Colab and an MPS Mac, and
is what the project proposal specifies. (The `01ec64` in the filename is just a
version hash from Meta.) To use a bigger variant, download its file above and pass
the matching registry key.

## Load and use it in code

```python
from segment_anything import sam_model_registry
from medseg.models.lora import inject_lora_into_sam

# load the frozen pretrained backbone from the file you downloaded
sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")

# freeze SAM, inject trainable LoRA adapters into the encoder's Q/V projections
inject_lora_into_sam(sam, r=4, alpha=4)
# -> only ~147K of ~94M params trainable (~0.16%)
```

`sam_model_registry[key](checkpoint=...)` builds the SAM architecture for that
variant and loads the weights from the `.pth` file. The `segment-anything` package
must be installed first (`pip install "git+https://github.com/facebookresearch/segment-anything.git"`).

The checkpoint path is currently passed directly; once `configs/model/sam_vit_b.yaml`
exists it will hold this path (and `lora_r`, `lora_alpha`) so it isn't hardcoded and
can be overridden per machine.

## What "frozen" means here

During LoRA (and decoder-only) fine-tuning, these downloaded weights are **not**
updated — `inject_lora_into_sam` sets `requires_grad = False` on all of SAM, and
only the small adapters train. So the checkpoint stays exactly as downloaded; the
learned part lives in a separate, tiny adapter file. See [LoRA](lora.md).
