# SAM 3 weights: access, download and use

The base model for every adaptation arm (zero-shot, decoder-only, LoRA, full
fine-tuning) is **SAM 3**, loaded through Meta's native `sam3` package. The
weights are license-gated and large, so neither the package source nor the
checkpoint lives in this repo. This doc is how you get both onto a machine.

## Why none of this is committed

- The checkpoint is ~6.9 GB and gated behind a license: it cannot be
  redistributed through our repo.
- The `sam3` package source we build against is a clone of
  `github.com/facebookresearch/sam3`. Where it exists locally (see the Mac
  section), it sits in `vendor/`, which is gitignored.

A fresh `git clone` of our repo therefore gives you the code but not the base
model. Setup is once per machine.

## 1. Get access (once per person)

1. Create a HuggingFace account and request access to the gated repo at
   `huggingface.co/facebook/sam3` (approval took about a day for us; reports
   range from days to weeks).
2. Create a **read** token: HF Settings, Access Tokens, "Create new token".
3. Authenticate the machine:

   ```bash
   hf auth login        # paste the read token
   hf auth whoami       # should print your username
   ```

## 2. Download the weights (once per machine)

Either pre-fill the cache explicitly (recommended, shows progress):

```bash
hf download facebook/sam3        # ~6.9 GB into ~/.cache/huggingface/hub/
```

or let the model builder fetch on first use (`load_from_HF=True` is the
default). Either way the weights land in the shared HF cache, and every later
load reads from disk.

### Pin the revision

HF model repos are git repos. If the upstream weights are ever updated, an
unpinned load would silently give different arms different base models and
break the controlled comparison. The snapshot we downloaded and verified is:

```
3c879f39826c281e95690f02c7821c4de09afae7   (downloaded 2026-07-17)
```

Proposed as the team-wide pin; belongs in the model config once that exists.
You can read your local snapshot hash off the cache path:

```bash
ls ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/
```

## 3. Load and use it in code

```python
from medseg.models import Sam3MedSeg

# the shared task model: SAM 3 vision backbone + FPN seg head
model = Sam3MedSeg(finetune_mode="lora")     # or "full" / "decoder_only"
```

`Sam3MedSeg` builds the backbone internally via
`sam3.model_builder.build_sam3_image_model()`, which loads the gated weights
from the HF cache. The `finetune_mode` flag selects what is trainable, so all
arms share one model definition. See [lora.md](lora.md) for what the LoRA mode
does and its parameter counts.

Notes on the builder itself:

- `build_sam3_image_model()` defaults to `device="cpu"`; move the model
  yourself (`.cuda()`, `.to("mps")`) as needed.
- The full `Sam3Image` it returns is 840.5M params (vision + language +
  detection). Our wrapper keeps only the vision path, 455.1M params; the
  language encoder and the DETR detection machinery are unused in this project.

## 4. Installing the `sam3` package

### Cluster / Linux with CUDA (the straightforward path)

Requirements from upstream: Python >= 3.12, torch >= 2.7, CUDA >= 12.6.

```bash
git clone https://github.com/facebookresearch/sam3.git
pip install -e sam3
```

### Apple Silicon Mac (local dev; officially unsupported upstream)

The package assumes CUDA in a few places, but the image-model path runs fine
on CPU/MPS with the workarounds below. This is what the local `sam3-native`
conda env contains (Python 3.12; our data/test deps are installed in the same
env so one env runs everything):

| Problem | Workaround |
|---|---|
| `pkg_resources` removed in setuptools >= 81 | `pip install "setuptools<81"` |
| Undeclared deps | `pip install einops pycocotools psutil` |
| `triton` has no macOS build, but is imported at module load | a stub `triton` package in the env's site-packages that satisfies the imports and raises if a kernel is ever actually launched (it also refuses imports that originate inside torch, so torch correctly detects "no triton") |
| Two hardcoded `device="cuda"` lines in `model/position_encoding.py` and `model/decoder.py` | patched locally to fall back to CPU when CUDA is unavailable |
| `sam3` pins `numpy<2`, our stack needs `numpy>=2` | keep numpy 2.x; the pin is metadata only, everything imports and runs |

The patched clone lives at `vendor/sam3-repo` (gitignored) and is installed
editable: `pip install -e vendor/sam3-repo`. None of these workarounds are
needed on the CUDA cluster.

## 5. What "frozen" means here

In the LoRA and decoder-only arms the downloaded weights are never updated:
`requires_grad` is set to `False` on the whole backbone and only the small
task-specific parts train (LoRA adapters and/or the seg head). The checkpoint
on disk stays exactly as downloaded; what we learn lives in separate, small
files. The full fine-tuning arm is the one that updates backbone weights, and
it starts from this same pinned checkpoint.
