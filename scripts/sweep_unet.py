"""Grid sweep: python scripts/sweep_unet.py [sweep.yaml]

"""
import copy, itertools, os, sys, time, traceback, yaml
sys.path.insert(0, "src")

from medseg.engine.experiment import run_experiment
from medseg.utils.tracking import completed_runs, log_run

SHORT = {"encoder_name": "enc", "encoder_weights": "pre", "head_lr": "lr",
         "backbone_lr": "blr", "batch_size": "bs", "aug_preset": "aug", "weight_decay": "wd"}


def slug(params: dict) -> str:
    parts = []
    for key, value in params.items():
        field = key.split(".")[-1]
        text = "none" if value is None else (f"{value:g}" if isinstance(value, float) else str(value))
        parts.append(f"{SHORT.get(field, field)}-{text}")
    return "_".join(parts)


spec = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/experiment/unet_sweep.yaml"))
base, grid = spec["base"], spec["grid"]
csv_path = os.path.join(base["train"].get("out_dir", "results"), "metrics", f"{spec['name']}.csv")
done = completed_runs(csv_path)

keys = list(grid)
todo = []
for combo in itertools.product(*grid.values()):
    params = dict(zip(keys, combo))
    run = f"{spec['name']}_{slug(params)}"
    if run not in done:
        todo.append((run, params))

print(f"{len(todo)} runs to go ({len(done)} already in {csv_path})")
start = time.time()

for i, (run, params) in enumerate(todo, 1):
    cfg = copy.deepcopy(base)
    for key, value in params.items():          # dotted keys, e.g. "train.head_lr"
        section, field = key.split(".")
        cfg[section][field] = value

    elapsed = (time.time() - start) / 60
    eta = elapsed / (i - 1) * (len(todo) - i + 1) if i > 1 else 0
    print(f"\n=== [{i}/{len(todo)}] {run}  (elapsed {elapsed:.0f}m, eta {eta:.0f}m)", flush=True)
    try:
        row = run_experiment(cfg, run)
    except Exception:
        traceback.print_exc()
        print(f"  FAILED {run} - continuing", flush=True)
        continue
    log_run(csv_path, row)
    print(f"  test_dice={row['test_dice']}  test_nsd={row['test_nsd']}  ({row['minutes']}m)", flush=True)

print(f"\ndone in {(time.time() - start) / 60:.0f}m -> {csv_path}")
