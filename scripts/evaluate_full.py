"""Evaluate a fully tuned SAM 3 checkpoint on its test set."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, "src")

from medseg.data.medseg_data import DataConfig, build_dataloaders
from medseg.engine.metrics import dice_score, nsd
from medseg.models import Sam3MedSeg
from medseg.utils.device import get_device


DEFAULT_CONFIG = "configs/experiment/sam3_full_busi_full.yaml"
EXAMPLE_COUNT = 5
PREDICTION_THRESHOLD = 0.5
NSD_TOLERANCE = 2
RANDOM_SEED = 42


@torch.no_grad()
def predict(model, image, device):
    """Return a probability mask at the image's original size."""
    model.eval()

    logits = model(image.unsqueeze(0).to(device))
    probability = torch.sigmoid(logits)

    return F.interpolate(
        probability,
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu()


def image_dice(probability, mask):
    """Calculate Dice for one displayed image."""
    prediction = (probability > PREDICTION_THRESHOLD).float()
    intersection = (prediction * mask).sum()
    denominator = prediction.sum() + mask.sum()

    return ((2 * intersection + 1e-6) / (denominator + 1e-6)).item()


@torch.no_grad()
def evaluate_test_set(model, test_loader, device):
    """Calculate mean test Dice and NSD using the training metric logic."""
    model.eval()
    scores = []

    for batch in test_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)

        # Training validation compares masks at SAM 3's 1008-pixel output size.
        masks = F.interpolate(
            masks.float(),
            size=logits.shape[-2:],
            mode="nearest",
        )
        probabilities = torch.sigmoid(logits)

        for index in range(images.shape[0]):
            scores.append(
                {
                    "dice": dice_score(
                        probabilities[index:index + 1],
                        masks[index:index + 1],
                    ),
                    "nsd": nsd(
                        probabilities[index:index + 1],
                        masks[index:index + 1],
                        tol=NSD_TOLERANCE,
                    ),
                }
            )

    mean_dice = sum(row["dice"] for row in scores) / len(scores)
    mean_nsd = sum(row["nsd"] for row in scores) / len(scores)

    return mean_dice, mean_nsd, scores


def save_examples(
    samples,
    baseline_probabilities,
    tuned_probabilities,
    dataset_name,
    output_path,
):
    """Save image, ground truth, baseline, and tuned predictions."""
    row_count = len(samples)
    figure, axes = plt.subplots(
        row_count,
        4,
        figsize=(14, 3.5 * row_count),
        squeeze=False,
    )

    for row, sample in enumerate(samples):
        image = sample["image"].permute(1, 2, 0).numpy()
        mask = sample["mask"][0].numpy()
        baseline = baseline_probabilities[row]
        tuned = tuned_probabilities[row]

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{dataset_name} test image {row}")

        axes[row, 1].imshow(image)
        axes[row, 1].imshow(mask, cmap="Reds", alpha=0.5)
        axes[row, 1].set_title("Ground truth")

        axes[row, 2].imshow(
            baseline,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title(
            f"Untrained\nDice {image_dice(baseline, sample['mask'][0]):.3f}"
        )

        axes[row, 3].imshow(
            tuned,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[row, 3].set_title(
            f"Full tuning\nDice {image_dice(tuned, sample['mask'][0]):.3f}"
        )

        for axis in axes[row]:
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    torch.manual_seed(RANDOM_SEED)

    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG

    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    dataset_name = config["data"]["name"]
    checkpoint_path = Path(config["train"]["ckpt_dir"]) / "best.pt"
    output_dir = Path("results/full_finetuning") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    loaders = build_dataloaders(DataConfig.from_dict(config["data"]))
    test_loader = loaders["test"]
    test_dataset = test_loader.dataset

    model = Sam3MedSeg(**config["model"]).to(device)

    example_count = min(EXAMPLE_COUNT, len(test_dataset))
    samples = [test_dataset[index] for index in range(example_count)]

    # This is the pretrained backbone with its original random segmentation head.
    baseline_probabilities = [
        predict(model, sample["image"], device)
        for sample in samples
    ]

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])

    tuned_probabilities = [
        predict(model, sample["image"], device)
        for sample in samples
    ]

    mean_dice, mean_nsd, image_scores = evaluate_test_set(
        model,
        test_loader,
        device,
    )

    figure_path = output_dir / "predictions.png"
    save_examples(
        samples,
        baseline_probabilities,
        tuned_probabilities,
        dataset_name,
        figure_path,
    )

    results = {
        "dataset": dataset_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_dice": checkpoint["dice"],
        "test_images": len(test_dataset),
        "test_mean_dice": mean_dice,
        "test_mean_nsd": mean_nsd,
        "image_scores": image_scores,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"saved: {metrics_path}")
    print(f"saved: {figure_path}")


if __name__ == "__main__":
    main()