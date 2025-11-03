"""End-to-end RGB-D semantic segmentation demo using the real datasets.

This script mirrors the repository training pipeline but avoids the helper
shell scripts.  Instead of fabricating a synthetic dataset it loads an existing
dataset from ``datasets/<DATASET_NAME>`` (for example ``NYUDepthv2`` or
``SUNRGBD``), instantiates :class:`utils.dataloader.RGBXDataset.RGBXDataset`,
wraps it with :class:`torch.utils.data.DataLoader`, builds the DFormer network
and runs a light-weight optimisation loop followed by a single inference pass.

Dependencies: PyTorch, mmcv>=1.7 and mmengine must be installed in the active
environment.
"""

from __future__ import annotations

import argparse
import importlib
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from models.builder import EncoderDecoder
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import TrainPre, ValPre

def load_experiment_config(dataset: str, variant: str, datasets_root: Path) -> edict:
    """Load a training config from ``local_configs`` with a custom dataset root."""

    datasets_root = datasets_root.resolve()

    base_module = importlib.import_module("local_configs._base_")
    base_module = importlib.reload(base_module)
    base_module.C.root_dir = base_module.config.root_dir = str(datasets_root)

    dataset_module_name = f"local_configs._base_.datasets.{dataset}"
    dataset_module = importlib.import_module(dataset_module_name)
    dataset_module = importlib.reload(dataset_module)

    variant_module_name = f"local_configs.{dataset}.{variant}"
    variant_module = importlib.import_module(variant_module_name)
    variant_module = importlib.reload(variant_module)

    config = deepcopy(variant_module.config)

    # Some demo-friendly defaults.
    if not hasattr(config, "pretrained_model"):
        config.pretrained_model = None
    if not hasattr(config, "pad"):
        config.pad = False
    if not hasattr(config, "num_workers"):
        config.num_workers = 0

    return config


def ensure_dataset_exists(config: edict) -> None:
    missing = []
    for attr in [
        "dataset_path",
        "rgb_root_folder",
        "x_root_folder",
        "gt_root_folder",
        "train_source",
        "eval_source",
    ]:
        value = Path(getattr(config, attr))
        if not value.exists():
            missing.append(value)

    if missing:
        formatted = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "The following dataset paths were not found. Please download the dataset "
            "and place it under the configured datasets root:\n  - " + formatted
        )


def build_datasets(
    config: edict,
    *,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
):
    """Instantiate training and validation datasets and dataloaders."""

    train_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
        "x_modal": getattr(config, "x_modal", ["d"]),
    }

    train_dataset = RGBXDataset(
        train_setting,
        split_name="train",
        preprocess=TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config),
    )

    if max_train_samples is not None:
        train_dataset._file_names = train_dataset._file_names[:max_train_samples]

    val_dataset = RGBXDataset(
        train_setting,
        split_name="val",
        preprocess=ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config),
    )

    if max_val_samples is not None:
        val_dataset._file_names = val_dataset._file_names[:max_val_samples]

    drop_last = len(train_dataset) >= config.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=len(train_dataset) > 1,
        drop_last=drop_last,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_model(config: edict, device: torch.device) -> EncoderDecoder:
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    model = EncoderDecoder(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
    return model.to(device)


def train_one_epoch(model: EncoderDecoder, loader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        rgb = batch["data"].to(device)
        modal_x = batch["modal_x"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(rgb, modal_x, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def run_inference(model: EncoderDecoder, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    batch = next(iter(loader))
    rgb = batch["data"].to(device)
    modal_x = batch["modal_x"].to(device)
    logits = model(rgb, modal_x)
    probabilities = torch.softmax(logits, dim=1)
    prediction = probabilities.argmax(dim=1).cpu().numpy()[0]
    return prediction, probabilities.max(dim=1).values.cpu().numpy()[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="NYUDepthv2",
        choices=["NYUDepthv2", "SUNRGBD"],
        help="Dataset name matching the folders under the datasets root.",
    )
    parser.add_argument(
        "--variant",
        default="DFormer_Tiny",
        help="Model variant from local_configs.<dataset> (e.g. DFormer_Tiny).",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets",
        help="Base directory containing the dataset folders.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=8,
        help="Optional cap on the number of training images to keep the demo light-weight (use 0 for all).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=2,
        help="Optional cap on the number of validation images processed during the demo (use 0 for all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size defined in the config (defaults to the config value).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the dataloader worker count (defaults to the config value).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_experiment_config(args.dataset, args.variant, args.datasets_root)

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    ensure_dataset_exists(config)

    max_train = args.max_train_samples if args.max_train_samples and args.max_train_samples > 0 else None
    max_val = args.max_val_samples if args.max_val_samples and args.max_val_samples > 0 else None

    train_loader, val_loader = build_datasets(
        config,
        max_train_samples=max_train,
        max_val_samples=max_val,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print(
        "Using dataset '%s' from %s (%d train / %d val samples)."
        % (
            config.dataset_name,
            config.dataset_path,
            len(train_loader.dataset),
            len(val_loader.dataset),
        )
    )
    if max_train is not None:
        print(f"  -> Training subset capped to {len(train_loader.dataset)} samples.")
    if max_val is not None:
        print(f"  -> Validation subset capped to {len(val_loader.dataset)} samples.")

    print(f"Training on {len(train_loader.dataset)} samples for {args.epochs} epoch(s)...")
    for epoch in range(1, args.epochs + 1):
        mean_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch}: loss={mean_loss:.4f}")

    prediction, confidence = run_inference(model, val_loader, device)
    unique, counts = np.unique(prediction, return_counts=True)
    print("Inference finished on one validation image.")
    print("Predicted class distribution (class_id: pixel_count):")
    for class_id, count in zip(unique, counts):
        print(f"  {class_id}: {int(count)}")
    print(f"Mean confidence: {confidence.mean():.3f}")


if __name__ == "__main__":
    main()
