"""Phase 1: Segmentation-only training.

Trains Swin UNETR for liver + tumor segmentation only.
Uses DiceCELoss with LiTS + Decathlon + HCC-TACE-Seg datasets.

Target metrics:
- Liver Dice > 0.95
- Tumor Dice > 0.65

Usage:
    python -m src.training.train_segmentation --config configs/segmentation_config.yaml
    python -m src.training.train_segmentation --config configs/segmentation_config.yaml --resume checkpoints/phase1_epoch50.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

try:
    from monai.losses import DiceCELoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

from src.models.backbone import build_swin_unetr
from src.training.trainer import BaseTrainer
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import get_dataloaders

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE1_CHECKPOINT_PREFIX = "phase1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 Training
# ---------------------------------------------------------------------------


def train_segmentation(config: dict, resume_path: Optional[str] = None) -> dict:
    """Run Phase 1: segmentation-only training.

    Parameters
    ----------
    config : dict
        Configuration dictionary (merged base + segmentation config).
    resume_path : Optional[str]
        Path to checkpoint to resume from.

    Returns
    -------
    dict
        Training summary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 1 — Segmentation Training on %s", device)

    # Build model (segmentation-only mode)
    model = build_swin_unetr(
        img_size=tuple(config["model"]["img_size"]),
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"].get("out_channels", 3),
        feature_size=config["model"]["feature_size"],
        pretrained=config["model"]["pretrained"],
        pretrained_path=config["model"].get("pretrained_path"),
        use_checkpoint=config["model"]["use_checkpoint"],
    )

    # Loss function
    if MONAI_AVAILABLE:
        criterion = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    training_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-4),
        weight_decay=1e-5,
    )

    # Build data loaders
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    train_loader, val_loader, _ = get_dataloaders(config, train_transforms, val_transforms)

    if train_loader is None:
        logger.error("No training data found. Ensure data is preprocessed and available.")
        return {"error": "No data"}

    # Build trainer
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    if resume_path is not None:
        trainer.resume_from_checkpoint(resume_path)
        logger.info("Resumed from checkpoint: %s", resume_path)

    # Train
    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_prefix=PHASE1_CHECKPOINT_PREFIX,
    )

    logger.info("Phase 1 complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for Phase 1 training."""
    parser = argparse.ArgumentParser(description="HepatoScan AI Phase 1: Segmentation Training")
    parser.add_argument("--config", type=Path, default=Path("configs/segmentation_config.yaml"))
    parser.add_argument("--base_config", type=Path, default=Path("configs/base_config.yaml"))
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load and merge configs
    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        seg_cfg = yaml.safe_load(f)

    # Simple merge (seg_cfg overrides base_cfg)
    config = {**base_cfg}
    for key, value in seg_cfg.items():
        if isinstance(value, dict) and key in config:
            config[key] = {**config[key], **value}
        else:
            config[key] = value

    train_segmentation(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
