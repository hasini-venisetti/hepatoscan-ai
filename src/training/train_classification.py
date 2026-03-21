"""Phase 2: Classification head training with frozen encoder.

Freezes the Swin UNETR encoder (trained in Phase 1) and trains
the classification head only using Focal Loss.

Target metrics:
- Binary AUC > 0.92
- Per-class F1 > 0.75

Usage:
    python -m src.training.train_classification --config configs/multitask_config.yaml --phase1_ckpt checkpoints/phase1_best.pt
    python -m src.training.train_classification --config configs/multitask_config.yaml --resume checkpoints/phase2_epoch50.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.models.hepatoscan import HepatoScanAI
from src.models.backbone import freeze_encoder
from src.losses.focal_loss import FocalLoss
from src.training.trainer import BaseTrainer
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import get_dataloaders, compute_class_weights

PHASE2_CHECKPOINT_PREFIX = "phase2"

logger = logging.getLogger(__name__)


def train_classification(
    config: dict,
    phase1_checkpoint: Optional[str] = None,
    resume_path: Optional[str] = None,
) -> dict:
    """Run Phase 2: classification head training.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    phase1_checkpoint : Optional[str]
        Path to Phase 1 best checkpoint to load encoder weights.
    resume_path : Optional[str]
        Path to Phase 2 checkpoint to resume from.

    Returns
    -------
    dict
        Training summary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 2 — Classification Training on %s", device)

    # Build full model
    model = HepatoScanAI(config)

    # Load Phase 1 encoder weights
    if phase1_checkpoint and Path(phase1_checkpoint).exists():
        logger.info("Loading Phase 1 weights from %s", phase1_checkpoint)
        checkpoint = torch.load(phase1_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Load matching keys only (backbone weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        logger.info("Loaded %d/%d parameter tensors from Phase 1", len(pretrained_dict), len(model_dict))

    # Freeze encoder — train only classification head
    freeze_encoder(model.backbone)

    # Build data loaders
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    train_loader, val_loader, _ = get_dataloaders(config, train_transforms, val_transforms)

    if train_loader is None:
        logger.error("No training data found.")
        return {"error": "No data"}

    # Classification loss with class weights
    criterion = FocalLoss(
        gamma=config.get("loss", {}).get("focal_loss_gamma", 2.0),
    )

    # Optimizer — higher LR since only head is trained
    training_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_cfg.get("learning_rate", 1e-3),
        weight_decay=1e-4,
    )

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
    )

    if resume_path is not None:
        trainer.resume_from_checkpoint(resume_path)

    summary = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_prefix=PHASE2_CHECKPOINT_PREFIX,
    )

    logger.info("Phase 2 complete: %s", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="HepatoScan AI Phase 2: Classification Training")
    parser.add_argument("--config", type=Path, default=Path("configs/multitask_config.yaml"))
    parser.add_argument("--base_config", type=Path, default=Path("configs/base_config.yaml"))
    parser.add_argument("--phase1_ckpt", type=str, default="checkpoints/phase1_best.pt")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        mt_cfg = yaml.safe_load(f)

    config = {**base_cfg}
    for key, value in mt_cfg.items():
        if isinstance(value, dict) and key in config:
            config[key] = {**config[key], **value}
        else:
            config[key] = value

    train_classification(config, phase1_checkpoint=args.phase1_ckpt, resume_path=args.resume)


if __name__ == "__main__":
    main()
