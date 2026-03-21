"""Phase 3: End-to-end multi-task joint training.

Unfreezes ALL layers and trains the full HepatoScan model with
uncertainty-weighted multi-task loss (Kendall et al. 2018).

Uses very low learning rate (1e-5) — this is fine-tuning, not from scratch.

Usage:
    python -m src.training.train_multitask --config configs/multitask_config.yaml --phase2_ckpt checkpoints/phase2_best.pt
    python -m src.training.train_multitask --resume checkpoints/phase3_epoch30.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.models.hepatoscan import HepatoScanAI
from src.models.backbone import unfreeze_all
from src.losses.multitask_loss import UncertaintyWeightedLoss
from src.training.trainer import BaseTrainer
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import get_dataloaders

PHASE3_CHECKPOINT_PREFIX = "phase3"

logger = logging.getLogger(__name__)


def train_multitask(
    config: dict,
    phase2_checkpoint: Optional[str] = None,
    resume_path: Optional[str] = None,
) -> dict:
    """Run Phase 3: end-to-end multi-task training.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    phase2_checkpoint : Optional[str]
        Path to Phase 2 checkpoint with classification head.
    resume_path : Optional[str]
        Path to Phase 3 checkpoint to resume from.

    Returns
    -------
    dict
        Training summary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 3 — Multi-task Joint Training on %s", device)

    # Build full model
    model = HepatoScanAI(config)

    # Load Phase 2 weights
    if phase2_checkpoint and Path(phase2_checkpoint).exists():
        logger.info("Loading Phase 2 weights from %s", phase2_checkpoint)
        checkpoint = torch.load(phase2_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        logger.info("Loaded %d/%d parameter tensors from Phase 2", len(pretrained_dict), len(model_dict))

    # Unfreeze ALL layers for end-to-end fine-tuning
    unfreeze_all(model)

    # Uncertainty-weighted multi-task loss
    loss_cfg = config.get("loss", {})
    staging_weights = loss_cfg.get("staging_class_weights", [1.0, 1.0, 1.0, 3.0, 3.0])
    criterion = UncertaintyWeightedLoss(
        num_seg_classes=config["model"].get("out_channels", 2),
        focal_gamma=loss_cfg.get("focal_loss_gamma", 2.0),
        staging_weights=staging_weights,
        initial_sigma=loss_cfg.get("initial_sigma", [1.0, 1.0, 1.0]),
    )

    # Optimizer — very low LR for fine-tuning
    training_cfg = config.get("training", {})
    # Include loss sigma parameters in optimizer
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=training_cfg.get("learning_rate", 1e-5),
        weight_decay=1e-5,
    )

    # Build data loaders
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    train_loader, val_loader, _ = get_dataloaders(config, train_transforms, val_transforms)

    if train_loader is None:
        logger.error("No training data found.")
        return {"error": "No data"}

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
        checkpoint_prefix=PHASE3_CHECKPOINT_PREFIX,
    )

    logger.info("Phase 3 complete: %s", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="HepatoScan AI Phase 3: Multi-task Training")
    parser.add_argument("--config", type=Path, default=Path("configs/multitask_config.yaml"))
    parser.add_argument("--base_config", type=Path, default=Path("configs/base_config.yaml"))
    parser.add_argument("--phase2_ckpt", type=str, default="checkpoints/phase2_best.pt")
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

    train_multitask(config, phase2_checkpoint=args.phase2_ckpt, resume_path=args.resume)


if __name__ == "__main__":
    main()
