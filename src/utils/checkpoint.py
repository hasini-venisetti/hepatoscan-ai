"""Checkpoint save and resume utilities.

Supports saving and loading model state, optimizer state, scheduler state,
epoch counter, best metric, and arbitrary training metadata for seamless
resume across Kaggle sessions.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: float = -float("inf"),
    global_step: int = 0,
    config: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Save a training checkpoint.

    Parameters
    ----------
    path : str | Path
        Output file path.
    model : nn.Module
        Model to save.
    optimizer : Optional[Optimizer]
        Optimizer state to save.
    scheduler : Optional
        LR scheduler state to save.
    epoch : int
        Current epoch number.
    best_metric : float
        Best validation metric so far.
    global_step : int
        Global training step counter.
    config : Optional[dict]
        Training configuration for reproducibility.
    extra : Optional[dict]
        Additional metadata to save.

    Returns
    -------
    Path
        Path to the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "global_step": global_step,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.info("Checkpoint saved: %s (epoch=%d, metric=%.4f)", path, epoch, best_metric)

    return path


def load_checkpoint(
    path: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
    strict: bool = False,
) -> dict:
    """Load a training checkpoint.

    Parameters
    ----------
    path : str | Path
        Path to the checkpoint file.
    model : Optional[nn.Module]
        Model to load weights into.
    optimizer : Optional[Optimizer]
        Optimizer to restore state.
    scheduler : Optional
        Scheduler to restore state.
    device : str
        Device to map tensors to. Default "cpu".
    strict : bool
        Whether to require an exact match of parameter keys.

    Returns
    -------
    dict
        Full checkpoint dictionary.

    Raises
    ------
    FileNotFoundError
        If checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    logger.info("Loaded checkpoint: %s (epoch=%d)", path, checkpoint.get("epoch", -1))

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        logger.info("Model weights restored")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state restored")

    return checkpoint
