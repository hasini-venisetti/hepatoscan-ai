"""Cosine annealing learning rate scheduler with linear warmup.

Implements: LR warmup for `warmup_epochs`, then cosine annealing
from `max_lr` to `min_lr` over the remaining epochs.

Usage:
    scheduler = CosineAnnealingWarmup(optimizer, warmup_epochs=10,
                                       max_epochs=300, min_lr=1e-6)
"""

import logging
import math
from typing import Optional

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class CosineAnnealingWarmup(LambdaLR):
    """Cosine annealing scheduler with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_epochs : int
        Number of warmup epochs with linearly increasing LR. Default 10.
    max_epochs : int
        Total number of training epochs. Default 300.
    min_lr_ratio : float
        Ratio of min_lr to initial_lr. Default 0.01 (LR decays to 1% of max).
    last_epoch : int
        Index of last epoch for resuming. Default -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 10,
        max_epochs: int = 300,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup: 0 → 1
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing: 1 → min_lr_ratio
                progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
                return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

        logger.info(
            "CosineAnnealingWarmup: warmup=%d, max_epochs=%d, min_lr_ratio=%.4f",
            warmup_epochs, max_epochs, min_lr_ratio,
        )


def build_scheduler(
    optimizer: optim.Optimizer,
    config: dict,
) -> CosineAnnealingWarmup:
    """Factory function to build scheduler from config.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    config : dict
        Training configuration with keys:
        warmup_epochs, max_epochs, learning_rate, min_lr.

    Returns
    -------
    CosineAnnealingWarmup
        Configured scheduler.
    """
    training_cfg = config.get("training", {})
    max_lr = training_cfg.get("learning_rate", 1e-4)
    min_lr = training_cfg.get("min_lr", 1e-6)
    min_lr_ratio = min_lr / max_lr if max_lr > 0 else 0.01

    return CosineAnnealingWarmup(
        optimizer=optimizer,
        warmup_epochs=training_cfg.get("warmup_epochs", 10),
        max_epochs=training_cfg.get("max_epochs", 300),
        min_lr_ratio=min_lr_ratio,
    )
