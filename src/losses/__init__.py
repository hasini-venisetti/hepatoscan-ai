"""HepatoScan AI — Loss functions."""

from src.losses.focal_loss import FocalLoss
from src.losses.multitask_loss import UncertaintyWeightedLoss

__all__ = ["FocalLoss", "UncertaintyWeightedLoss"]
