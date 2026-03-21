"""Focal Loss for imbalanced classification.

Addresses the severe class imbalance in liver lesion datasets where
benign lesions vastly outnumber rare malignant subtypes like ICC.

References:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GAMMA = 2.0
DEFAULT_ALPHA = 0.25

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification with class imbalance.

    Down-weights well-classified examples and focuses training on
    hard-to-classify minority class samples.

    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher γ = more focus on hard examples.
        Default 2.0 (standard value from original paper).
    alpha : Optional[torch.Tensor]
        Per-class weight tensor. Shape (num_classes,).
        If None, uses equal weights.
    reduction : str
        Loss reduction: 'mean', 'sum', or 'none'. Default 'mean'.
    """

    def __init__(
        self,
        gamma: float = DEFAULT_GAMMA,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

        logger.info("FocalLoss: gamma=%.1f, alpha=%s, reduction=%s", gamma, alpha, reduction)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw logits from the model. Shape (B, C) for C classes.
        targets : torch.Tensor
            Ground truth class indices. Shape (B,) with values in [0, C-1].

        Returns
        -------
        torch.Tensor
            Scalar loss value (if reduction='mean' or 'sum') or
            per-sample loss (if reduction='none').
        """
        # Filter out invalid targets (label = -1 means unknown)
        valid_mask = targets >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)

        # Get probability of the correct class for each sample
        targets_one_hot = F.one_hot(targets.long(), num_classes=inputs.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B,)

        # Compute focal weight
        focal_weight = (1.0 - p_t) ** self.gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction="none")

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply class-specific alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets.long()]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
