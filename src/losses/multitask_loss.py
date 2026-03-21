"""Uncertainty-weighted multi-task loss (Kendall et al. 2018).

Instead of fixed λ weights, learns task-specific uncertainty parameters σ
that automatically balance the contribution of each task loss.

Total Loss = (1/2σ₁²) × Dice_CE_Loss(segmentation)
           + (1/2σ₂²) × Focal_Loss(classification)
           + (1/2σ₃²) × CrossEntropy(staging)
           + log(σ₁) + log(σ₂) + log(σ₃)

The model learns to weight each task based on its own uncertainty —
this is SOTA multi-task learning and prevents any single task from
dominating training.

References:
    Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics", CVPR 2018.
"""

import logging

import torch
import torch.nn as nn

try:
    from monai.losses import DiceCELoss

    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

from src.losses.focal_loss import FocalLoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_SIGMA = 1.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Uncertainty-Weighted Multi-Task Loss
# ---------------------------------------------------------------------------


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted multi-task loss combining segmentation,
    classification, and staging losses with learnable task weights.

    Parameters
    ----------
    num_seg_classes : int
        Number of segmentation output channels. Default 2.
    focal_gamma : float
        Focal loss gamma parameter. Default 2.0.
    cls_weights : list[float] | None
        Class weights for classification focal loss.
    staging_weights : list[float] | None
        Class weights for staging cross-entropy.
        BCLC Stage C and D are rare — triple their weight.
    initial_sigma : list[float]
        Initial values for learnable log-σ parameters.
        Default [1.0, 1.0, 1.0] for [seg, cls, staging].
    """

    def __init__(
        self,
        num_seg_classes: int = 2,
        focal_gamma: float = 2.0,
        cls_weights: list[float] | None = None,
        staging_weights: list[float] | None = None,
        initial_sigma: list[float] | None = None,
    ) -> None:
        super().__init__()

        # Segmentation loss: Dice + CE (MONAI)
        if MONAI_AVAILABLE:
            self.seg_loss = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                include_background=False,
            )
        else:
            self.seg_loss = nn.CrossEntropyLoss()

        # Classification loss: Focal Loss
        cls_alpha = torch.tensor(cls_weights, dtype=torch.float32) if cls_weights else None
        self.cls_loss = FocalLoss(gamma=focal_gamma, alpha=cls_alpha)

        # Staging loss: Cross-Entropy with class weights
        staging_weight_tensor = (
            torch.tensor(staging_weights, dtype=torch.float32)
            if staging_weights else None
        )
        self.staging_loss = nn.CrossEntropyLoss(
            weight=staging_weight_tensor,
            ignore_index=-1,
        )

        # Learnable log-σ² parameters (Kendall et al. 2018)
        if initial_sigma is None:
            initial_sigma = [DEFAULT_INITIAL_SIGMA] * 3

        # Initialize as log(σ²) so that σ² = exp(log_var)
        self.log_var_seg = nn.Parameter(torch.tensor(initial_sigma[0]).log())
        self.log_var_cls = nn.Parameter(torch.tensor(initial_sigma[1]).log())
        self.log_var_staging = nn.Parameter(torch.tensor(initial_sigma[2]).log())

        logger.info(
            "UncertaintyWeightedLoss: initial_sigma=%s, focal_gamma=%.1f",
            initial_sigma, focal_gamma,
        )

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_logits: torch.Tensor | None = None,
        cls_targets: torch.Tensor | None = None,
        staging_logits: torch.Tensor | None = None,
        staging_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute uncertainty-weighted multi-task loss.

        Parameters
        ----------
        seg_logits : torch.Tensor
            Segmentation output logits. Shape (B, C, D, H, W).
        seg_targets : torch.Tensor
            Segmentation ground truth. Shape (B, 1, D, H, W).
        cls_logits : Optional[torch.Tensor]
            Classification logits. Shape (B, num_classes).
        cls_targets : Optional[torch.Tensor]
            Classification targets. Shape (B,).
        staging_logits : Optional[torch.Tensor]
            Staging logits. Shape (B, 5).
        staging_targets : Optional[torch.Tensor]
            Staging targets. Shape (B,).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - 'total_loss': Scalar combined loss
            - 'seg_loss': Individual segmentation loss
            - 'cls_loss': Individual classification loss (0 if not provided)
            - 'staging_loss': Individual staging loss (0 if not provided)
            - 'sigma_seg': Current σ for segmentation task
            - 'sigma_cls': Current σ for classification task
            - 'sigma_staging': Current σ for staging task
        """
        # Compute individual losses
        loss_seg = self.seg_loss(seg_logits, seg_targets)

        loss_cls = torch.tensor(0.0, device=seg_logits.device)
        if cls_logits is not None and cls_targets is not None:
            valid_cls = cls_targets >= 0
            if valid_cls.any():
                loss_cls = self.cls_loss(cls_logits[valid_cls], cls_targets[valid_cls])

        loss_staging = torch.tensor(0.0, device=seg_logits.device)
        if staging_logits is not None and staging_targets is not None:
            valid_staging = staging_targets >= 0
            if valid_staging.any():
                loss_staging = self.staging_loss(
                    staging_logits[valid_staging],
                    staging_targets[valid_staging].long(),
                )

        # Uncertainty-weighted combination
        # L_total = (1/2σ²_i) * L_i + log(σ_i)
        # Using log-variance parameterization: σ² = exp(log_var)
        precision_seg = torch.exp(-self.log_var_seg)
        precision_cls = torch.exp(-self.log_var_cls)
        precision_staging = torch.exp(-self.log_var_staging)

        total_loss = (
            0.5 * precision_seg * loss_seg + 0.5 * self.log_var_seg
            + 0.5 * precision_cls * loss_cls + 0.5 * self.log_var_cls
            + 0.5 * precision_staging * loss_staging + 0.5 * self.log_var_staging
        )

        return {
            "total_loss": total_loss,
            "seg_loss": loss_seg.detach(),
            "cls_loss": loss_cls.detach(),
            "staging_loss": loss_staging.detach(),
            "sigma_seg": torch.exp(0.5 * self.log_var_seg).detach(),
            "sigma_cls": torch.exp(0.5 * self.log_var_cls).detach(),
            "sigma_staging": torch.exp(0.5 * self.log_var_staging).detach(),
        }
