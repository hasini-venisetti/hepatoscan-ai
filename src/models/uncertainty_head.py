"""Monte Carlo Dropout uncertainty quantification.

Implements MC Dropout for epistemic uncertainty estimation at inference time.
Keeps dropout ENABLED during inference, runs N forward passes, and computes
mean prediction, standard deviation, and confidence intervals.

This mimics how FDA-approved clinical AI systems (Viz.ai, Aidoc) handle
uncertainty to flag low-confidence cases for radiologist review.

References:
    Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing
    Model Uncertainty in Deep Learning", ICML 2016.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NUM_SAMPLES = 20
DEFAULT_CONFIDENCE_THRESHOLD = 0.3  # std threshold for flagging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MC Dropout Wrapper
# ---------------------------------------------------------------------------


class MCDropoutWrapper(nn.Module):
    """Monte Carlo Dropout wrapper for uncertainty quantification.

    Wraps any model that uses nn.Dropout layers and runs multiple stochastic
    forward passes during inference to estimate prediction uncertainty.

    Parameters
    ----------
    model : nn.Module
        The base model with nn.Dropout layers.
    num_samples : int
        Number of MC forward passes during inference. Default 20.
    confidence_threshold : float
        Standard deviation threshold above which a prediction is flagged
        as low confidence. Default 0.3.
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.confidence_threshold = confidence_threshold

    def _enable_dropout(self) -> None:
        """Enable dropout layers during inference for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, *args, **kwargs) -> dict:
        """Standard forward pass (delegates to wrapped model).

        Parameters
        ----------
        *args : Any
            Positional arguments for the wrapped model.
        **kwargs : Any
            Keyword arguments for the wrapped model.

        Returns
        -------
        dict
            Output from the wrapped model.
        """
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        *args,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run MC Dropout inference to estimate uncertainty.

        Performs multiple stochastic forward passes with dropout enabled
        and aggregates results to compute mean, std, and confidence flags.

        Parameters
        ----------
        *args : Any
            Positional arguments for the wrapped model.
        num_samples : Optional[int]
            Number of MC samples. If None, uses self.num_samples.
        **kwargs : Any
            Keyword arguments for the wrapped model.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - 'mean_prediction': Mean across MC samples for each output
            - 'std_prediction': Std dev across MC samples
            - 'confidence': 1 - normalized std (higher = more confident)
            - 'low_confidence_flag': Boolean tensor, True = needs review
            - 'all_samples': List of all individual forward pass outputs
        """
        n = num_samples if num_samples is not None else self.num_samples

        # Enable dropout for stochastic inference
        self._enable_dropout()

        all_outputs = []
        for i in range(n):
            output = self.model(*args, **kwargs)
            all_outputs.append(output)

        result = self._aggregate_mc_outputs(all_outputs)
        return result

    def _aggregate_mc_outputs(
        self,
        all_outputs: list[dict],
    ) -> dict[str, torch.Tensor]:
        """Aggregate MC Dropout outputs into mean, std, and confidence.

        Parameters
        ----------
        all_outputs : list[dict]
            List of output dictionaries from individual forward passes.

        Returns
        -------
        dict[str, torch.Tensor]
            Aggregated predictions with uncertainty estimates.
        """
        result = {}

        if not all_outputs:
            return result

        # Get all keys from first output
        sample_output = all_outputs[0]

        for key in sample_output:
            values = []
            for output in all_outputs:
                if key in output and isinstance(output[key], torch.Tensor):
                    values.append(output[key])

            if not values:
                continue

            stacked = torch.stack(values, dim=0)  # (N, ...)

            mean_pred = stacked.mean(dim=0)
            std_pred = stacked.std(dim=0)

            # Confidence: 1 - normalized std
            max_std = std_pred.max()
            if max_std > 0:
                normalized_std = std_pred / max_std
            else:
                normalized_std = torch.zeros_like(std_pred)

            confidence = 1.0 - normalized_std

            result[f"{key}_mean"] = mean_pred
            result[f"{key}_std"] = std_pred
            result[f"{key}_confidence"] = confidence

            # Flag low confidence regions/predictions
            low_conf = std_pred > self.confidence_threshold
            result[f"{key}_low_confidence"] = low_conf

        # Overall uncertainty score (scalar per batch element)
        if "binary_logits_std" in result:
            batch_uncertainty = result["binary_logits_std"].mean(dim=-1)
            result["overall_uncertainty"] = batch_uncertainty
            result["needs_review"] = batch_uncertainty > self.confidence_threshold

        return result


def compute_uncertainty_map(
    segmentation_samples: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute voxel-wise uncertainty map from MC segmentation samples.

    Parameters
    ----------
    segmentation_samples : list[torch.Tensor]
        List of N segmentation probability maps, each (B, C, D, H, W).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (mean_segmentation, uncertainty_map) — both (B, C, D, H, W).
        uncertainty_map contains the voxel-wise standard deviation.
    """
    stacked = torch.stack(segmentation_samples, dim=0)  # (N, B, C, D, H, W)

    mean_seg = stacked.mean(dim=0)     # (B, C, D, H, W)
    std_seg = stacked.std(dim=0)       # (B, C, D, H, W)

    logger.debug(
        "Uncertainty map: mean range [%.4f, %.4f], max std=%.4f",
        mean_seg.min().item(), mean_seg.max().item(), std_seg.max().item(),
    )

    return mean_seg, std_seg
