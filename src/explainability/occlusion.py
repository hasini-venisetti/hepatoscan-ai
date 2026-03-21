"""MONAI OcclusionSensitivity wrapper for model explainability.

Provides an alternative to Grad-CAM for validation of attention regions
by systematically occluding patches and measuring prediction change.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_occlusion_sensitivity(
    model: nn.Module,
    input_tensor: torch.Tensor,
    patch_size: int = 8,
    stride: int = 4,
    baseline_value: float = 0.0,
    task: str = "classification",
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Compute occlusion sensitivity map for a 3D volume.

    Systematically occludes patches and measures the change in
    model confidence for the predicted class.

    Parameters
    ----------
    model : nn.Module
        The model to explain.
    input_tensor : torch.Tensor
        Input volume. Shape (1, C, D, H, W).
    patch_size : int
        Size of the occlusion patch cube. Default 8.
    stride : int
        Stride between occlusion positions. Default 4.
    baseline_value : float
        Value to fill occluded regions. Default 0.0 (black).
    task : str
        Task for sensitivity: 'classification' or 'staging'.
    device : Optional[torch.device]
        Device for computation.

    Returns
    -------
    np.ndarray
        Sensitivity map. Shape (D, H, W). Higher values = more important.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_tensor = input_tensor.to(device)

    # Get baseline prediction
    with torch.no_grad():
        baseline_output = model(input_tensor)
        if isinstance(baseline_output, dict):
            if task == "classification":
                logits = baseline_output.get("binary_logits", None)
            else:
                logits = baseline_output.get("stage_logits", None)
        else:
            logits = baseline_output

        if logits is None:
            return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        baseline_probs = torch.softmax(logits, dim=-1)
        predicted_class = baseline_probs.argmax(dim=-1).item()
        baseline_conf = baseline_probs[0, predicted_class].item()

    # Create sensitivity map
    spatial_shape = input_tensor.shape[2:]  # (D, H, W)
    sensitivity_map = np.zeros(spatial_shape, dtype=np.float32)
    count_map = np.zeros(spatial_shape, dtype=np.float32)

    for d in range(0, spatial_shape[0] - patch_size + 1, stride):
        for h in range(0, spatial_shape[1] - patch_size + 1, stride):
            for w in range(0, spatial_shape[2] - patch_size + 1, stride):
                # Create occluded input
                occluded = input_tensor.clone()
                occluded[:, :, d:d + patch_size, h:h + patch_size, w:w + patch_size] = baseline_value

                with torch.no_grad():
                    occ_output = model(occluded)
                    if isinstance(occ_output, dict):
                        if task == "classification":
                            occ_logits = occ_output.get("binary_logits", logits)
                        else:
                            occ_logits = occ_output.get("stage_logits", logits)
                    else:
                        occ_logits = occ_output

                    occ_probs = torch.softmax(occ_logits, dim=-1)
                    occ_conf = occ_probs[0, predicted_class].item()

                # Sensitivity = drop in confidence
                drop = baseline_conf - occ_conf
                sensitivity_map[d:d + patch_size, h:h + patch_size, w:w + patch_size] += drop
                count_map[d:d + patch_size, h:h + patch_size, w:w + patch_size] += 1

    # Average overlapping regions
    count_map[count_map == 0] = 1
    sensitivity_map /= count_map

    # Normalize
    s_min, s_max = sensitivity_map.min(), sensitivity_map.max()
    if s_max > s_min:
        sensitivity_map = (sensitivity_map - s_min) / (s_max - s_min)

    return sensitivity_map
