"""3D Grad-CAM for Swin UNETR explainability.

Generates class-activation heatmaps showing which regions of the CT
volume the model focused on for its classification decision.

Pipeline:
1. Register forward hook on the last Swin Transformer block
2. Forward pass through the model
3. Compute gradients of target class score w.r.t. feature map
4. Global average pool the gradients across spatial dimensions
5. Weight feature map channels by pooled gradients
6. Apply ReLU and upsample to original CT resolution
7. Normalize activation map to [0, 1]

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM3D:
    """3D Grad-CAM for Swin UNETR volumetric models.

    Parameters
    ----------
    model : nn.Module
        The HepatoScan model or Swin UNETR backbone.
    target_layer_name : str
        Name of the target layer to hook. Default uses last encoder block.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []

        # Find and hook the target layer
        target_layer = self._find_target_layer(target_layer_name)
        if target_layer is not None:
            self._register_hooks(target_layer)
            logger.info("GradCAM3D initialized on layer: %s", target_layer_name or "auto-detected")
        else:
            logger.warning("Could not find target layer for Grad-CAM")

    def _find_target_layer(self, name: Optional[str] = None) -> Optional[nn.Module]:
        """Find the target layer to attach hooks.

        Parameters
        ----------
        name : Optional[str]
            Layer name. If None, auto-detects the last encoder layer.

        Returns
        -------
        Optional[nn.Module]
            Target layer module.
        """
        if name is not None:
            for n, module in self.model.named_modules():
                if name in n:
                    return module

        # Auto-detect: try common Swin UNETR layer names
        candidate_names = [
            "swinViT.layers3",     # Last Swin Transformer stage
            "swinViT.layers",      # Any Swin stage
            "backbone.swinViT",    # Nested model
            "encoder",             # Generic encoder
        ]

        for candidate in candidate_names:
            for n, module in self.model.named_modules():
                if candidate in n:
                    return module

        # Fallback: use last conv/linear layer
        last_layer = None
        for module in self.model.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                last_layer = module
        return last_layer

    def _register_hooks(self, target_layer: nn.Module) -> None:
        """Register forward and backward hooks on the target layer.

        Parameters
        ----------
        target_layer : nn.Module
            Layer to hook.
        """
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = "classification",
    ) -> np.ndarray:
        """Generate Grad-CAM activation map.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input CT volume. Shape (1, C, D, H, W).
        target_class : Optional[int]
            Target class index for gradient computation.
            If None, uses the predicted class.
        task : str
            Which task's output to use: 'classification' or 'staging'.

        Returns
        -------
        np.ndarray
            Normalized activation map. Shape (D, H, W) in [0, 1].
        """
        self.model.eval()

        # Enable gradients for this pass
        input_tensor = input_tensor.detach().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if isinstance(output, dict):
            if task == "classification":
                logits = output.get("binary_logits", None)
            elif task == "staging":
                logits = output.get("stage_logits", None)
            else:
                logits = output.get("seg_logits", None)
        else:
            logits = output

        if logits is None:
            logger.warning("No logits found for task '%s', returning zero map", task)
            spatial = input_tensor.shape[2:]
            return np.zeros(spatial, dtype=np.float32)

        # Determine target class
        if target_class is None:
            if logits.dim() >= 2 and logits.shape[-1] > 1:
                target_class = logits.argmax(dim=-1).item()
            else:
                target_class = 0

        # Backward pass
        self.model.zero_grad()

        if logits.dim() >= 2 and logits.shape[-1] > 1:
            target_score = logits[0, target_class]
        else:
            target_score = logits.sum()

        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM hooks did not capture gradients/activations")
            spatial = input_tensor.shape[2:]
            return np.zeros(spatial, dtype=np.float32)

        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations

        # Handle different activation shapes
        if gradients.dim() == 5:
            # (B, C, D, H, W) — standard 3D feature map
            weights = gradients.mean(dim=[2, 3, 4], keepdim=True)  # GAP over spatial
            cam = (weights * activations).sum(dim=1, keepdim=True)  # Weight and sum channels
        elif gradients.dim() == 3:
            # (B, N, C) — transformer patch tokens
            weights = gradients.mean(dim=1, keepdim=True)  # Average over patches
            cam = (weights * activations).sum(dim=-1, keepdim=True)
            # Reshape to spatial dimensions
            n_patches = cam.shape[1]
            side = int(round(n_patches ** (1.0 / 3.0)))
            if side ** 3 == n_patches:
                cam = cam.view(1, 1, side, side, side)
            else:
                cam = cam.view(1, 1, -1, 1, 1)
        else:
            weights = gradients.mean(dim=list(range(2, gradients.dim())), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # Upsample to input resolution
        target_size = input_tensor.shape[2:]
        cam = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def cleanup(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.cleanup()
