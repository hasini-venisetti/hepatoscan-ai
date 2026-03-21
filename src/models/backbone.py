"""Swin UNETR backbone with self-supervised pretrained weight loading.

Wraps MONAI's SwinUNETR architecture for 3D medical image segmentation.
Uses gradient checkpointing to fit within Kaggle P100 GPU (16GB VRAM).

References:
    Tang et al., "Self-Supervised Pre-Training of Swin Transformers for 3D
    Medical Image Analysis", CVPR 2022.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR

    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IMG_SIZE = (96, 96, 96)
DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 2  # background + liver+tumor
DEFAULT_FEATURE_SIZE = 48

# Google Drive ID for MONAI SSL pretrained weights
PRETRAINED_WEIGHTS_GDRIVE_ID = "1IabIoTesAFiBGnAmrxiUjPcKMfVSLqQF"
PRETRAINED_WEIGHTS_URL = f"https://drive.google.com/uc?id={PRETRAINED_WEIGHTS_GDRIVE_ID}"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backbone builder
# ---------------------------------------------------------------------------


def build_swin_unetr(
    img_size: tuple[int, int, int] = DEFAULT_IMG_SIZE,
    in_channels: int = DEFAULT_IN_CHANNELS,
    out_channels: int = DEFAULT_OUT_CHANNELS,
    feature_size: int = DEFAULT_FEATURE_SIZE,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    use_checkpoint: bool = True,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
) -> nn.Module:
    """Build a Swin UNETR backbone with optional pretrained weights.

    Parameters
    ----------
    img_size : tuple[int, int, int]
        Input volume spatial dimensions. Default (96, 96, 96).
    in_channels : int
        Number of input channels. Default 3 (multi-phase CT).
    out_channels : int
        Number of segmentation output classes. Default 2.
    feature_size : int
        Base feature dimension for Swin Transformer. Default 48.
    pretrained : bool
        Whether to load SSL pretrained weights. Default True.
    pretrained_path : Optional[str]
        Path to pretrained weights file. If None and pretrained=True,
        will attempt to download from MONAI model zoo.
    use_checkpoint : bool
        Enable gradient checkpointing for memory efficiency. Default True.
        REQUIRED for training on Kaggle P100 (16GB VRAM).
    drop_rate : float
        Dropout rate for the Swin Transformer. Default 0.0.
    attn_drop_rate : float
        Attention dropout rate. Default 0.0.

    Returns
    -------
    nn.Module
        Configured SwinUNETR model.

    Raises
    ------
    ImportError
        If MONAI is not installed.
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for SwinUNETR. Install with: pip install monai[all]")

    model = SwinUNETR(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )

    if pretrained:
        _load_pretrained_weights(model, pretrained_path)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "SwinUNETR built: img_size=%s, features=%d, params=%.2fM (trainable=%.2fM)",
        img_size, feature_size, total_params / 1e6, trainable_params / 1e6,
    )

    return model


def _load_pretrained_weights(
    model: nn.Module,
    weights_path: Optional[str] = None,
) -> None:
    """Load SSL pretrained weights into SwinUNETR.

    Parameters
    ----------
    model : nn.Module
        SwinUNETR model instance.
    weights_path : Optional[str]
        Path to saved weights file. If None, attempts download.
    """
    if weights_path is not None and Path(weights_path).exists():
        logger.info("Loading pretrained weights from %s", weights_path)
        try:
            weight = torch.load(weights_path, map_location="cpu", weights_only=False)
            model.load_from(weights=weight)
            logger.info("Successfully loaded pretrained weights")
        except Exception as e:
            logger.warning("Failed to load pretrained weights: %s. Training from scratch.", e)
    else:
        logger.warning(
            "Pretrained weights not found at '%s'. "
            "Download from: %s "
            "Training will start from random initialization.",
            weights_path, PRETRAINED_WEIGHTS_URL,
        )


def freeze_encoder(model: nn.Module) -> None:
    """Freeze the Swin UNETR encoder layers for transfer learning.

    Used in Phase 2 training (classification head only).

    Parameters
    ----------
    model : nn.Module
        SwinUNETR model.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if "swinViT" in name or "encoder" in name:
            param.requires_grad = False
            frozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Froze %d encoder parameters. Trainable params: %.2fM", frozen_count, trainable / 1e6)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters for end-to-end fine-tuning.

    Used in Phase 3 training.

    Parameters
    ----------
    model : nn.Module
        Model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    logger.info("Unfroze all parameters. Total trainable: %.2fM", total / 1e6)


def get_encoder_output_dim(feature_size: int = DEFAULT_FEATURE_SIZE) -> int:
    """Get the encoder output feature dimension for downstream heads.

    Parameters
    ----------
    feature_size : int
        Base feature size of the Swin UNETR.

    Returns
    -------
    int
        Dimension of the encoder's final feature vector after global avg pooling.
    """
    # SwinUNETR encoder output has feature_size * 16 channels at the bottleneck
    return feature_size * 16  # 48 * 16 = 768 for default config
