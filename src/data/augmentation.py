"""MONAI augmentation / transform pipelines.

Defines training and validation transforms using MONAI's
dictionary-based transform API for 3D medical image data.

Augmentation strategy addresses:
- Domain shift between datasets (RandGaussianNoise, RandBiasField, RandGibbsNoise)
- Class imbalance (RandCropByPosNegLabeld with configurable pos ratio)
- Small tumor detection (high positive sample ratio)

Usage:
    from src.data.augmentation import get_train_transforms, get_val_transforms
"""

import logging
from typing import Any

try:
    from monai.transforms import (
        AddChanneld,
        Compose,
        CropForegroundd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandGaussianNoised,
        RandGaussianSmoothd,
        RandRotate90d,
        RandScaleIntensityd,
        RandShiftIntensityd,
        Spacingd,
        ToTensord,
    )

    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transform pipeline construction
# ---------------------------------------------------------------------------


def get_train_transforms(config: dict) -> Any:
    """Build the training augmentation pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary with augmentation and data settings.
        Expected keys under 'augmentation': rand_flip_prob, rand_rotate_prob,
        rand_noise_std, rand_scale_intensity, etc.
        Expected keys under 'data': patch_size, num_samples, voxel_spacing.

    Returns
    -------
    monai.transforms.Compose
        Composed transform pipeline for training.

    Raises
    ------
    ImportError
        If MONAI is not installed.
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for transforms. Install with: pip install monai[all]")

    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})

    patch_size = data_cfg.get("patch_size", [96, 96, 96])
    num_samples = data_cfg.get("num_samples", 4)
    voxel_spacing = data_cfg.get("voxel_spacing", [1.5, 1.5, 1.5])

    flip_prob = aug_cfg.get("rand_flip_prob", 0.5)
    rotate_prob = aug_cfg.get("rand_rotate_prob", 0.3)
    noise_std = aug_cfg.get("rand_noise_std", 0.01)

    transforms_list = [
        # --- Loading & Orientation ---
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=voxel_spacing,
            mode=("bilinear", "nearest"),
        ),

        # --- Intensity preprocessing ---
        # Note: HU clipping and normalization already done in preprocess.py,
        # but we apply mild normalization for robustness
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # --- Foreground cropping ---
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # --- Random patch extraction ---
        # 4 positive + 4 negative patches per volume
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),

        # --- Spatial augmentations ---
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=rotate_prob, max_k=3),

        # --- Intensity augmentations (domain shift robustness) ---
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=noise_std),
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.2,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),

        # --- Type conversion ---
        EnsureTyped(keys=["image", "label"]),
    ]

    logger.info(
        "Train transforms: patch_size=%s, num_samples=%d, flip_prob=%.2f",
        patch_size, num_samples, flip_prob,
    )
    return Compose(transforms_list)


def get_val_transforms(config: dict) -> Any:
    """Build the validation / test transform pipeline.

    No augmentation — only loading, orientation, spacing, and normalization.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    monai.transforms.Compose
        Composed transform pipeline for validation/test.
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for transforms. Install with: pip install monai[all]")

    data_cfg = config.get("data", {})
    voxel_spacing = data_cfg.get("voxel_spacing", [1.5, 1.5, 1.5])

    transforms_list = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=voxel_spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ]

    return Compose(transforms_list)


def get_inference_transforms(config: dict) -> Any:
    """Build transforms for inference (no labels required).

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    monai.transforms.Compose
        Composed transform pipeline for inference.
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for transforms. Install with: pip install monai[all]")

    data_cfg = config.get("data", {})
    voxel_spacing = data_cfg.get("voxel_spacing", [1.5, 1.5, 1.5])

    transforms_list = [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=voxel_spacing,
            mode="bilinear",
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ]

    return Compose(transforms_list)
