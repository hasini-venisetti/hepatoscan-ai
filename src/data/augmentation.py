"""MONAI augmentation / transform pipelines."""

import logging
from typing import Any

from monai.transforms import (
    EnsureChannelFirstd,
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
    SpatialPadd,
    Spacingd,
)

MONAI_AVAILABLE = True
logger = logging.getLogger(__name__)


def get_train_transforms(config: dict) -> Any:
    data_cfg = config.get("data", {})
    aug_cfg  = config.get("augmentation", {})

    patch_size    = data_cfg.get("patch_size", [96, 96, 96])
    num_samples   = data_cfg.get("num_samples", 4)
    voxel_spacing = data_cfg.get("voxel_spacing", [1.5, 1.5, 1.5])
    flip_prob     = aug_cfg.get("rand_flip_prob", 0.5)
    rotate_prob   = aug_cfg.get("rand_rotate_prob", 0.3)
    noise_std     = aug_cfg.get("rand_noise_std", 0.01)

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=voxel_spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            allow_smaller=True,
        ),
        # Pad volumes smaller than patch size BEFORE cropping
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode="constant",
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
            allow_smaller=False,
        ),
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=flip_prob, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=rotate_prob, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=noise_std),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5),
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms(config: dict) -> Any:
    data_cfg      = config.get("data", {})
    voxel_spacing = data_cfg.get("voxel_spacing", [1.5, 1.5, 1.5])
    patch_size    = data_cfg.get("patch_size", [96, 96, 96])

    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=voxel_spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            allow_smaller=True,
        ),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode="constant",
        ),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_inference_transforms(config: dict) -> Any:
    data_cfg      = config.get("data", {})
    voxel_spacing = data_cfg.get("voxel_spacing", [1.5, 1.5, 1.5])
    patch_size    = data_cfg.get("patch_size", [96, 96, 96])

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=voxel_spacing, mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        SpatialPadd(keys=["image"], spatial_size=patch_size, mode="constant"),
        EnsureTyped(keys=["image"]),
    ])
