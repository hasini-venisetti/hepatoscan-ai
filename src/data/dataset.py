"""MONAI Dataset classes for HepatoScan AI.

Provides unified dataset loading across LiTS, HCC-TACE-Seg,
3D-IRCADb, Decathlon, and LLD-MMRI datasets with support for
multi-task labels (segmentation + classification + staging).

Usage:
    from src.data.dataset import HepatoScanDataset, get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from monai.data import CacheDataset, DataLoader, partition_dataset
    from monai.utils import set_determinism
except ImportError:
    CacheDataset = None
    DataLoader = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSIFICATION_LABELS = {
    "benign": 0,
    "malignant": 1,
}

MALIGNANT_SUBTYPES = {
    "HCC": 0,
    "ICC": 1,
    "Colorectal Metastasis": 2,
    "Other": 3,
}

BENIGN_SUBTYPES = {
    "Hemangioma": 0,
    "Cyst": 1,
    "FNH": 2,
    "Adenoma": 3,
}

BCLC_STAGES = {
    "Stage 0": 0,
    "Stage A": 1,
    "Stage B": 2,
    "Stage C": 3,
    "Stage D": 4,
}

RANDOM_SEED = 42

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data list construction
# ---------------------------------------------------------------------------


def build_data_list(
    images_dir: Path,
    masks_dir: Path,
    metadata_csv: Optional[Path] = None,
    split_file: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Build a list of data dictionaries for MONAI dataset.

    Parameters
    ----------
    images_dir : Path
        Directory containing preprocessed NIfTI images.
    masks_dir : Path
        Directory containing preprocessed NIfTI masks.
    metadata_csv : Optional[Path]
        Path to metadata CSV with classification/staging labels.
    split_file : Optional[Path]
        Path to JSON split file. If None, uses all available data.

    Returns
    -------
    list[dict[str, Any]]
        List of dicts with keys: 'image', 'label', and optional
        'diagnosis', 'malignancy', 'cancer_type', 'bclc_stage', etc.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    # Load metadata if available
    metadata = {}
    if metadata_csv is not None and Path(metadata_csv).exists():
        try:
            df = pd.read_csv(metadata_csv)
            for _, row in df.iterrows():
                pid = str(row.get("patient_id", ""))
                if pid:
                    metadata[pid] = row.to_dict()
            logger.info("Loaded metadata for %d patients", len(metadata))
        except Exception as e:
            logger.warning("Could not load metadata CSV: %s", e)

    # Load split file if provided
    split_cases = None
    if split_file is not None and Path(split_file).exists():
        with open(split_file, "r") as f:
            split_data = json.load(f)
            split_cases = set(split_data.get("cases", []))
            logger.info("Split file specifies %d cases", len(split_cases))

    # Build data list
    image_files = sorted(images_dir.glob("*.nii*"))
    data_list = []

    for img_path in image_files:
        patient_id = img_path.stem.replace(".nii", "").replace(".gz", "")

        # Skip if not in split
        if split_cases is not None and len(split_cases) > 0 and patient_id not in split_cases:
            continue

        # Find corresponding mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            alt_names = [
                img_path.name.replace("volume", "segmentation"),
                img_path.name.replace("img", "label"),
            ]
            for alt in alt_names:
                candidate = masks_dir / alt
                if candidate.exists():
                    mask_path = candidate
                    break

        if not mask_path.exists():
            logger.warning("No mask for %s, skipping", img_path.name)
            continue

        entry = {
            "image": str(img_path),
            "label": str(mask_path),
            "patient_id": patient_id,
        }

        # Add classification labels from metadata
        meta = metadata.get(patient_id, {})
        if "malignancy" in meta and pd.notna(meta["malignancy"]):
            entry["malignancy"] = CLASSIFICATION_LABELS.get(str(meta["malignancy"]).lower(), -1)
        else:
            entry["malignancy"] = -1  # Unknown

        if "cancer_type" in meta and pd.notna(meta["cancer_type"]):
            cancer_type = str(meta["cancer_type"])
            if cancer_type in MALIGNANT_SUBTYPES:
                entry["cancer_subtype"] = MALIGNANT_SUBTYPES[cancer_type]
            elif cancer_type in BENIGN_SUBTYPES:
                entry["cancer_subtype"] = BENIGN_SUBTYPES[cancer_type]
            else:
                entry["cancer_subtype"] = -1
        else:
            entry["cancer_subtype"] = -1

        if "bclc_stage" in meta and pd.notna(meta["bclc_stage"]):
            entry["bclc_stage"] = BCLC_STAGES.get(str(meta["bclc_stage"]), -1)
        else:
            entry["bclc_stage"] = -1

        # Clinical features for staging head
        entry["afp_level"] = float(meta.get("afp_level", 0.0)) if pd.notna(meta.get("afp_level")) else 0.0
        entry["age"] = float(meta.get("age", 0.0)) if pd.notna(meta.get("age")) else 0.0

        data_list.append(entry)

    logger.info("Built data list with %d entries", len(data_list))
    return data_list


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class HepatoScanDataset:
    """Factory for MONAI CacheDataset instances with multi-task support.

    Parameters
    ----------
    data_list : list[dict]
        List of data dictionaries from build_data_list().
    transforms : Any
        MONAI Compose transforms to apply.
    cache_rate : float
        Fraction of data to cache in memory. Default 0.1.
    num_workers : int
        Number of data loading workers.

    Examples
    --------
    >>> data_list = build_data_list(images_dir, masks_dir, metadata_csv)
    >>> transforms = get_train_transforms(config)
    >>> dataset = HepatoScanDataset(data_list, transforms, cache_rate=0.1)
    >>> sample = dataset[0]
    """

    def __init__(
        self,
        data_list: list[dict[str, Any]],
        transforms: Any,
        cache_rate: float = 0.1,
        num_workers: int = 2,
    ) -> None:
        if CacheDataset is None:
            raise ImportError("MONAI is required. Install with: pip install monai[all]")

        self.data_list = data_list
        self.transforms = transforms
        self.cache_rate = cache_rate

        self._dataset = CacheDataset(
            data=data_list,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict:
        return self._dataset[index]

    @property
    def dataset(self) -> "CacheDataset":
        """Access the underlying MONAI CacheDataset."""
        return self._dataset


def compute_class_weights(data_list: list[dict], key: str = "malignancy") -> np.ndarray:
    """Compute inverse-frequency class weights for imbalanced classification.

    Parameters
    ----------
    data_list : list[dict]
        Data list with classification labels.
    key : str
        Key for the label in data dicts.

    Returns
    -------
    np.ndarray
        Class weights as n_total / (n_classes * n_per_class).
    """
    labels = [d[key] for d in data_list if d.get(key, -1) >= 0]
    if not labels:
        return np.array([1.0])

    unique, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    n_classes = len(unique)
    weights = n_total / (n_classes * counts.astype(np.float64))

    logger.info("Class weights for '%s': %s (counts: %s)", key, weights, counts)
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------


def get_dataloaders(
    config: dict,
    train_transforms: Any,
    val_transforms: Any,
) -> tuple:
    """Create train, validation, and test dataloaders.

    Parameters
    ----------
    config : dict
        Configuration dictionary with data paths and hyperparameters.
    train_transforms : Any
        MONAI transforms for training data.
    val_transforms : Any
        MONAI transforms for validation/test data.

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader) DataLoader instances.
    """
    if DataLoader is None:
        raise ImportError("MONAI is required. Install with: pip install monai[all]")

    set_determinism(seed=RANDOM_SEED)

    data_cfg = config.get("data", {})
    images_dir = Path(data_cfg.get("images_dir", "data/processed/images"))
    masks_dir = Path(data_cfg.get("masks_dir", "data/processed/masks"))
    metadata_csv = data_cfg.get("metadata_csv", "data/processed/metadata.csv")

    # Build full data list
    full_data = build_data_list(images_dir, masks_dir, Path(metadata_csv))

    if not full_data:
        logger.warning("No data found. Returning empty dataloaders.")
        return None, None, None

    # Split data
    train_ratio = data_cfg.get("train_split", 0.70)
    val_ratio = data_cfg.get("val_split", 0.15)

    # Use MONAI's partition_dataset for reproducible splits
    splits = partition_dataset(
        data=full_data,
        ratios=[train_ratio, val_ratio, 1.0 - train_ratio - val_ratio],
        shuffle=True,
        seed=RANDOM_SEED,
    )
    train_data, val_data, test_data = splits

    logger.info("Data split: train=%d, val=%d, test=%d", len(train_data), len(val_data), len(test_data))

    # Create datasets
    cache_rate = data_cfg.get("cache_rate", 0.1)
    batch_size = config.get("training", {}).get("batch_size", 1)

    train_ds = HepatoScanDataset(train_data, train_transforms, cache_rate=cache_rate)
    val_ds = HepatoScanDataset(val_data, val_transforms, cache_rate=1.0)
    test_ds = HepatoScanDataset(test_data, val_transforms, cache_rate=1.0)

    train_loader = DataLoader(train_ds.dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds.dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds.dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
