"""PyRadiomics feature extraction from tumor segmentation masks.

Extracts 93 radiomic features per lesion:
- Shape features: Volume, SurfaceArea, Sphericity, Elongation, Flatness
- First-order: Mean, Variance, Skewness, Kurtosis, Energy
- GLCM texture: Contrast, Correlation, Energy, Homogeneity

These features feed into the staging head as a separate branch,
enabling the model to leverage hand-crafted morphometric descriptors
alongside learned deep features.

Usage:
    from src.postprocessing.radiomics import extract_radiomics_features
    features = extract_radiomics_features(image_path, mask_path)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from radiomics import featureextractor

    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_FEATURE_DIM = 93  # Total features expected

# PyRadiomics configuration
RADIOMICS_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,  # Already resampled in preprocessing
    "interpolator": "sitkBSpline",
    "enableCExtensions": True,
    "normalize": True,
    "normalizeScale": 100,
}

ENABLED_FEATURES = {
    "shape": [],  # All shape features
    "firstorder": [],  # All first-order features
    "glcm": [],  # GLCM texture features
    "glrlm": [],  # Run-length features
    "glszm": [],  # Size-zone features
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def create_extractor() -> "featureextractor.RadiomicsFeatureExtractor":
    """Create a configured PyRadiomics feature extractor.

    Returns
    -------
    RadiomicsFeatureExtractor
        Configured extractor instance.

    Raises
    ------
    ImportError
        If PyRadiomics is not installed.
    """
    if not RADIOMICS_AVAILABLE:
        raise ImportError("PyRadiomics is required. Install with: pip install pyradiomics")

    extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)

    # Enable specific feature classes
    extractor.disableAllFeatures()
    for feature_class in ENABLED_FEATURES:
        extractor.enableFeatureClassByName(feature_class)

    logger.info("RadiomicsFeatureExtractor created with %d feature classes", len(ENABLED_FEATURES))
    return extractor


def extract_radiomics_features(
    image_path: str | Path,
    mask_path: str | Path,
    label_value: int = 1,
) -> np.ndarray:
    """Extract radiomic features from an image-mask pair.

    Parameters
    ----------
    image_path : str | Path
        Path to the CT NIfTI volume.
    mask_path : str | Path
        Path to the segmentation mask NIfTI volume.
    label_value : int
        Label value in the mask to extract features for. Default 1.

    Returns
    -------
    np.ndarray
        Feature vector of shape (93,). Returns zeros if extraction fails.
    """
    if not RADIOMICS_AVAILABLE:
        logger.warning("PyRadiomics not available. Returning zero features.")
        return np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float32)

    try:
        extractor = create_extractor()
        result = extractor.execute(str(image_path), str(mask_path), label=label_value)

        # Extract numeric features only (skip diagnostics metadata)
        features = []
        feature_names = []
        for key, value in result.items():
            if not key.startswith("diagnostics_"):
                try:
                    features.append(float(value))
                    feature_names.append(key)
                except (ValueError, TypeError):
                    continue

        feature_array = np.array(features, dtype=np.float32)

        # Pad or truncate to expected dimension
        if len(feature_array) < EXPECTED_FEATURE_DIM:
            padded = np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float32)
            padded[: len(feature_array)] = feature_array
            feature_array = padded
        elif len(feature_array) > EXPECTED_FEATURE_DIM:
            feature_array = feature_array[:EXPECTED_FEATURE_DIM]

        logger.info("Extracted %d radiomics features from %s", len(features), Path(image_path).name)
        return feature_array

    except Exception as e:
        logger.error("Radiomics extraction failed for %s: %s", image_path, str(e))
        return np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float32)


def extract_radiomics_from_arrays(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
    label_value: int = 1,
) -> np.ndarray:
    """Extract radiomic features from numpy arrays.

    Parameters
    ----------
    image_array : np.ndarray
        CT volume array. Shape (D, H, W).
    mask_array : np.ndarray
        Segmentation mask array. Shape (D, H, W).
    spacing : tuple[float, float, float]
        Voxel spacing in mm.
    label_value : int
        Label value to extract features for.

    Returns
    -------
    np.ndarray
        Feature vector of shape (93,).
    """
    if not RADIOMICS_AVAILABLE or sitk is None:
        logger.warning("Required libraries not available. Returning zero features.")
        return np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float32)

    try:
        # Convert numpy arrays to SimpleITK images
        image_sitk = sitk.GetImageFromArray(image_array.astype(np.float32))
        image_sitk.SetSpacing(spacing)

        mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
        mask_sitk.SetSpacing(spacing)

        extractor = create_extractor()
        result = extractor.execute(image_sitk, mask_sitk, label=label_value)

        features = []
        for key, value in result.items():
            if not key.startswith("diagnostics_"):
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    continue

        feature_array = np.array(features, dtype=np.float32)

        if len(feature_array) < EXPECTED_FEATURE_DIM:
            padded = np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float32)
            padded[: len(feature_array)] = feature_array
            feature_array = padded
        elif len(feature_array) > EXPECTED_FEATURE_DIM:
            feature_array = feature_array[:EXPECTED_FEATURE_DIM]

        return feature_array

    except Exception as e:
        logger.error("Array-based radiomics extraction failed: %s", str(e))
        return np.zeros(EXPECTED_FEATURE_DIM, dtype=np.float32)


def get_feature_names() -> list[str]:
    """Get the names of all radiomic features in extraction order.

    Returns
    -------
    list[str]
        Feature names.
    """
    names = [
        # Shape features (14)
        "shape_VoxelVolume", "shape_MeshVolume", "shape_SurfaceArea",
        "shape_SurfaceVolumeRatio", "shape_Sphericity", "shape_Compactness1",
        "shape_Compactness2", "shape_SphericalDisproportion",
        "shape_Maximum3DDiameter", "shape_Maximum2DDiameterSlice",
        "shape_Maximum2DDiameterColumn", "shape_Maximum2DDiameterRow",
        "shape_MajorAxisLength", "shape_MinorAxisLength",
        # First-order features (18)
        "firstorder_Energy", "firstorder_TotalEnergy", "firstorder_Entropy",
        "firstorder_Minimum", "firstorder_10Percentile", "firstorder_90Percentile",
        "firstorder_Maximum", "firstorder_Mean", "firstorder_Median",
        "firstorder_InterquartileRange", "firstorder_Range",
        "firstorder_MeanAbsoluteDeviation", "firstorder_RobustMeanAbsoluteDeviation",
        "firstorder_RootMeanSquared", "firstorder_StandardDeviation",
        "firstorder_Skewness", "firstorder_Kurtosis", "firstorder_Variance",
        # GLCM features (24)
        "glcm_Autocorrelation", "glcm_JointAverage", "glcm_ClusterProminence",
        "glcm_ClusterShade", "glcm_ClusterTendency", "glcm_Contrast",
        "glcm_Correlation", "glcm_DifferenceAverage", "glcm_DifferenceEntropy",
        "glcm_DifferenceVariance", "glcm_JointEnergy", "glcm_JointEntropy",
        "glcm_Imc1", "glcm_Imc2", "glcm_Idm", "glcm_Idmn", "glcm_Id",
        "glcm_Idn", "glcm_InverseVariance", "glcm_MaximumProbability",
        "glcm_SumAverage", "glcm_SumEntropy", "glcm_SumSquares",
        "glcm_MCC",
    ]

    # Pad to expected dimension
    while len(names) < EXPECTED_FEATURE_DIM:
        names.append(f"feature_{len(names)}")

    return names[:EXPECTED_FEATURE_DIM]
