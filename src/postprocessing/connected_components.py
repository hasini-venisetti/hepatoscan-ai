"""Connected component analysis for lesion counting and measurement.

After segmentation, extracts per-lesion morphometric measurements:
volume, maximum diameter, centroid, and bounding box.

Pipeline:
1. Sigmoid → binary mask (threshold=0.5)
2. Morphological closing (3×3×3 kernel)
3. Remove small components (< 50 voxels)
4. Label remaining components
5. Compute per-lesion metrics
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.5
MIN_COMPONENT_SIZE_VOXELS = 50
CLOSING_KERNEL_SIZE = 3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LesionMeasurement:
    """Morphometric measurements for a single lesion.

    Attributes
    ----------
    lesion_id : int
        Unique lesion identifier within the volume.
    volume_cc : float
        Lesion volume in cubic centimeters.
    max_diameter_cm : float
        Maximum diameter (longest bounding box axis) in centimeters.
    centroid_xyz : tuple[float, float, float]
        Center of mass in voxel coordinates.
    bounding_box : tuple
        Bounding box as (min_z, max_z, min_y, max_y, min_x, max_x).
    num_voxels : int
        Number of voxels in the lesion.
    """

    lesion_id: int = 0
    volume_cc: float = 0.0
    max_diameter_cm: float = 0.0
    centroid_xyz: tuple = (0.0, 0.0, 0.0)
    bounding_box: tuple = (0, 0, 0, 0, 0, 0)
    num_voxels: int = 0


@dataclass
class LesionAnalysis:
    """Aggregate lesion analysis results.

    Attributes
    ----------
    lesion_count : int
        Total number of detected lesions.
    total_volume_cc : float
        Combined volume of all lesions.
    max_diameter_cm : float
        Diameter of the largest lesion.
    lesions : list[LesionMeasurement]
        Per-lesion measurement details.
    """

    lesion_count: int = 0
    total_volume_cc: float = 0.0
    max_diameter_cm: float = 0.0
    lesions: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def binarize_mask(
    prediction: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    apply_sigmoid: bool = True,
) -> np.ndarray:
    """Convert raw prediction to binary mask.

    Parameters
    ----------
    prediction : np.ndarray
        Raw model output (logits or probabilities).
    threshold : float
        Binarization threshold. Default 0.5.
    apply_sigmoid : bool
        Apply sigmoid before thresholding. Default True.

    Returns
    -------
    np.ndarray
        Binary mask (uint8).
    """
    if apply_sigmoid:
        prediction = 1.0 / (1.0 + np.exp(-prediction.astype(np.float64)))

    binary = (prediction >= threshold).astype(np.uint8)
    return binary


def morphological_closing(
    mask: np.ndarray,
    kernel_size: int = CLOSING_KERNEL_SIZE,
) -> np.ndarray:
    """Apply morphological closing to fill small holes.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    kernel_size : int
        Size of the structuring element. Default 3.

    Returns
    -------
    np.ndarray
        Closed mask.
    """
    structure = np.ones((kernel_size, kernel_size, kernel_size), dtype=np.uint8)
    closed = ndimage.binary_closing(mask, structure=structure).astype(np.uint8)
    return closed


def remove_small_components(
    mask: np.ndarray,
    min_size: int = MIN_COMPONENT_SIZE_VOXELS,
) -> np.ndarray:
    """Remove connected components smaller than min_size voxels.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    min_size : int
        Minimum component size in voxels. Default 50.

    Returns
    -------
    np.ndarray
        Cleaned mask with small components removed.
    """
    labeled, num_features = ndimage.label(mask)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        component = labeled == i
        if component.sum() >= min_size:
            cleaned[component] = 1

    removed = num_features - cleaned.max() if cleaned.max() > 0 else num_features
    if removed > 0:
        logger.debug("Removed %d small components (< %d voxels)", removed, min_size)

    return cleaned


def extract_lesion_measurements(
    prediction: np.ndarray,
    voxel_spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
    threshold: float = DEFAULT_THRESHOLD,
    apply_sigmoid: bool = True,
    min_size: int = MIN_COMPONENT_SIZE_VOXELS,
) -> LesionAnalysis:
    """Extract per-lesion measurements from segmentation output.

    Parameters
    ----------
    prediction : np.ndarray
        Raw segmentation output. Shape (D, H, W) or (1, D, H, W).
    voxel_spacing : tuple[float, float, float]
        Voxel spacing in mm (z, y, x).
    threshold : float
        Binarization threshold.
    apply_sigmoid : bool
        Whether to apply sigmoid before thresholding.
    min_size : int
        Minimum lesion size in voxels.

    Returns
    -------
    LesionAnalysis
        Complete lesion analysis with per-lesion measurements.
    """
    # Handle batch/channel dimensions
    if prediction.ndim == 4:
        prediction = prediction[0]  # Remove channel dim
    if prediction.ndim == 5:
        prediction = prediction[0, 0]  # Remove batch + channel

    # Step 1: Binarize
    binary = binarize_mask(prediction, threshold, apply_sigmoid)

    # Step 2: Morphological closing
    closed = morphological_closing(binary)

    # Step 3: Remove small components
    cleaned = remove_small_components(closed, min_size)

    # Step 4: Label remaining components
    labeled, num_lesions = ndimage.label(cleaned)

    # Step 5: Compute per-lesion measurements
    voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    voxel_volume_cc = voxel_volume_mm3 / 1000.0  # mm³ → cc

    lesions = []
    for i in range(1, num_lesions + 1):
        component = labeled == i
        n_voxels = int(component.sum())

        # Volume
        volume_cc = n_voxels * voxel_volume_cc

        # Centroid
        centroid = ndimage.center_of_mass(component)

        # Bounding box
        slices = ndimage.find_objects(labeled == i)
        if slices and slices[0] is not None:
            bbox = slices[0]
            # Max diameter: longest bounding box axis in cm
            extents = [
                (bbox[0].stop - bbox[0].start) * voxel_spacing[0] / 10.0,  # mm → cm
                (bbox[1].stop - bbox[1].start) * voxel_spacing[1] / 10.0,
                (bbox[2].stop - bbox[2].start) * voxel_spacing[2] / 10.0,
            ]
            max_diameter_cm = max(extents)
            bounding_box = (
                bbox[0].start, bbox[0].stop,
                bbox[1].start, bbox[1].stop,
                bbox[2].start, bbox[2].stop,
            )
        else:
            max_diameter_cm = 0.0
            bounding_box = (0, 0, 0, 0, 0, 0)

        lesion = LesionMeasurement(
            lesion_id=i,
            volume_cc=volume_cc,
            max_diameter_cm=max_diameter_cm,
            centroid_xyz=tuple(centroid),
            bounding_box=bounding_box,
            num_voxels=n_voxels,
        )
        lesions.append(lesion)

    # Sort by volume (largest first)
    lesions.sort(key=lambda x: x.volume_cc, reverse=True)

    analysis = LesionAnalysis(
        lesion_count=len(lesions),
        total_volume_cc=sum(l.volume_cc for l in lesions),
        max_diameter_cm=max((l.max_diameter_cm for l in lesions), default=0.0),
        lesions=lesions,
    )

    logger.info(
        "Lesion analysis: %d lesions, total volume=%.1f cc, max diameter=%.1f cm",
        analysis.lesion_count, analysis.total_volume_cc, analysis.max_diameter_cm,
    )
    return analysis
