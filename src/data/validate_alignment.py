"""Image-mask alignment validation.

Validates that all preprocessed image-mask pairs have consistent
spatial metadata (spacing, origin, direction, shape) to prevent
silent data corruption during training.

Usage:
    python -m src.data.validate_alignment --data_dir data/processed
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPACING_TOLERANCE = 1e-3  # mm
ORIGIN_TOLERANCE = 1e-2   # mm
DIRECTION_TOLERANCE = 1e-4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_single_pair(
    image_path: Path,
    mask_path: Path,
    spacing_tol: float = SPACING_TOLERANCE,
    origin_tol: float = ORIGIN_TOLERANCE,
    direction_tol: float = DIRECTION_TOLERANCE,
) -> dict:
    """Validate spatial alignment between an image and its mask.

    Parameters
    ----------
    image_path : Path
        Path to the image NIfTI file.
    mask_path : Path
        Path to the mask NIfTI file.
    spacing_tol : float
        Tolerance for spacing comparison.
    origin_tol : float
        Tolerance for origin comparison.
    direction_tol : float
        Tolerance for direction matrix comparison.

    Returns
    -------
    dict
        Validation result with keys: 'valid', 'errors', 'warnings'.

    Raises
    ------
    ValueError
        If a critical mismatch is found with the patient ID.
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for alignment validation.")

    result = {"valid": True, "errors": [], "warnings": [], "image": str(image_path), "mask": str(mask_path)}

    try:
        img = sitk.ReadImage(str(image_path))
        msk = sitk.ReadImage(str(mask_path))
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to read files: {e}")
        return result

    patient_id = image_path.stem.replace(".nii", "")

    # Check 1: Shape match
    img_size = img.GetSize()
    msk_size = msk.GetSize()
    if img_size != msk_size:
        # For multi-channel images, the mask may have fewer dimensions
        # Compare only spatial dimensions
        img_spatial = img_size[:3]
        msk_spatial = msk_size[:3]
        if img_spatial != msk_spatial:
            result["valid"] = False
            result["errors"].append(
                f"Shape mismatch [{patient_id}]: image={img_size}, mask={msk_size}"
            )

    # Check 2: Spacing match
    img_spacing = img.GetSpacing()
    msk_spacing = msk.GetSpacing()
    for axis in range(min(3, len(img_spacing), len(msk_spacing))):
        if abs(img_spacing[axis] - msk_spacing[axis]) > spacing_tol:
            result["valid"] = False
            result["errors"].append(
                f"Spacing mismatch [{patient_id}] axis {axis}: "
                f"image={img_spacing[axis]:.6f}, mask={msk_spacing[axis]:.6f}"
            )

    # Check 3: Origin match
    img_origin = img.GetOrigin()
    msk_origin = msk.GetOrigin()
    for axis in range(min(3, len(img_origin), len(msk_origin))):
        if abs(img_origin[axis] - msk_origin[axis]) > origin_tol:
            result["valid"] = False
            result["errors"].append(
                f"Origin mismatch [{patient_id}] axis {axis}: "
                f"image={img_origin[axis]:.4f}, mask={msk_origin[axis]:.4f}"
            )

    # Check 4: Direction match
    img_direction = img.GetDirection()
    msk_direction = msk.GetDirection()
    if len(img_direction) == len(msk_direction):
        for i in range(len(img_direction)):
            if abs(img_direction[i] - msk_direction[i]) > direction_tol:
                result["valid"] = False
                result["errors"].append(
                    f"Direction mismatch [{patient_id}] element {i}: "
                    f"image={img_direction[i]:.6f}, mask={msk_direction[i]:.6f}"
                )
                break  # Report first direction mismatch only

    # Check 5: Mask label sanity
    msk_array = sitk.GetArrayFromImage(msk)
    unique_labels = np.unique(msk_array)
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        result["warnings"].append(f"Empty mask [{patient_id}]: all zeros")
    elif np.max(msk_array) > 10:
        result["warnings"].append(
            f"Unexpected label values [{patient_id}]: max={np.max(msk_array)}, "
            f"unique={unique_labels.tolist()}"
        )

    return result


def validate_dataset(
    images_dir: Path,
    masks_dir: Path,
    strict: bool = True,
) -> tuple[int, int, list[dict]]:
    """Validate all image-mask pairs in a dataset.

    Parameters
    ----------
    images_dir : Path
        Directory containing image NIfTI files.
    masks_dir : Path
        Directory containing mask NIfTI files.
    strict : bool
        If True, raise ValueError on any validation failure.

    Returns
    -------
    tuple[int, int, list[dict]]
        (num_valid, num_total, list of validation results)

    Raises
    ------
    ValueError
        If strict=True and any pair fails validation.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_files = sorted(images_dir.glob("*.nii*"))
    logger.info("Found %d images in %s", len(image_files), images_dir)

    results = []
    num_valid = 0
    num_total = 0

    for img_file in image_files:
        # Find corresponding mask
        mask_file = masks_dir / img_file.name
        if not mask_file.exists():
            # Try alternative naming conventions
            alt_name = img_file.name.replace("volume", "segmentation").replace("img", "label")
            mask_file = masks_dir / alt_name

        if not mask_file.exists():
            logger.warning("No mask found for %s, skipping", img_file.name)
            results.append({
                "valid": False,
                "errors": [f"No matching mask for {img_file.name}"],
                "warnings": [],
                "image": str(img_file),
                "mask": "NOT_FOUND",
            })
            num_total += 1
            continue

        num_total += 1
        result = validate_single_pair(img_file, mask_file)
        results.append(result)

        if result["valid"]:
            num_valid += 1
        else:
            for error in result["errors"]:
                logger.error(error)

        for warning in result.get("warnings", []):
            logger.warning(warning)

    # Print summary
    logger.info("=" * 60)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 60)
    logger.info("%d/%d pairs validated successfully", num_valid, num_total)

    if num_valid < num_total:
        failed = num_total - num_valid
        logger.error("%d pairs FAILED validation", failed)
        if strict:
            raise ValueError(
                f"{failed}/{num_total} image-mask pairs failed alignment validation. "
                "Fix these before training. See logs for details."
            )

    return num_valid, num_total, results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for alignment validation."""
    parser = argparse.ArgumentParser(
        description="Validate image-mask alignment for HepatoScan AI"
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("data/processed/images"),
        help="Directory containing preprocessed images",
    )
    parser.add_argument(
        "--masks_dir",
        type=Path,
        default=Path("data/processed/masks"),
        help="Directory containing preprocessed masks",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if any pair fails validation",
    )
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    num_valid, num_total, _ = validate_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        strict=args.strict,
    )

    if num_valid == num_total:
        logger.info("✓ All %d pairs passed validation", num_total)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
