"""Preprocessing pipeline for CT volumes.

Implements the full preprocessing chain:
1. Load NIfTI volume
2. Reorient to RAS+ standard orientation
3. Resample to 1.5mm isotropic voxel spacing
4. Clip Hounsfield Units to liver window [-100, 400]
5. Normalize to [0, 1] range
6. Stack multi-phase inputs (3-channel) or duplicate single-phase
7. Save as compressed NIfTI

Usage:
    python -m src.data.preprocess --input_dir data/raw/lits --output_dir data/processed --dataset lits
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import nibabel as nib
except ImportError:
    nib = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VOXEL_SPACING = (1.5, 1.5, 1.5)  # mm, isotropic
HU_MIN = -100
HU_MAX = 400
OUTPUT_DTYPE = np.float32
MASK_DTYPE = np.uint8

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------------------


def load_nifti(filepath: Path) -> "sitk.Image":
    """Load a NIfTI file using SimpleITK.

    Parameters
    ----------
    filepath : Path
        Path to the .nii or .nii.gz file.

    Returns
    -------
    sitk.Image
        Loaded 3D image volume.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If the file cannot be read.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    try:
        image = sitk.ReadImage(str(filepath))
        logger.debug("Loaded %s: size=%s, spacing=%s", filepath.name, image.GetSize(), image.GetSpacing())
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to read NIfTI file {filepath}: {e}") from e


def reorient_to_ras(image: "sitk.Image") -> "sitk.Image":
    """Reorient a 3D volume to RAS+ (Right-Anterior-Superior) standard orientation.

    Parameters
    ----------
    image : sitk.Image
        Input 3D volume in arbitrary orientation.

    Returns
    -------
    sitk.Image
        Volume reoriented to RAS+.
    """
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAS")
    reoriented = orienter.Execute(image)
    logger.debug("Reoriented to RAS+: direction=%s", reoriented.GetDirection())
    return reoriented


def resample_volume(
    image: "sitk.Image",
    target_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
    is_mask: bool = False,
) -> "sitk.Image":
    """Resample a volume to the target voxel spacing.

    Parameters
    ----------
    image : sitk.Image
        Input 3D volume.
    target_spacing : tuple[float, float, float]
        Target voxel spacing in mm. Default (1.5, 1.5, 1.5).
    is_mask : bool
        If True, use nearest-neighbor interpolation (for label maps).
        If False, use B-spline interpolation (for intensity images).

    Returns
    -------
    sitk.Image
        Resampled volume.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Compute new size based on target spacing
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(float(HU_MIN))

    resampled = resampler.Execute(image)
    logger.debug(
        "Resampled: %s → %s (spacing: %s → %s)",
        original_size, new_size, original_spacing, target_spacing,
    )
    return resampled


def clip_hounsfield_units(
    image_array: np.ndarray,
    hu_min: int = HU_MIN,
    hu_max: int = HU_MAX,
) -> np.ndarray:
    """Clip Hounsfield Unit values to liver window.

    Parameters
    ----------
    image_array : np.ndarray
        Raw CT intensity values in HU.
    hu_min : int
        Lower clip boundary. Default -100 HU.
    hu_max : int
        Upper clip boundary. Default 400 HU.

    Returns
    -------
    np.ndarray
        Clipped array.
    """
    return np.clip(image_array, hu_min, hu_max)


def normalize_to_unit_range(
    image_array: np.ndarray,
    hu_min: int = HU_MIN,
    hu_max: int = HU_MAX,
) -> np.ndarray:
    """Normalize clipped HU values to [0, 1] range.

    Parameters
    ----------
    image_array : np.ndarray
        Clipped CT intensity values.
    hu_min : int
        Minimum HU value (maps to 0.0).
    hu_max : int
        Maximum HU value (maps to 1.0).

    Returns
    -------
    np.ndarray
        Normalized array in [0, 1] with float32 dtype.
    """
    normalized = (image_array.astype(np.float32) - hu_min) / (hu_max - hu_min)
    return normalized.astype(OUTPUT_DTYPE)


def stack_multi_phase(
    phase_arrays: list[np.ndarray],
    num_channels: int = 3,
) -> np.ndarray:
    """Stack multi-phase CT arrays into a multi-channel volume.

    For HCC-TACE-Seg: 3 phases (arterial, portal, delayed) → 3 channels.
    For single-phase datasets: duplicate to fill 3 channels.

    Parameters
    ----------
    phase_arrays : list[np.ndarray]
        List of 3D arrays, one per phase. Each has shape (D, H, W).
    num_channels : int
        Target number of channels. Default 3.

    Returns
    -------
    np.ndarray
        Multi-channel volume with shape (C, D, H, W).
    """
    if len(phase_arrays) == 0:
        raise ValueError("At least one phase array is required")

    if len(phase_arrays) >= num_channels:
        # Use first num_channels phases
        stacked = np.stack(phase_arrays[:num_channels], axis=0)
    else:
        # Duplicate the available phases to fill channels
        repeated = []
        for i in range(num_channels):
            repeated.append(phase_arrays[i % len(phase_arrays)])
        stacked = np.stack(repeated, axis=0)

    logger.debug("Stacked %d phases → shape %s", len(phase_arrays), stacked.shape)
    return stacked


def preprocess_single_volume(
    image_path: Path,
    mask_path: Optional[Path],
    output_image_path: Path,
    output_mask_path: Optional[Path],
    target_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
    hu_min: int = HU_MIN,
    hu_max: int = HU_MAX,
) -> dict:
    """Run the full preprocessing pipeline on a single volume.

    Parameters
    ----------
    image_path : Path
        Path to the input CT NIfTI file.
    mask_path : Optional[Path]
        Path to the segmentation mask NIfTI file. None if no mask.
    output_image_path : Path
        Output path for the preprocessed image.
    output_mask_path : Optional[Path]
        Output path for the preprocessed mask. None if no mask.
    target_spacing : tuple[float, float, float]
        Target voxel spacing in mm.
    hu_min : int
        Lower HU clip boundary.
    hu_max : int
        Upper HU clip boundary.

    Returns
    -------
    dict
        Processing metadata: original_spacing, new_size, etc.
    """
    # Step 1: Load
    image = load_nifti(image_path)

    # Step 2: Reorient to RAS+
    image = reorient_to_ras(image)

    # Step 3: Resample to isotropic spacing
    original_spacing = image.GetSpacing()
    image = resample_volume(image, target_spacing, is_mask=False)

    # Step 4–5: Clip HU and normalize
    image_array = sitk.GetArrayFromImage(image)  # Shape: (D, H, W)
    image_array = clip_hounsfield_units(image_array, hu_min, hu_max)
    image_array = normalize_to_unit_range(image_array, hu_min, hu_max)

    # Step 7: Duplicate single phase to 3 channels
    multi_channel = stack_multi_phase([image_array], num_channels=3)  # Shape: (3, D, H, W)

    # Save preprocessed image
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as NIfTI using nibabel for multi-channel support
    if nib is not None:
        # Transpose to (H, W, D, C) for NIfTI convention
        nifti_array = np.transpose(multi_channel, (2, 3, 1, 0))  # (H, W, D, C)
        affine = np.eye(4)
        affine[0, 0] = target_spacing[0]
        affine[1, 1] = target_spacing[1]
        affine[2, 2] = target_spacing[2]
        nifti_img = nib.Nifti1Image(nifti_array, affine)
        nib.save(nifti_img, str(output_image_path))
    else:
        # Fallback: save first channel only via SimpleITK
        out_img = sitk.GetImageFromArray(multi_channel[0])
        out_img.SetSpacing(target_spacing)
        sitk.WriteImage(out_img, str(output_image_path))

    metadata = {
        "image_path": str(output_image_path),
        "original_spacing": original_spacing,
        "target_spacing": target_spacing,
        "output_shape": multi_channel.shape,
    }

    # Process mask if provided
    if mask_path is not None and output_mask_path is not None:
        mask = load_nifti(mask_path)
        mask = reorient_to_ras(mask)
        mask = resample_volume(mask, target_spacing, is_mask=True)

        mask_array = sitk.GetArrayFromImage(mask).astype(MASK_DTYPE)

        output_mask_path = Path(output_mask_path)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        if nib is not None:
            mask_nifti = np.transpose(mask_array, (1, 2, 0))  # (H, W, D)
            affine = np.eye(4)
            affine[0, 0] = target_spacing[0]
            affine[1, 1] = target_spacing[1]
            affine[2, 2] = target_spacing[2]
            nifti_mask = nib.Nifti1Image(mask_nifti, affine)
            nib.save(nifti_mask, str(output_mask_path))
        else:
            out_msk = sitk.GetImageFromArray(mask_array)
            out_msk.SetSpacing(target_spacing)
            sitk.WriteImage(out_msk, str(output_mask_path))

        metadata["mask_path"] = str(output_mask_path)
        metadata["mask_shape"] = mask_array.shape
        metadata["unique_labels"] = np.unique(mask_array).tolist()

    logger.info("Preprocessed: %s → %s", image_path.name, output_image_path.name)
    return metadata


def preprocess_dataset(
    input_images_dir: Path,
    input_masks_dir: Optional[Path],
    output_images_dir: Path,
    output_masks_dir: Optional[Path],
    target_spacing: tuple[float, float, float] = DEFAULT_VOXEL_SPACING,
) -> list[dict]:
    """Preprocess an entire dataset directory.

    Parameters
    ----------
    input_images_dir : Path
        Directory containing input NIfTI images.
    input_masks_dir : Optional[Path]
        Directory containing corresponding masks. None if no masks.
    output_images_dir : Path
        Output directory for preprocessed images.
    output_masks_dir : Optional[Path]
        Output directory for preprocessed masks.
    target_spacing : tuple[float, float, float]
        Target voxel spacing.

    Returns
    -------
    list[dict]
        List of processing metadata dicts.
    """
    input_images_dir = Path(input_images_dir)
    output_images_dir = Path(output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    if output_masks_dir is not None:
        output_masks_dir = Path(output_masks_dir)
        output_masks_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_images_dir.glob("*.nii*"))
    logger.info("Found %d images to preprocess", len(image_files))

    results = []
    for img_file in image_files:
        # Find corresponding mask
        mask_file = None
        if input_masks_dir is not None:
            mask_candidates = [
                input_masks_dir / img_file.name,
                input_masks_dir / img_file.name.replace("volume", "segmentation"),
                input_masks_dir / img_file.name.replace("img", "label"),
            ]
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_file = candidate
                    break

        output_img = output_images_dir / img_file.name
        output_msk = (output_masks_dir / img_file.name) if output_masks_dir else None

        try:
            meta = preprocess_single_volume(
                image_path=img_file,
                mask_path=mask_file,
                output_image_path=output_img,
                output_mask_path=output_msk,
                target_spacing=target_spacing,
            )
            results.append(meta)
        except Exception as e:
            logger.error("Failed to preprocess %s: %s", img_file.name, str(e))
            continue

    logger.info("Successfully preprocessed %d/%d volumes", len(results), len(image_files))
    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess CT volumes for HepatoScan AI")
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory with NIfTI images")
    parser.add_argument("--mask_dir", type=Path, default=None, help="Input directory with NIfTI masks")
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/images"), help="Output image dir")
    parser.add_argument("--output_mask_dir", type=Path, default=Path("data/processed/masks"), help="Output mask dir")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5], help="Target voxel spacing (mm)")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    preprocess_dataset(
        input_images_dir=args.input_dir,
        input_masks_dir=args.mask_dir,
        output_images_dir=args.output_dir,
        output_masks_dir=args.output_mask_dir,
        target_spacing=tuple(args.spacing),
    )


if __name__ == "__main__":
    main()
