"""DICOM to NIfTI conversion with validation.

Converts DICOM series from HCC-TACE-Seg and 3D-IRCADb datasets
to NIfTI format, preserving spatial metadata and segmentation masks.
Handles DICOM-SEG segmentation objects and standard DICOM RT structures.

Usage:
    python -m src.data.convert_dicom --input_dir data/raw/hcc_tace --output_dir data/processed
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
    import pydicom
except ImportError:
    pydicom = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODALITIES = {"CT", "MR", "PT"}
MIN_SLICES_THRESHOLD = 10
DICOM_EXTENSIONS = {".dcm", ".dicom", ""}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------


def find_dicom_series(input_dir: Path) -> list[str]:
    """Discover all DICOM series UIDs in a directory tree.

    Parameters
    ----------
    input_dir : Path
        Root directory to search for DICOM files.

    Returns
    -------
    list[str]
        List of unique Series Instance UIDs found.
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for DICOM conversion. Install with: pip install SimpleITK")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(input_dir))
    logger.info("Found %d DICOM series in %s", len(series_ids), input_dir)
    return list(series_ids)


def read_dicom_series(input_dir: Path, series_id: str) -> "sitk.Image":
    """Read a single DICOM series into a SimpleITK Image.

    Parameters
    ----------
    input_dir : Path
        Directory containing DICOM files.
    series_id : str
        Series Instance UID to read.

    Returns
    -------
    sitk.Image
        The loaded 3D volume.

    Raises
    ------
    ValueError
        If the series contains fewer than MIN_SLICES_THRESHOLD slices.
    """
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(input_dir), series_id)

    if len(dicom_files) < MIN_SLICES_THRESHOLD:
        raise ValueError(
            f"Series {series_id} has only {len(dicom_files)} slices "
            f"(minimum {MIN_SLICES_THRESHOLD} required)."
        )

    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()
    logger.info(
        "Loaded series %s: size=%s, spacing=%s, origin=%s",
        series_id[:12],
        image.GetSize(),
        image.GetSpacing(),
        image.GetOrigin(),
    )
    return image


def read_dicom_seg(seg_path: Path, reference_image: "sitk.Image") -> "sitk.Image":
    """Read a DICOM-SEG segmentation object.

    Parameters
    ----------
    seg_path : Path
        Path to the DICOM-SEG file.
    reference_image : sitk.Image
        Reference image for spatial alignment verification.

    Returns
    -------
    sitk.Image
        Binary segmentation mask aligned to reference_image.

    Raises
    ------
    FileNotFoundError
        If seg_path does not exist.
    ValueError
        If segmentation cannot be parsed.
    """
    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

    try:
        # Try reading as a standard DICOM image (some SEGs are stored this way)
        seg_image = sitk.ReadImage(str(seg_path))
    except Exception:
        logger.warning("Could not read %s as standard DICOM, attempting pixel data extraction", seg_path)
        if pydicom is None:
            raise ImportError("pydicom is required for DICOM-SEG parsing.")

        ds = pydicom.dcmread(str(seg_path))
        pixel_data = ds.pixel_array.astype(np.uint8)

        # Create SimpleITK image from numpy array
        seg_image = sitk.GetImageFromArray(pixel_data)
        seg_image.CopyInformation(reference_image)

    return seg_image


def convert_to_nifti(
    image: "sitk.Image",
    output_path: Path,
    compress: bool = True,
) -> Path:
    """Write a SimpleITK Image to NIfTI format.

    Parameters
    ----------
    image : sitk.Image
        The 3D volume to save.
    output_path : Path
        Output file path (will add .nii.gz if compress=True).
    compress : bool
        Whether to use gzip compression. Default True.

    Returns
    -------
    Path
        Path to the saved NIfTI file.
    """
    output_path = Path(output_path)
    if compress and not str(output_path).endswith(".nii.gz"):
        output_path = output_path.with_suffix(".nii.gz")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_path))
    logger.info("Saved NIfTI: %s", output_path)
    return output_path


def validate_converted_pair(
    image_path: Path,
    mask_path: Path,
) -> bool:
    """Validate that a converted image-mask pair has consistent metadata.

    Parameters
    ----------
    image_path : Path
        Path to the image NIfTI file.
    mask_path : Path
        Path to the mask NIfTI file.

    Returns
    -------
    bool
        True if the pair is valid.

    Raises
    ------
    ValueError
        If spatial metadata does not match.
    """
    img = sitk.ReadImage(str(image_path))
    msk = sitk.ReadImage(str(mask_path))

    # Check size
    if img.GetSize() != msk.GetSize():
        raise ValueError(
            f"Size mismatch for {image_path.name}: "
            f"image={img.GetSize()}, mask={msk.GetSize()}"
        )

    # Check spacing (with tolerance)
    spacing_tolerance = 1e-3
    for i, (s_img, s_msk) in enumerate(zip(img.GetSpacing(), msk.GetSpacing())):
        if abs(s_img - s_msk) > spacing_tolerance:
            raise ValueError(
                f"Spacing mismatch in axis {i} for {image_path.name}: "
                f"image={s_img}, mask={s_msk}"
            )

    # Check origin
    origin_tolerance = 1e-2
    for i, (o_img, o_msk) in enumerate(zip(img.GetOrigin(), msk.GetOrigin())):
        if abs(o_img - o_msk) > origin_tolerance:
            raise ValueError(
                f"Origin mismatch in axis {i} for {image_path.name}: "
                f"image={o_img}, mask={o_msk}"
            )

    logger.debug("Validated pair: %s ↔ %s", image_path.name, mask_path.name)
    return True


def convert_hcc_tace_dataset(
    input_dir: Path,
    output_dir: Path,
) -> int:
    """Convert HCC-TACE-Seg dataset from DICOM to NIfTI.

    Handles multi-phase CT (arterial, portal venous, delayed) and
    DICOM-SEG tumor segmentation masks.

    Parameters
    ----------
    input_dir : Path
        Root directory of the HCC-TACE-Seg dataset.
    output_dir : Path
        Output directory for NIfTI files.

    Returns
    -------
    int
        Number of successfully converted patients.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    patient_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        logger.info("Processing patient: %s", patient_id)

        try:
            series_ids = find_dicom_series(patient_dir)

            if not series_ids:
                logger.warning("No DICOM series found for patient %s, skipping", patient_id)
                continue

            # Read the first (primary) series as the CT volume
            ct_image = read_dicom_series(patient_dir, series_ids[0])

            # Look for segmentation files
            seg_files = list(patient_dir.rglob("*.dcm")) + list(patient_dir.rglob("*SEG*"))
            seg_image = None
            for seg_file in seg_files:
                try:
                    seg_image = read_dicom_seg(seg_file, ct_image)
                    break
                except Exception:
                    continue

            # Save image
            img_path = convert_to_nifti(ct_image, images_dir / f"{patient_id}.nii.gz")

            # Save mask (or create empty mask if no segmentation found)
            if seg_image is not None:
                msk_path = convert_to_nifti(seg_image, masks_dir / f"{patient_id}.nii.gz")
            else:
                logger.warning("No segmentation found for patient %s, creating empty mask", patient_id)
                empty_mask = sitk.Image(ct_image.GetSize(), sitk.sitkUInt8)
                empty_mask.CopyInformation(ct_image)
                msk_path = convert_to_nifti(empty_mask, masks_dir / f"{patient_id}.nii.gz")

            # Validate the pair
            validate_converted_pair(img_path, msk_path)
            converted_count += 1

        except Exception as e:
            logger.error("Failed to convert patient %s: %s", patient_id, str(e))
            continue

    logger.info("Successfully converted %d/%d patients", converted_count, len(patient_dirs))
    return converted_count


def convert_ircadb_dataset(
    input_dir: Path,
    output_dir: Path,
) -> int:
    """Convert 3D-IRCADb-01 dataset from DICOM to NIfTI.

    Parameters
    ----------
    input_dir : Path
        Root directory of the 3D-IRCADb-01 dataset.
    output_dir : Path
        Output directory for NIfTI files.

    Returns
    -------
    int
        Number of successfully converted patients.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    patient_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        logger.info("Processing IRCADb patient: %s", patient_id)

        try:
            # IRCADb structure: patient_dir/PATIENT_DICOM/ for images
            #                   patient_dir/MASKS_DICOM/liver/ for liver mask
            #                   patient_dir/MASKS_DICOM/tumor*/ for tumor masks
            dicom_dir = patient_dir / "PATIENT_DICOM"
            masks_dicom_dir = patient_dir / "MASKS_DICOM"

            if not dicom_dir.exists():
                # Try alternative naming
                dicom_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and "PATIENT" in d.name.upper()]
                if dicom_dirs:
                    dicom_dir = dicom_dirs[0]
                else:
                    logger.warning("No PATIENT_DICOM directory for %s", patient_id)
                    continue

            # Read CT volume
            series_ids = find_dicom_series(dicom_dir)
            if not series_ids:
                logger.warning("No series found for IRCADb patient %s", patient_id)
                continue

            ct_image = read_dicom_series(dicom_dir, series_ids[0])
            img_path = convert_to_nifti(ct_image, images_dir / f"ircadb_{patient_id}.nii.gz")

            # Read liver mask
            liver_mask_dir = masks_dicom_dir / "liver" if masks_dicom_dir.exists() else None
            if liver_mask_dir and liver_mask_dir.exists():
                liver_series = find_dicom_series(liver_mask_dir)
                if liver_series:
                    liver_mask = read_dicom_series(liver_mask_dir, liver_series[0])
                    # Combine liver + tumor into single multi-label mask
                    mask_array = sitk.GetArrayFromImage(liver_mask)
                    mask_array = (mask_array > 0).astype(np.uint8)

                    # Look for tumor masks
                    if masks_dicom_dir.exists():
                        for tumor_dir in masks_dicom_dir.iterdir():
                            if tumor_dir.is_dir() and "tumor" in tumor_dir.name.lower():
                                try:
                                    tumor_series = find_dicom_series(tumor_dir)
                                    if tumor_series:
                                        tumor_mask = read_dicom_series(tumor_dir, tumor_series[0])
                                        tumor_array = sitk.GetArrayFromImage(tumor_mask)
                                        mask_array[tumor_array > 0] = 2  # Label 2 = tumor
                                except Exception as e:
                                    logger.warning("Could not read tumor mask in %s: %s", tumor_dir, e)

                    combined_mask = sitk.GetImageFromArray(mask_array)
                    combined_mask.CopyInformation(ct_image)
                    msk_path = convert_to_nifti(combined_mask, masks_dir / f"ircadb_{patient_id}.nii.gz")
                else:
                    empty_mask = sitk.Image(ct_image.GetSize(), sitk.sitkUInt8)
                    empty_mask.CopyInformation(ct_image)
                    msk_path = convert_to_nifti(empty_mask, masks_dir / f"ircadb_{patient_id}.nii.gz")
            else:
                empty_mask = sitk.Image(ct_image.GetSize(), sitk.sitkUInt8)
                empty_mask.CopyInformation(ct_image)
                msk_path = convert_to_nifti(empty_mask, masks_dir / f"ircadb_{patient_id}.nii.gz")

            validate_converted_pair(img_path, msk_path)
            converted_count += 1

        except Exception as e:
            logger.error("Failed to convert IRCADb patient %s: %s", patient_id, str(e))
            continue

    logger.info("Successfully converted %d/%d IRCADb patients", converted_count, len(patient_dirs))
    return converted_count


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for DICOM conversion."""
    parser = argparse.ArgumentParser(
        description="Convert DICOM datasets to NIfTI format for HepatoScan AI"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory containing DICOM files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for NIfTI files (default: data/processed)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hcc_tace", "ircadb", "auto"],
        default="auto",
        help="Dataset type to convert (default: auto-detect)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.dataset == "hcc_tace":
        convert_hcc_tace_dataset(args.input_dir, args.output_dir)
    elif args.dataset == "ircadb":
        convert_ircadb_dataset(args.input_dir, args.output_dir)
    else:
        # Auto-detect based on directory structure
        if (args.input_dir / "PATIENT_DICOM").exists() or any(
            (d / "PATIENT_DICOM").exists() for d in args.input_dir.iterdir() if d.is_dir()
        ):
            logger.info("Auto-detected IRCADb dataset format")
            convert_ircadb_dataset(args.input_dir, args.output_dir)
        else:
            logger.info("Auto-detected HCC-TACE-Seg dataset format")
            convert_hcc_tace_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
