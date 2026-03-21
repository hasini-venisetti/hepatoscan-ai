"""NIfTI loading and saving helpers.

Convenience wrappers around nibabel and SimpleITK for consistent
NIfTI file I/O across the project.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import nibabel as nib
    NB_AVAILABLE = True
except ImportError:
    NB_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_nifti_volume(
    filepath: str | Path,
    return_affine: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI volume as a numpy array.

    Parameters
    ----------
    filepath : str | Path
        Path to .nii or .nii.gz file.
    return_affine : bool
        If True, also return the affine transformation matrix.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Volume data array, and optionally the 4×4 affine matrix.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If neither nibabel nor SimpleITK is available.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    if NB_AVAILABLE:
        img = nib.load(str(filepath))
        data = img.get_fdata().astype(np.float32)
        if return_affine:
            return data, img.affine
        return data
    elif SITK_AVAILABLE:
        img = sitk.ReadImage(str(filepath))
        data = sitk.GetArrayFromImage(img).astype(np.float32)
        if return_affine:
            # Construct affine from SimpleITK metadata
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            direction = img.GetDirection()
            affine = np.eye(4)
            for i in range(3):
                affine[i, i] = spacing[i]
                affine[i, 3] = origin[i]
            return data, affine
        return data
    else:
        raise ImportError("nibabel or SimpleITK required. Install with: pip install nibabel SimpleITK")


def save_nifti_volume(
    data: np.ndarray,
    filepath: str | Path,
    affine: Optional[np.ndarray] = None,
    spacing: Optional[tuple[float, float, float]] = None,
) -> Path:
    """Save a numpy array as a NIfTI volume.

    Parameters
    ----------
    data : np.ndarray
        Volume data to save.
    filepath : str | Path
        Output file path.
    affine : Optional[np.ndarray]
        4×4 affine matrix. Overrides spacing if provided.
    spacing : Optional[tuple]
        Voxel spacing in mm. Used if affine is None.

    Returns
    -------
    Path
        Path to the saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if NB_AVAILABLE:
        if affine is None:
            affine = np.eye(4)
            if spacing:
                affine[0, 0] = spacing[0]
                affine[1, 1] = spacing[1]
                affine[2, 2] = spacing[2]
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(filepath))
    elif SITK_AVAILABLE:
        img = sitk.GetImageFromArray(data)
        if spacing:
            img.SetSpacing(spacing)
        sitk.WriteImage(img, str(filepath))
    else:
        raise ImportError("nibabel or SimpleITK required.")

    logger.debug("Saved NIfTI: %s, shape=%s", filepath.name, data.shape)
    return filepath


def get_volume_info(filepath: str | Path) -> dict:
    """Get metadata about a NIfTI volume.

    Parameters
    ----------
    filepath : str | Path
        Path to the NIfTI file.

    Returns
    -------
    dict
        Volume metadata: shape, spacing, origin, dtype, etc.
    """
    filepath = Path(filepath)
    if NB_AVAILABLE:
        img = nib.load(str(filepath))
        header = img.header
        return {
            "shape": img.shape,
            "dtype": header.get_data_dtype(),
            "spacing": tuple(header.get_zooms()),
            "affine": img.affine.tolist(),
        }
    elif SITK_AVAILABLE:
        img = sitk.ReadImage(str(filepath))
        return {
            "shape": img.GetSize(),
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
        }
    else:
        raise ImportError("nibabel or SimpleITK required.")
