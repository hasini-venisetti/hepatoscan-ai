"""Visualization utilities for CT slices, masks, and overlays.

Generates publication-quality figures for:
- CT slices with liver + tumor mask overlays
- Axial / coronal / sagittal views
- 3D volume rendering via Plotly
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Color maps for overlays
LIVER_COLOR = (0.0, 0.8, 0.0, 0.3)    # Green, semi-transparent
TUMOR_COLOR = (1.0, 0.0, 0.0, 0.5)    # Red, semi-transparent
GRADCAM_CMAP = "hot"


def plot_slice_with_overlay(
    ct_slice: np.ndarray,
    liver_mask: Optional[np.ndarray] = None,
    tumor_mask: Optional[np.ndarray] = None,
    gradcam: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 10),
) -> Optional["plt.Figure"]:
    """Plot a single CT slice with optional mask and Grad-CAM overlays.

    Parameters
    ----------
    ct_slice : np.ndarray
        2D CT slice (H, W).
    liver_mask : Optional[np.ndarray]
        2D liver segmentation mask.
    tumor_mask : Optional[np.ndarray]
        2D tumor segmentation mask.
    gradcam : Optional[np.ndarray]
        2D Grad-CAM activation map.
    title : str
        Figure title.
    save_path : Optional[str | Path]
        Path to save the figure. None to not save.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    Optional[plt.Figure]
        Figure object, or None if matplotlib unavailable.
    """
    if not MPL_AVAILABLE:
        logger.warning("matplotlib not available, skipping visualization")
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # CT slice in grayscale
    ax.imshow(ct_slice, cmap="gray", origin="lower")

    # Liver mask overlay (green)
    if liver_mask is not None:
        liver_overlay = np.ma.masked_where(liver_mask == 0, liver_mask)
        ax.imshow(liver_overlay, cmap=mcolors.ListedColormap(["none", "green"]),
                  alpha=0.3, origin="lower")

    # Tumor mask overlay (red)
    if tumor_mask is not None:
        tumor_overlay = np.ma.masked_where(tumor_mask == 0, tumor_mask)
        ax.imshow(tumor_overlay, cmap=mcolors.ListedColormap(["none", "red"]),
                  alpha=0.5, origin="lower")

    # Grad-CAM overlay
    if gradcam is not None:
        ax.imshow(gradcam, cmap=GRADCAM_CMAP, alpha=0.4, origin="lower")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved figure: %s", save_path)

    return fig


def plot_three_views(
    volume: np.ndarray,
    liver_mask: Optional[np.ndarray] = None,
    tumor_mask: Optional[np.ndarray] = None,
    slice_indices: Optional[tuple[int, int, int]] = None,
    title: str = "CT Volume — Axial / Coronal / Sagittal",
    save_path: Optional[str | Path] = None,
) -> Optional["plt.Figure"]:
    """Plot axial, coronal, and sagittal views of a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        3D CT volume (D, H, W).
    liver_mask : Optional[np.ndarray]
        3D liver mask.
    tumor_mask : Optional[np.ndarray]
        3D tumor mask.
    slice_indices : Optional[tuple[int, int, int]]
        Slice indices for (axial, coronal, sagittal). If None, uses mid-slices.
    title : str
        Figure title.
    save_path : Optional[str | Path]
        Path to save the figure.

    Returns
    -------
    Optional[plt.Figure]
        Figure object.
    """
    if not MPL_AVAILABLE:
        return None

    if slice_indices is None:
        slice_indices = (volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2)

    views = [
        ("Axial", volume[slice_indices[0], :, :],
         liver_mask[slice_indices[0], :, :] if liver_mask is not None else None,
         tumor_mask[slice_indices[0], :, :] if tumor_mask is not None else None),
        ("Coronal", volume[:, slice_indices[1], :],
         liver_mask[:, slice_indices[1], :] if liver_mask is not None else None,
         tumor_mask[:, slice_indices[1], :] if tumor_mask is not None else None),
        ("Sagittal", volume[:, :, slice_indices[2]],
         liver_mask[:, :, slice_indices[2]] if liver_mask is not None else None,
         tumor_mask[:, :, slice_indices[2]] if tumor_mask is not None else None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (view_name, ct, liver, tumor) in zip(axes, views):
        ax.imshow(ct, cmap="gray", origin="lower")
        if liver is not None:
            overlay = np.ma.masked_where(liver == 0, liver)
            ax.imshow(overlay, cmap=mcolors.ListedColormap(["none", "green"]), alpha=0.3, origin="lower")
        if tumor is not None:
            overlay = np.ma.masked_where(tumor == 0, tumor)
            ax.imshow(overlay, cmap=mcolors.ListedColormap(["none", "red"]), alpha=0.5, origin="lower")
        ax.set_title(view_name, fontsize=12)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def create_3d_plotly_visualization(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    title: str = "3D CT Volume",
) -> Optional[dict]:
    """Create interactive 3D Plotly visualization data.

    Parameters
    ----------
    volume : np.ndarray
        3D CT volume.
    mask : Optional[np.ndarray]
        3D segmentation mask for isosurface rendering.
    threshold : float
        Isosurface threshold.
    title : str
        Plot title.

    Returns
    -------
    Optional[dict]
        Plotly figure JSON data, or None if plotly unavailable.
    """
    try:
        import plotly.graph_objects as go

        traces = []

        # CT volume rendering (downsampled for performance)
        step = max(1, min(volume.shape) // 64)
        downsampled = volume[::step, ::step, ::step]

        z, y, x = np.mgrid[0:downsampled.shape[0], 0:downsampled.shape[1], 0:downsampled.shape[2]]

        traces.append(go.Volume(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=downsampled.flatten(),
            isomin=0.2, isomax=0.8,
            opacity=0.1,
            surface_count=5,
            colorscale="Gray",
            name="CT Volume",
        ))

        # Mask isosurface
        if mask is not None:
            mask_down = mask[::step, ::step, ::step]
            if mask_down.sum() > 0:
                traces.append(go.Isosurface(
                    x=x.flatten(), y=y.flatten(), z=z.flatten(),
                    value=mask_down.flatten().astype(float),
                    isomin=threshold, isomax=1.0,
                    opacity=0.6,
                    colorscale="Reds",
                    name="Lesion",
                ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            width=800, height=800,
        )

        return fig

    except ImportError:
        logger.warning("Plotly not available for 3D visualization")
        return None
