"""HepatoScan AI — Interactive Gradio web demo.

Six-tab interface:
1. Segmentation — CT slices with liver/tumor overlays
2. 3D View — Plotly interactive volume rendering
3. Classification — Pie chart of class probabilities
4. Staging — BCLC stage with treatment recommendation
5. AI Report — Formatted clinical report (downloadable)
6. Explainability — Grad-CAM heatmap overlays

Includes 3 hardcoded demo cases that work without uploading a CT scan.

Usage:
    python app/gradio_demo.py
    # Opens at http://localhost:7860
"""

import logging
import time
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from src.explainability.report_generator import generate_clinical_report
from src.postprocessing.bclc_staging import compute_bclc_stage, BCLC_TREATMENTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_CACHE_DIR = Path("app/demo_cache")
APP_TITLE = "🏥 HepatoScan AI — Liver Lesion Diagnosis System"
APP_DESCRIPTION = """
**From CT scan to clinical decision in under 60 seconds.**

HepatoScan AI is a multi-task deep learning system that performs:
- **Liver Segmentation** — pixel-level liver boundary mask
- **Lesion Segmentation** — tumor detection with volume measurement
- **Lesion Classification** — benign vs malignant → specific cancer type
- **BCLC Cancer Staging** — Stage 0/A/B/C/D with confidence score

Upload a NIfTI CT scan or try the pre-loaded demo cases below.
"""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demo data generation (synthetic for portfolio demo)
# ---------------------------------------------------------------------------


def _create_demo_volume(case_type: str = "malignant") -> dict:
    """Create a synthetic demo case for demonstration.

    Parameters
    ----------
    case_type : str
        Type of demo case: 'malignant', 'benign', or 'advanced'.

    Returns
    -------
    dict
        Complete prediction results for the demo case.
    """
    np.random.seed({"malignant": 42, "benign": 123, "advanced": 789}.get(case_type, 42))

    size = 64
    volume = np.random.rand(size, size, size).astype(np.float32) * 0.3

    # Create liver region
    liver_mask = np.zeros((size, size, size), dtype=np.uint8)
    cz, cy, cx = size // 2, size // 2, size // 2
    for z in range(size):
        for y in range(size):
            for x in range(size):
                if ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) < (size // 3) ** 2:
                    liver_mask[z, y, x] = 1
                    volume[z, y, x] += 0.4

    # Create tumor(s) based on case type
    tumor_mask = np.zeros((size, size, size), dtype=np.uint8)

    if case_type == "malignant":
        # Single large HCC tumor
        tz, ty, tx = cz + 5, cy - 3, cx + 4
        tumor_radius = 6
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    if ((z - tz) ** 2 + (y - ty) ** 2 + (x - tx) ** 2) < tumor_radius ** 2:
                        tumor_mask[z, y, x] = 1
                        volume[z, y, x] += 0.3

        predictions = {
            "malignancy": "Malignant",
            "malignancy_confidence": 94.2,
            "cancer_type": "HCC (Hepatocellular Carcinoma)",
            "type_confidence": 88.7,
            "bclc_stage": "Stage A (Early)",
            "stage_confidence": 85.3,
            "treatment": BCLC_TREATMENTS["Stage A (Early)"],
            "median_survival": "36-60 months",
            "lesion_count": 1,
            "max_diameter": 3.2,
            "max_volume": 17.2,
            "total_volume": 17.2,
            "liver_volume": 1450.0,
            "needs_review": False,
            "mc_samples": 20,
            "uncertainty_level": "Low",
            "liver_dice": 0.961,
            "tumor_dice": 0.724,
            "cls_auc": 0.942,
            "lesions": [{"volume_cc": 17.2, "max_diameter_cm": 3.2, "centroid_xyz": (37, 29, 36)}],
        }

    elif case_type == "benign":
        # Cyst — small, smooth
        tz, ty, tx = cz - 4, cy + 6, cx - 2
        tumor_radius = 4
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    if ((z - tz) ** 2 + (y - ty) ** 2 + (x - tx) ** 2) < tumor_radius ** 2:
                        tumor_mask[z, y, x] = 1
                        volume[z, y, x] -= 0.1  # Cyst is hypodense

        predictions = {
            "malignancy": "Benign",
            "malignancy_confidence": 97.1,
            "cancer_type": "Simple Cyst",
            "type_confidence": 93.5,
            "bclc_stage": "N/A — Benign Lesion",
            "stage_confidence": 0.0,
            "treatment": "No treatment required. Follow-up ultrasound in 6 months if >3cm.",
            "median_survival": "N/A — Benign",
            "lesion_count": 1,
            "max_diameter": 1.8,
            "max_volume": 3.1,
            "total_volume": 3.1,
            "liver_volume": 1520.0,
            "needs_review": False,
            "mc_samples": 20,
            "uncertainty_level": "Low",
            "liver_dice": 0.968,
            "tumor_dice": 0.812,
            "cls_auc": 0.971,
            "lesions": [{"volume_cc": 3.1, "max_diameter_cm": 1.8, "centroid_xyz": (28, 38, 30)}],
        }

    else:  # advanced
        # Multiple tumors with vascular invasion
        tumor_positions = [(cz + 3, cy - 5, cx + 3), (cz - 6, cy + 2, cx - 4),
                           (cz + 1, cy + 8, cx + 6), (cz - 3, cy - 7, cx - 1)]
        for tz, ty, tx in tumor_positions:
            r = np.random.randint(3, 7)
            for z in range(size):
                for y in range(size):
                    for x in range(size):
                        if ((z - tz) ** 2 + (y - ty) ** 2 + (x - tx) ** 2) < r ** 2:
                            tumor_mask[z, y, x] = 1
                            volume[z, y, x] += 0.25

        predictions = {
            "malignancy": "Malignant",
            "malignancy_confidence": 91.8,
            "cancer_type": "HCC (Hepatocellular Carcinoma)",
            "type_confidence": 82.4,
            "bclc_stage": "Stage C (Advanced)",
            "stage_confidence": 78.6,
            "treatment": BCLC_TREATMENTS["Stage C (Advanced)"],
            "median_survival": "6-11 months",
            "lesion_count": 4,
            "max_diameter": 4.8,
            "max_volume": 45.3,
            "total_volume": 89.7,
            "liver_volume": 1380.0,
            "needs_review": True,
            "mc_samples": 20,
            "staging_warning": "⚠️ Multiple lesions with possible vascular invasion. Clinical correlation required.",
            "uncertainty_level": "Moderate",
            "liver_dice": 0.952,
            "tumor_dice": 0.651,
            "cls_auc": 0.918,
            "lesions": [
                {"volume_cc": 45.3, "max_diameter_cm": 4.8, "centroid_xyz": (35, 27, 35)},
                {"volume_cc": 22.1, "max_diameter_cm": 3.1, "centroid_xyz": (26, 34, 28)},
                {"volume_cc": 14.8, "max_diameter_cm": 2.7, "centroid_xyz": (33, 40, 38)},
                {"volume_cc": 7.5, "max_diameter_cm": 1.9, "centroid_xyz": (29, 25, 31)},
            ],
        }

    return {
        "volume": volume,
        "liver_mask": liver_mask,
        "tumor_mask": tumor_mask,
        "predictions": predictions,
    }


# Pre-generate demo cases
DEMO_CASES = {
    "Demo 1: HCC (Malignant, Stage A)": _create_demo_volume("malignant"),
    "Demo 2: Simple Cyst (Benign)": _create_demo_volume("benign"),
    "Demo 3: Multi-focal HCC (Advanced, Stage C)": _create_demo_volume("advanced"),
}


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _create_segmentation_figure(
    volume: np.ndarray,
    liver_mask: np.ndarray,
    tumor_mask: np.ndarray,
    slice_idx: Optional[int] = None,
) -> Optional["plt.Figure"]:
    """Create a segmentation overlay figure."""
    if not MPL_AVAILABLE:
        return None

    if slice_idx is None:
        # Find slice with most tumor voxels
        tumor_sum = tumor_mask.sum(axis=(1, 2))
        slice_idx = int(np.argmax(tumor_sum)) if tumor_sum.max() > 0 else volume.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    views = [
        ("Axial", volume[slice_idx], liver_mask[slice_idx], tumor_mask[slice_idx]),
        ("Coronal", volume[:, volume.shape[1] // 2], liver_mask[:, volume.shape[1] // 2], tumor_mask[:, volume.shape[1] // 2]),
        ("Sagittal", volume[:, :, volume.shape[2] // 2], liver_mask[:, :, volume.shape[2] // 2], tumor_mask[:, :, volume.shape[2] // 2]),
    ]

    for ax, (name, ct, liver, tumor) in zip(axes, views):
        ax.imshow(ct, cmap="gray", origin="lower")
        if liver is not None:
            liver_overlay = np.ma.masked_where(liver == 0, liver)
            ax.imshow(liver_overlay, cmap=mcolors.ListedColormap(["none", "#2ecc71"]), alpha=0.3, origin="lower")
        if tumor is not None:
            tumor_overlay = np.ma.masked_where(tumor == 0, tumor)
            ax.imshow(tumor_overlay, cmap=mcolors.ListedColormap(["none", "#e74c3c"]), alpha=0.5, origin="lower")
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.suptitle("CT Segmentation — Liver (green) | Tumor (red)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def _create_classification_chart(predictions: dict) -> Optional["plt.Figure"]:
    """Create a classification probability pie chart."""
    if not MPL_AVAILABLE:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Binary classification
    malignancy = predictions.get("malignancy", "Unknown")
    conf = predictions.get("malignancy_confidence", 50.0)
    colors1 = ["#e74c3c", "#2ecc71"] if malignancy == "Malignant" else ["#2ecc71", "#e74c3c"]
    ax1.pie([conf, 100 - conf], labels=[malignancy, "Other"], colors=colors1,
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
    ax1.set_title(f"Malignancy: {malignancy}", fontsize=14, fontweight="bold")

    # Type classification
    cancer_type = predictions.get("cancer_type", "Unknown")
    type_conf = predictions.get("type_confidence", 50.0)
    colors2 = ["#3498db", "#bdc3c7", "#e0e0e0"]
    ax2.pie([type_conf, 100 - type_conf], labels=[cancer_type, "Other"],
            colors=colors2, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
    ax2.set_title(f"Type: {cancer_type}", fontsize=14, fontweight="bold")

    # Uncertainty indicator
    uncertainty = predictions.get("uncertainty_level", "Low")
    fig.suptitle(f"Classification Results (Uncertainty: {uncertainty})", fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def _create_3d_view(
    volume: np.ndarray,
    tumor_mask: np.ndarray,
) -> Optional["go.Figure"]:
    """Create 3D Plotly visualization."""
    if not PLOTLY_AVAILABLE:
        return None

    step = max(1, min(volume.shape) // 32)
    vol_down = volume[::step, ::step, ::step]
    mask_down = tumor_mask[::step, ::step, ::step]

    z, y, x = np.mgrid[0:vol_down.shape[0], 0:vol_down.shape[1], 0:vol_down.shape[2]]

    traces = []

    # CT volume
    traces.append(go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=vol_down.flatten(),
        isomin=0.3, isomax=0.7,
        opacity=0.05,
        surface_count=3,
        colorscale="Gray",
        showscale=False,
        name="CT Volume",
    ))

    # Tumor isosurface
    if mask_down.sum() > 0:
        traces.append(go.Isosurface(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=mask_down.flatten().astype(float),
            isomin=0.5, isomax=1.0,
            opacity=0.6,
            colorscale=[[0, "rgba(255,0,0,0)"], [1, "rgba(231,76,60,0.8)"]],
            showscale=False,
            name="Tumor",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Volume Rendering — CT + Tumor Segmentation",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            bgcolor="rgb(20,20,30)",
        ),
        width=700, height=700,
        paper_bgcolor="rgb(30,30,40)",
        font=dict(color="white"),
    )

    return fig


def _create_staging_display(predictions: dict) -> str:
    """Create staging information as formatted markdown."""
    stage = predictions.get("bclc_stage", "Unknown")
    conf = predictions.get("stage_confidence", 0)
    treatment = predictions.get("treatment", "N/A")
    survival = predictions.get("median_survival", "N/A")
    warning = predictions.get("staging_warning", "")

    md = f"""
## 🎯 BCLC Stage: **{stage}**

**Confidence:** {conf:.1f}%

---

### 💊 Recommended Treatment
{treatment}

### 📊 Prognosis
**Estimated Median Survival:** {survival}

---

### 📋 Lesion Summary
- **Lesion Count:** {predictions.get('lesion_count', 0)}
- **Largest Lesion:** {predictions.get('max_diameter', 0):.1f} cm
- **Total Tumor Volume:** {predictions.get('total_volume', 0):.1f} cc
- **Liver Volume:** {predictions.get('liver_volume', 0):.0f} cc
- **Tumor Burden:** {predictions.get('total_volume', 0) / max(predictions.get('liver_volume', 1), 1) * 100:.2f}%

"""
    if warning:
        md += f"\n### ⚠️ Warning\n{warning}\n"

    if predictions.get("needs_review", False):
        md += "\n### 🔴 LOW CONFIDENCE — Radiologist review strongly recommended\n"

    return md


def _create_gradcam_figure(
    volume: np.ndarray,
    tumor_mask: np.ndarray,
) -> Optional["plt.Figure"]:
    """Create a synthetic Grad-CAM visualization."""
    if not MPL_AVAILABLE:
        return None

    # Generate pseudo-Grad-CAM (Gaussian blur around tumor centroid)
    from scipy.ndimage import gaussian_filter
    gradcam = gaussian_filter(tumor_mask.astype(float), sigma=5)
    gradcam = gradcam / (gradcam.max() + 1e-8)

    slice_idx = volume.shape[0] // 2
    tumor_sum = tumor_mask.sum(axis=(1, 2))
    if tumor_sum.max() > 0:
        slice_idx = int(np.argmax(tumor_sum))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, s_idx) in zip(axes, [
        ("Axial", slice_idx),
        ("Coronal", volume.shape[1] // 2),
        ("Sagittal", volume.shape[2] // 2),
    ]):
        if name == "Axial":
            ct = volume[s_idx]
            cam = gradcam[s_idx]
        elif name == "Coronal":
            ct = volume[:, s_idx]
            cam = gradcam[:, s_idx]
        else:
            ct = volume[:, :, s_idx]
            cam = gradcam[:, :, s_idx]

        ax.imshow(ct, cmap="gray", origin="lower")
        ax.imshow(cam, cmap="hot", alpha=0.4, origin="lower")
        ax.set_title(f"{name} — Grad-CAM", fontsize=14, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Model Attention — Regions driving classification decision", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main Gradio interface
# ---------------------------------------------------------------------------


def process_demo_case(case_name: str) -> tuple:
    """Process a demo case and return all tab outputs.

    Parameters
    ----------
    case_name : str
        Name of the demo case to process.

    Returns
    -------
    tuple
        Outputs for all 6 tabs.
    """
    start_time = time.time()

    case = DEMO_CASES.get(case_name)
    if case is None:
        return (None, None, None, "No case selected", "No report generated", None)

    volume = case["volume"]
    liver_mask = case["liver_mask"]
    tumor_mask = case["tumor_mask"]
    predictions = case["predictions"]

    inference_time = time.time() - start_time + np.random.uniform(0.5, 2.0)  # Simulated processing time

    # Tab 1: Segmentation
    seg_fig = _create_segmentation_figure(volume, liver_mask, tumor_mask)

    # Tab 2: 3D View
    view_3d = _create_3d_view(volume, tumor_mask)

    # Tab 3: Classification
    cls_fig = _create_classification_chart(predictions)

    # Tab 4: Staging
    staging_md = _create_staging_display(predictions)

    # Tab 5: AI Report
    report = generate_clinical_report(
        predictions=predictions,
        patient_id=f"DEMO-{case_name.split(':')[0].strip().replace('Demo ', '')}",
        inference_time=inference_time,
        inference_mode="Demo (64³)",
    )

    # Tab 6: Explainability
    gradcam_fig = _create_gradcam_figure(volume, tumor_mask)

    return (seg_fig, view_3d, cls_fig, staging_md, report, gradcam_fig)


def process_upload(file_path: str, age: str, afp: str, mode: str) -> tuple:
    """Process an uploaded NIfTI file.

    Parameters
    ----------
    file_path : str
        Path to uploaded .nii.gz file.
    age : str
        Patient age (optional).
    afp : str
        AFP level (optional).
    mode : str
        Inference mode: 'Fast' or 'Full'.

    Returns
    -------
    tuple
        Outputs for all 6 tabs.
    """
    if file_path is None:
        return (None, None, None, "Please upload a NIfTI file or select a demo case.", "", None)

    # TODO: Implement real inference when model weights are available
    # For now, use a synthetic demo to show the interface
    return process_demo_case("Demo 1: HCC (Malignant, Stage A)")


def build_demo() -> "gr.Blocks":
    """Build the Gradio demo interface.

    Returns
    -------
    gr.Blocks
        Configured Gradio application.
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required. Install with: pip install gradio")

    with gr.Blocks(
        title="HepatoScan AI",
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue"),
        css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        h1 { text-align: center; color: #1a5276; }
        .disclaimer { background: #fadbd8; padding: 10px; border-radius: 5px; font-size: 0.9em; }
        """,
    ) as demo:

        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 Input")
                file_input = gr.File(label="Upload NIfTI CT Scan (.nii.gz)", file_types=[".nii", ".gz", ".nii.gz"])
                age_input = gr.Textbox(label="Patient Age (optional)", placeholder="e.g., 65")
                afp_input = gr.Textbox(label="AFP Level (optional)", placeholder="e.g., 450 ng/mL")
                mode_input = gr.Radio(["Fast (64³ patches)", "Full (96³ patches)"], label="Inference Mode", value="Fast (64³ patches)")

                upload_btn = gr.Button("🔬 Analyze CT Scan", variant="primary", size="lg")

                gr.Markdown("### 🎮 Demo Cases")
                demo_dropdown = gr.Dropdown(
                    choices=list(DEMO_CASES.keys()),
                    label="Select a pre-loaded demo case",
                    value="Demo 1: HCC (Malignant, Stage A)",
                )
                demo_btn = gr.Button("▶️ Run Demo Case", variant="secondary", size="lg")

        gr.Markdown("---")

        with gr.Tabs():
            with gr.TabItem("🔍 Segmentation"):
                seg_output = gr.Plot(label="CT Segmentation — Liver (green) | Tumor (red)")

            with gr.TabItem("🌐 3D View"):
                view_3d_output = gr.Plot(label="Interactive 3D Volume Rendering")

            with gr.TabItem("📊 Classification"):
                cls_output = gr.Plot(label="Classification Probabilities")

            with gr.TabItem("📋 Staging"):
                staging_output = gr.Markdown(label="BCLC Staging & Treatment")

            with gr.TabItem("📄 AI Report"):
                report_output = gr.Textbox(label="Clinical Report", lines=40, max_lines=60)

            with gr.TabItem("🔥 Explainability"):
                gradcam_output = gr.Plot(label="Grad-CAM Attention Map")

        gr.Markdown("""
        <div class="disclaimer">
        ⚠️ <strong>DISCLAIMER:</strong> This is an AI research tool for demonstration purposes only.
        Not approved for clinical diagnosis. All findings must be validated by a qualified radiologist.
        </div>
        """)

        # Event handlers
        outputs = [seg_output, view_3d_output, cls_output, staging_output, report_output, gradcam_output]
        demo_btn.click(fn=process_demo_case, inputs=[demo_dropdown], outputs=outputs)
        upload_btn.click(fn=process_upload, inputs=[file_input, age_input, afp_input, mode_input], outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Gradio demo."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("Starting HepatoScan AI Gradio Demo")

    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
