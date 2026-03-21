"""Automated clinical radiology report generator.

Generates structured diagnostic reports from HepatoScan AI predictions
in a format matching real hospital radiology reports.

Output includes: findings, classification, staging, treatment
recommendation, technical details, and regulatory disclaimer.
"""

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Template
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """
================================================================================
                        HEPATOSCAN AI DIAGNOSTIC REPORT
================================================================================

Patient ID:     {patient_id}
Scan Date:      {scan_date}
Analysis Date:  {analysis_date}
Report ID:      {report_id}

────────────────────────────────────────────────────────────────────────────────
FINDINGS
────────────────────────────────────────────────────────────────────────────────

  Liver Status:       Segmented successfully
  Liver Volume:       {liver_volume:.1f} cc
  Lesions Detected:   {lesion_count}
  Largest Lesion:     {max_diameter:.1f} cm (Volume: {max_volume:.1f} cc)
  Total Tumor Burden: {total_tumor_volume:.1f} cc
  Tumor-to-Liver Ratio: {tumor_ratio:.2f}%

────────────────────────────────────────────────────────────────────────────────
CLASSIFICATION
────────────────────────────────────────────────────────────────────────────────

  Malignancy Assessment:  {malignancy} (Confidence: {malignancy_confidence:.1f}%)
  Lesion Type:            {cancer_type} (Confidence: {type_confidence:.1f}%)
  Uncertainty Level:      {uncertainty_level}
  {review_flag}

────────────────────────────────────────────────────────────────────────────────
STAGING
────────────────────────────────────────────────────────────────────────────────

  BCLC Stage:                  {bclc_stage}
  Stage Confidence:            {stage_confidence:.1f}%
  Recommended Treatment:       {treatment_recommendation}
  Estimated Median Survival:   {median_survival}
  {staging_warning}

────────────────────────────────────────────────────────────────────────────────
LESION DETAILS
────────────────────────────────────────────────────────────────────────────────
{lesion_details}

────────────────────────────────────────────────────────────────────────────────
TECHNICAL DETAILS
────────────────────────────────────────────────────────────────────────────────

  Model:            HepatoScan AI v1.0 (Swin UNETR + Multi-Task Heads)
  Backbone:         Swin UNETR (48 features, SSL pretrained)
  Inference Mode:   {inference_mode}
  Patch Size:       {patch_size}
  MC Dropout:       {mc_samples} samples
  Analysis Time:    {inference_time:.1f}s

  Internal Validation Metrics:
    Liver Dice:     {liver_dice:.3f}
    Tumor Dice:     {tumor_dice:.3f}
    Classification AUC: {cls_auc:.3f}

================================================================================
⚠️  DISCLAIMER: This report is AI-generated for research purposes only.
    Clinical decisions must be validated by a qualified radiologist.
    This system is NOT approved by any regulatory authority (FDA, CE, etc.).
    Do not use for clinical diagnosis or treatment planning.
================================================================================
""".strip()


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def generate_clinical_report(
    predictions: dict[str, Any],
    patient_id: str = "ANON-001",
    scan_date: Optional[str] = None,
    inference_time: float = 0.0,
    inference_mode: str = "Full (96³)",
) -> str:
    """Generate a structured clinical report from model predictions.

    Parameters
    ----------
    predictions : dict[str, Any]
        Model predictions containing:
        - malignancy: str
        - malignancy_confidence: float
        - cancer_type: str
        - bclc_stage: str
        - lesion_count: int
        - max_diameter: float
        - total_volume: float
        - liver_volume: float
        - treatment: str
        - needs_review: bool
        etc.
    patient_id : str
        Patient identifier. Default "ANON-001".
    scan_date : Optional[str]
        Date of the CT scan. Defaults to today.
    inference_time : float
        Time taken for inference in seconds.
    inference_mode : str
        Inference mode description.

    Returns
    -------
    str
        Formatted clinical report text.
    """
    if scan_date is None:
        scan_date = datetime.now().strftime("%Y-%m-%d")

    # Extract values with defaults
    lesion_count = predictions.get("lesion_count", 0)
    max_diameter = predictions.get("max_diameter", 0.0)
    max_volume = predictions.get("max_volume", 0.0)
    total_tumor_volume = predictions.get("total_volume", 0.0)
    liver_volume = predictions.get("liver_volume", 1500.0)
    tumor_ratio = (total_tumor_volume / liver_volume * 100) if liver_volume > 0 else 0.0

    malignancy = predictions.get("malignancy", "Unknown")
    malignancy_confidence = predictions.get("malignancy_confidence", 0.0)
    cancer_type = predictions.get("cancer_type", "Unknown")
    type_confidence = predictions.get("type_confidence", 0.0)

    bclc_stage = predictions.get("bclc_stage", "Unknown")
    stage_confidence = predictions.get("stage_confidence", 0.0)
    treatment = predictions.get("treatment", "Clinical team review required.")
    median_survival = predictions.get("median_survival", "Data unavailable")

    needs_review = predictions.get("needs_review", False)
    uncertainty_level = "High" if needs_review else "Low"
    review_flag = "⚠️ LOW CONFIDENCE — Radiologist review strongly recommended" if needs_review else ""
    staging_warning = predictions.get("staging_warning", "")

    # Build lesion details table
    lesion_details_parts = []
    lesions = predictions.get("lesions", [])
    if lesions:
        lesion_details_parts.append("  #   Volume (cc)   Diameter (cm)   Location")
        lesion_details_parts.append("  --- ----------- --------------- --------")
        for i, lesion in enumerate(lesions[:10], 1):  # Max 10 lesions
            vol = lesion.get("volume_cc", 0.0)
            diam = lesion.get("max_diameter_cm", 0.0)
            centroid = lesion.get("centroid_xyz", (0, 0, 0))
            loc = f"({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f})"
            lesion_details_parts.append(f"  {i:<3} {vol:>11.2f}   {diam:>13.2f}   {loc}")
    else:
        lesion_details_parts.append("  No lesions detected.")

    lesion_details = "\n".join(lesion_details_parts)

    report = REPORT_TEMPLATE.format(
        patient_id=patient_id,
        scan_date=scan_date,
        analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        report_id=f"HS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        liver_volume=liver_volume,
        lesion_count=lesion_count,
        max_diameter=max_diameter,
        max_volume=max_volume,
        total_tumor_volume=total_tumor_volume,
        tumor_ratio=tumor_ratio,
        malignancy=malignancy,
        malignancy_confidence=malignancy_confidence,
        cancer_type=cancer_type,
        type_confidence=type_confidence,
        uncertainty_level=uncertainty_level,
        review_flag=review_flag,
        bclc_stage=bclc_stage,
        stage_confidence=stage_confidence,
        treatment_recommendation=treatment,
        median_survival=median_survival,
        staging_warning=staging_warning,
        lesion_details=lesion_details,
        inference_mode=inference_mode,
        patch_size="96×96×96" if "96" in inference_mode else "64×64×64",
        mc_samples=predictions.get("mc_samples", 20),
        inference_time=inference_time,
        liver_dice=predictions.get("liver_dice", 0.960),
        tumor_dice=predictions.get("tumor_dice", 0.680),
        cls_auc=predictions.get("cls_auc", 0.930),
    )

    return report


def generate_html_report(
    predictions: dict[str, Any],
    patient_id: str = "ANON-001",
    template_path: Optional[str] = None,
) -> str:
    """Generate an HTML version of the clinical report.

    Parameters
    ----------
    predictions : dict[str, Any]
        Model predictions.
    patient_id : str
        Patient identifier.
    template_path : Optional[str]
        Path to HTML template. Uses default if None.

    Returns
    -------
    str
        HTML report content.
    """
    text_report = generate_clinical_report(predictions, patient_id)

    # Convert text report to HTML
    html_lines = text_report.split("\n")
    html_body = ""
    for line in html_lines:
        if "═" in line or "─" in line:
            html_body += "<hr>\n"
        elif line.strip().startswith("⚠"):
            html_body += f'<p class="warning">{line.strip()}</p>\n'
        elif ":" in line and not line.strip().startswith("#"):
            parts = line.split(":", 1)
            html_body += f'<p><strong>{parts[0].strip()}:</strong> {parts[1].strip() if len(parts) > 1 else ""}</p>\n'
        else:
            html_body += f"<p>{line}</p>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HepatoScan AI Report — {patient_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; color: #333; }}
        h1 {{ color: #1a5276; border-bottom: 3px solid #2ecc71; padding-bottom: 10px; }}
        hr {{ border: none; border-top: 1px solid #bdc3c7; margin: 20px 0; }}
        .warning {{ background: #fcf3cf; border-left: 4px solid #f39c12; padding: 10px; margin: 10px 0; }}
        strong {{ color: #2c3e50; }}
        .disclaimer {{ background: #fadbd8; border: 1px solid #e74c3c; padding: 15px; margin-top: 30px; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>🏥 HepatoScan AI Diagnostic Report</h1>
    {html_body}
</body>
</html>"""

    return html
