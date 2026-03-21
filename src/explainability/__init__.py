"""HepatoScan AI — Explainability module."""

from src.explainability.gradcam_3d import GradCAM3D
from src.explainability.report_generator import generate_clinical_report

__all__ = ["GradCAM3D", "generate_clinical_report"]
