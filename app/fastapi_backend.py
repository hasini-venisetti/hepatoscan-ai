"""HepatoScan AI — FastAPI REST API for inference.

Endpoints:
    POST /api/predict — Run inference on uploaded NIfTI CT scan
    GET  /api/health  — Health check
    GET  /api/info    — Model information

Usage:
    uvicorn app.fastapi_backend:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.explainability.report_generator import generate_clinical_report, generate_html_report
from src.postprocessing.bclc_staging import compute_bclc_stage

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HepatoScan AI API",
    description="REST API for liver lesion diagnosis from CT scans",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
) if FASTAPI_AVAILABLE else None

if app is not None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global model reference (loaded on startup)
_model = None
_device = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


if FASTAPI_AVAILABLE:

    @app.get("/api/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": _model is not None,
            "version": "1.0.0",
        }

    @app.get("/api/info")
    async def model_info() -> dict:
        """Model information endpoint."""
        return {
            "name": "HepatoScan AI",
            "version": "1.0.0",
            "backbone": "Swin UNETR (48 features, SSL pretrained)",
            "tasks": [
                "Liver Segmentation",
                "Lesion Segmentation",
                "Lesion Classification (Benign/Malignant + Subtype)",
                "BCLC Cancer Staging (0/A/B/C/D)",
            ],
            "input": "Multi-phase contrast-enhanced CT scan (NIfTI)",
            "output": "Segmentation masks, classification, staging, clinical report",
            "patch_size": [96, 96, 96],
            "supported_formats": [".nii", ".nii.gz"],
        }

    @app.post("/api/predict")
    async def predict(
        file: UploadFile = File(..., description="NIfTI CT scan (.nii.gz)"),
        patient_id: str = Form(default="ANON-001"),
        age: Optional[str] = Form(default=None),
        afp_level: Optional[str] = Form(default=None),
        inference_mode: str = Form(default="fast"),
    ) -> dict:
        """Run inference on an uploaded CT scan.

        Parameters
        ----------
        file : UploadFile
            NIfTI CT scan file.
        patient_id : str
            Patient identifier for the report.
        age : Optional[str]
            Patient age.
        afp_level : Optional[str]
            AFP level in ng/mL.
        inference_mode : str
            'fast' (64³ patches) or 'full' (96³ patches).

        Returns
        -------
        dict
            Complete prediction results with report.
        """
        start_time = time.time()

        # Validate file
        if not file.filename.endswith((".nii", ".nii.gz")):
            raise HTTPException(400, "Only NIfTI files (.nii, .nii.gz) are supported")

        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # TODO: Real inference when model weights are available
            # For now, return synthetic results for API testing
            predictions = {
                "malignancy": "Malignant",
                "malignancy_confidence": 91.5,
                "cancer_type": "HCC",
                "type_confidence": 85.3,
                "bclc_stage": "Stage A (Early)",
                "stage_confidence": 82.1,
                "treatment": "Resection / ablation / liver transplant",
                "median_survival": "36-60 months",
                "lesion_count": 1,
                "max_diameter": 3.0,
                "max_volume": 14.1,
                "total_volume": 14.1,
                "liver_volume": 1450.0,
                "needs_review": False,
                "mc_samples": 20,
            }

            inference_time = time.time() - start_time

            # Generate report
            report_text = generate_clinical_report(
                predictions=predictions,
                patient_id=patient_id,
                inference_time=inference_time,
                inference_mode="Full (96³)" if inference_mode == "full" else "Fast (64³)",
            )

            return {
                "status": "success",
                "patient_id": patient_id,
                "inference_time_seconds": round(inference_time, 2),
                "predictions": predictions,
                "report": report_text,
            }

        except Exception as e:
            logger.error("Inference failed: %s", str(e))
            raise HTTPException(500, f"Inference failed: {str(e)}")

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @app.post("/api/report/html")
    async def get_html_report(
        predictions: dict = None,
        patient_id: str = Form(default="ANON-001"),
    ) -> HTMLResponse:
        """Generate HTML clinical report."""
        if predictions is None:
            predictions = {"malignancy": "Unknown", "cancer_type": "Unknown", "bclc_stage": "Unknown"}

        html = generate_html_report(predictions, patient_id)
        return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the FastAPI server."""
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting HepatoScan AI API on port 8000")
    uvicorn.run("app.fastapi_backend:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
