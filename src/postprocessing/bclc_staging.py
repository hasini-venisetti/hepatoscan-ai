"""Rule-based BCLC staging with treatment recommendations.

Implements imaging-derived Barcelona Clinic Liver Cancer (BCLC) staging
based on lesion count, maximum diameter, vascular invasion, extrahepatic
spread, and Child-Pugh classification.

When clinical metadata is available, provides enhanced staging accuracy.
When metadata is missing, falls back to imaging-only staging with a
warning flag.

References:
    Reig et al., "BCLC strategy for prognosis prediction and treatment
    recommendation: The 2022 update", J Hepatol 2022.
"""

import logging
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BCLC_TREATMENTS = {
    "Stage 0 (Very Early)": (
        "Ablation or resection — curative intent. "
        "5-year survival >70%. Single nodule ≤2cm."
    ),
    "Stage A (Early)": (
        "Resection / ablation / liver transplant — curative. "
        "Milan criteria applicable. 5-year survival 50-70%."
    ),
    "Stage B (Intermediate)": (
        "TACE (trans-arterial chemoembolization) — palliative. "
        "Median survival 16 months. Multinodular, no vascular invasion."
    ),
    "Stage C (Advanced)": (
        "Sorafenib / Atezolizumab+Bevacizumab immunotherapy — systemic treatment. "
        "Vascular invasion or extrahepatic spread present."
    ),
    "Stage D (Terminal)": (
        "Best supportive care. Clinical team review required. "
        "End-stage liver function (Child-Pugh C) or ECOG PS >2."
    ),
}

BCLC_SURVIVAL = {
    "Stage 0 (Very Early)": ">60 months",
    "Stage A (Early)": "36-60 months",
    "Stage B (Intermediate)": "16-20 months",
    "Stage C (Advanced)": "6-11 months",
    "Stage D (Terminal)": "<3 months",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BCLCStagingResult:
    """BCLC staging result with treatment recommendation.

    Attributes
    ----------
    stage : str
        BCLC stage (e.g., "Stage A (Early)").
    confidence : str
        Confidence level: "high" (with clinical data) or "imaging-only".
    treatment : str
        Evidence-based treatment recommendation.
    median_survival : str
        Expected median survival for this stage.
    imaging_only : bool
        True if clinical metadata was not available.
    warning : str | None
        Warning message if staging is uncertain.
    """

    stage: str = "Unknown"
    confidence: str = "imaging-only"
    treatment: str = ""
    median_survival: str = ""
    imaging_only: bool = True
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Core staging function
# ---------------------------------------------------------------------------


def compute_bclc_stage(
    lesion_count: int,
    max_diameter_cm: float,
    has_vascular_invasion: bool = False,
    has_extrahepatic_spread: bool = False,
    child_pugh_score: Optional[str] = None,
    ecog_ps: Optional[int] = None,
    total_volume_cc: Optional[float] = None,
) -> BCLCStagingResult:
    """Compute BCLC stage from imaging and clinical features.

    Implements the 2022 BCLC update staging algorithm with graceful
    degradation when clinical metadata is not available.

    Parameters
    ----------
    lesion_count : int
        Number of detected liver lesions.
    max_diameter_cm : float
        Maximum diameter of the largest lesion in cm.
    has_vascular_invasion : bool
        Whether vascular invasion is detected. Default False.
    has_extrahepatic_spread : bool
        Whether extrahepatic spread is detected. Default False.
    child_pugh_score : Optional[str]
        Child-Pugh classification: "A", "B", or "C". None if unavailable.
    ecog_ps : Optional[int]
        ECOG Performance Status (0-4). None if unavailable.
    total_volume_cc : Optional[float]
        Total tumor volume in cc. Used for additional context.

    Returns
    -------
    BCLCStagingResult
        Complete staging result with treatment recommendation.
    """
    # Check if clinical data is available
    has_clinical = child_pugh_score is not None or ecog_ps is not None
    imaging_only = not has_clinical

    warning = None

    # Stage D: Terminal — requires clinical data to confirm
    if child_pugh_score == "C" or (ecog_ps is not None and ecog_ps > 2):
        stage = "Stage D (Terminal)"
    # Stage C: Advanced — vascular invasion or extrahepatic spread
    elif has_vascular_invasion or has_extrahepatic_spread:
        stage = "Stage C (Advanced)"
    # Stage 0: Very Early — single nodule ≤2cm
    elif lesion_count == 1 and max_diameter_cm <= 2.0:
        stage = "Stage 0 (Very Early)"
    # Stage A: Early — single ≤5cm OR up to 3 nodules each ≤3cm
    elif (lesion_count == 1 and max_diameter_cm <= 5.0) or \
         (lesion_count <= 3 and max_diameter_cm <= 3.0):
        stage = "Stage A (Early)"
    # Stage B: Intermediate — multinodular, no vascular invasion
    elif lesion_count > 3 and not has_vascular_invasion:
        stage = "Stage B (Intermediate)"
    # Edge cases
    elif lesion_count <= 3 and max_diameter_cm > 5.0:
        # Large single/few nodules without vascular invasion
        stage = "Stage A (Early)"
        warning = (
            "Large tumor(s) detected. Consider surgical evaluation. "
            "Vascular invasion assessment recommended."
        )
    else:
        stage = "Stage B (Intermediate)"
        warning = "Staging uncertain — additional clinical data recommended."

    # Add imaging-only warning
    if imaging_only and stage in ("Stage C (Advanced)", "Stage D (Terminal)"):
        if stage == "Stage D (Terminal)":
            stage = "Stage C (Advanced)"
            warning = (
                "⚠️ Stage D requires clinical confirmation (Child-Pugh, ECOG PS). "
                "Classified as Stage C based on imaging alone."
            )
        elif not has_vascular_invasion:
            warning = (
                "⚠️ Vascular invasion assessment is imaging-derived. "
                "Clinical correlation recommended."
            )

    if imaging_only and warning is None:
        warning = "Staging based on imaging features only. Clinical data not available."

    treatment = BCLC_TREATMENTS.get(stage, "Clinical team review required.")
    survival = BCLC_SURVIVAL.get(stage, "Data unavailable")

    result = BCLCStagingResult(
        stage=stage,
        confidence="high" if has_clinical else "imaging-only",
        treatment=treatment,
        median_survival=survival,
        imaging_only=imaging_only,
        warning=warning,
    )

    logger.info(
        "BCLC Staging: %s (lesions=%d, max_diam=%.1f cm, vascular=%s, confidence=%s)",
        stage, lesion_count, max_diameter_cm, has_vascular_invasion, result.confidence,
    )

    return result


def get_treatment_recommendation(stage: str) -> str:
    """Get evidence-based treatment recommendation for a BCLC stage.

    Parameters
    ----------
    stage : str
        BCLC stage string.

    Returns
    -------
    str
        Treatment recommendation text.
    """
    return BCLC_TREATMENTS.get(stage, "Clinical team review required.")


def get_survival_estimate(stage: str) -> str:
    """Get median survival estimate for a BCLC stage.

    Parameters
    ----------
    stage : str
        BCLC stage string.

    Returns
    -------
    str
        Estimated median survival.
    """
    return BCLC_SURVIVAL.get(stage, "Data unavailable")
