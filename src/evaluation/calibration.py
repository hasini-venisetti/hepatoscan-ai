"""Confidence calibration analysis.

Computes Expected Calibration Error (ECE) and generates reliability
diagrams for model confidence calibration assessment.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_NUM_BINS = 10


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = DEFAULT_NUM_BINS,
) -> dict:
    """Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    num_bins : int
        Number of bins for calibration. Default 10.

    Returns
    -------
    dict
        Calibration metrics: ece, bin_accuracies, bin_confidences, bin_counts.
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= low) & (y_prob < high)
        count = in_bin.sum()

        if count > 0:
            accuracy = y_true[in_bin].mean()
            confidence = y_prob[in_bin].mean()
            bin_accuracies.append(float(accuracy))
            bin_confidences.append(float(confidence))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

        bin_counts.append(int(count))

    # Weighted ECE
    total = sum(bin_counts)
    ece = 0.0
    if total > 0:
        for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
            ece += (count / total) * abs(acc - conf)

    return {
        "ece": float(ece),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = DEFAULT_NUM_BINS,
    save_path: Optional[str] = None,
    title: str = "Confidence Calibration",
) -> Optional[object]:
    """Generate a reliability diagram (calibration curve).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_prob : np.ndarray
        Predicted probabilities.
    num_bins : int
        Number of calibration bins.
    save_path : Optional[str]
        Path to save the figure.
    title : str
        Plot title.

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure, or None if unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for calibration plot")
        return None

    cal = expected_calibration_error(y_true, y_prob, num_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.bar(range(num_bins), cal["bin_accuracies"], width=0.8, alpha=0.6, color="steelblue", label="Accuracy")
    ax1.plot(range(num_bins), cal["bin_confidences"], "r--o", label="Mean Confidence", markersize=4)
    ax1.plot([0, num_bins - 1], [0, 1], "k--", alpha=0.3, label="Perfect Calibration")
    ax1.set_xlabel("Confidence Bin")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{title} (ECE = {cal['ece']:.4f})")
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Bin counts histogram
    ax2.bar(range(num_bins), cal["bin_counts"], color="gray", alpha=0.6)
    ax2.set_xlabel("Confidence Bin")
    ax2.set_ylabel("Count")
    ax2.set_title("Sample Distribution")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Calibration plot saved: %s", save_path)

    return fig
