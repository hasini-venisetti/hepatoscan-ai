"""Comprehensive evaluation metrics for segmentation, classification, and staging.

Segmentation: Dice, IoU, HD95, ASD, Sensitivity, Specificity
Classification: AUC-ROC, F1, Precision, Recall, Confusion Matrix, ECE
Staging: Accuracy, Weighted Kappa, Per-stage metrics
"""

import logging
from typing import Optional

import numpy as np

try:
    from scipy.spatial.distance import directed_hausdorff
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report, cohen_kappa_score,
        accuracy_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Segmentation Metrics
# ============================================================================


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Compute Dice Similarity Coefficient.

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask.
    target : np.ndarray
        Binary ground truth mask.
    smooth : float
        Smoothing factor to avoid division by zero.

    Returns
    -------
    float
        Dice coefficient in [0, 1].
    """
    pred_flat = pred.flatten().astype(bool)
    target_flat = target.flatten().astype(bool)

    intersection = np.logical_and(pred_flat, target_flat).sum()
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Compute Intersection over Union (Jaccard Index).

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask.
    target : np.ndarray
        Binary ground truth mask.
    smooth : float
        Smoothing factor.

    Returns
    -------
    float
        IoU score in [0, 1].
    """
    pred_flat = pred.flatten().astype(bool)
    target_flat = target.flatten().astype(bool)

    intersection = np.logical_and(pred_flat, target_flat).sum()
    union = np.logical_or(pred_flat, target_flat).sum()
    return float((intersection + smooth) / (union + smooth))


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute 95th percentile Hausdorff Distance.

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask.
    target : np.ndarray
        Binary ground truth mask.
    voxel_spacing : tuple
        Voxel spacing in mm for physical distance.

    Returns
    -------
    float
        HD95 in mm. Returns inf if either mask is empty.
    """
    pred_points = np.argwhere(pred > 0).astype(float)
    target_points = np.argwhere(target > 0).astype(float)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float("inf")

    # Scale by voxel spacing
    for i, s in enumerate(voxel_spacing):
        if i < pred_points.shape[1]:
            pred_points[:, i] *= s
            target_points[:, i] *= s

    # Compute distances from pred to target and vice versa
    from scipy.spatial import cKDTree

    tree_target = cKDTree(target_points)
    tree_pred = cKDTree(pred_points)

    dist_pred_to_target, _ = tree_target.query(pred_points)
    dist_target_to_pred, _ = tree_pred.query(target_points)

    all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
    return float(np.percentile(all_distances, 95))


def average_surface_distance(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute Average Surface Distance (ASD).

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction mask.
    target : np.ndarray
        Binary ground truth mask.
    voxel_spacing : tuple
        Voxel spacing in mm.

    Returns
    -------
    float
        ASD in mm.
    """
    pred_points = np.argwhere(pred > 0).astype(float)
    target_points = np.argwhere(target > 0).astype(float)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float("inf")

    for i, s in enumerate(voxel_spacing):
        if i < pred_points.shape[1]:
            pred_points[:, i] *= s
            target_points[:, i] *= s

    from scipy.spatial import cKDTree
    tree_target = cKDTree(target_points)
    tree_pred = cKDTree(pred_points)

    dist_p2t, _ = tree_target.query(pred_points)
    dist_t2p, _ = tree_pred.query(target_points)

    return float((dist_p2t.mean() + dist_t2p.mean()) / 2.0)


def segmentation_sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute voxel-level sensitivity (recall) for lesion detection.

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction.
    target : np.ndarray
        Binary ground truth.

    Returns
    -------
    float
        Sensitivity.
    """
    tp = np.logical_and(pred > 0, target > 0).sum()
    fn = np.logical_and(pred == 0, target > 0).sum()
    return float(tp / (tp + fn + 1e-8))


def segmentation_specificity(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute voxel-level specificity.

    Parameters
    ----------
    pred : np.ndarray
        Binary prediction.
    target : np.ndarray
        Binary ground truth.

    Returns
    -------
    float
        Specificity.
    """
    tn = np.logical_and(pred == 0, target == 0).sum()
    fp = np.logical_and(pred > 0, target == 0).sum()
    return float(tn / (tn + fp + 1e-8))


# ============================================================================
# Classification Metrics
# ============================================================================


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : Optional[np.ndarray]
        Predicted probabilities for AUC computation.
    class_names : Optional[list[str]]
        Class names for reporting.

    Returns
    -------
    dict
        All classification metrics.
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available for classification metrics")
        return {}

    # Filter invalid labels
    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_prob is not None:
        y_prob = y_prob[valid]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    if class_names and len(class_names) == len(per_class_f1):
        metrics["per_class_f1"] = {name: float(f) for name, f in zip(class_names, per_class_f1)}
    else:
        metrics["per_class_f1"] = per_class_f1.tolist()

    # AUC-ROC
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                metrics["auc_roc"] = float(roc_auc_score(y_true, prob))
            else:
                metrics["auc_roc_macro"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except ValueError as e:
            logger.warning("AUC computation failed: %s", e)

    return metrics


# ============================================================================
# Staging Metrics
# ============================================================================


def compute_staging_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: Optional[list[str]] = None,
) -> dict:
    """Compute BCLC staging-specific metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth staging labels (0-4).
    y_pred : np.ndarray
        Predicted staging labels.
    stage_names : Optional[list[str]]
        Names for each stage.

    Returns
    -------
    dict
        Staging metrics including accuracy and weighted kappa.
    """
    if not SKLEARN_AVAILABLE:
        return {}

    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {"staging_accuracy": 0.0, "weighted_kappa": 0.0}

    metrics = {
        "staging_accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Per-stage sensitivity
    if stage_names is None:
        stage_names = [f"Stage {i}" for i in range(5)]

    per_stage = {}
    for i, name in enumerate(stage_names):
        mask = y_true == i
        if mask.sum() > 0:
            correct = (y_pred[mask] == i).sum()
            per_stage[name] = {"sensitivity": float(correct / mask.sum()), "count": int(mask.sum())}

    metrics["per_stage"] = per_stage
    return metrics


# ============================================================================
# Aggregate
# ============================================================================


def compute_all_metrics(
    seg_pred: Optional[np.ndarray] = None,
    seg_target: Optional[np.ndarray] = None,
    cls_true: Optional[np.ndarray] = None,
    cls_pred: Optional[np.ndarray] = None,
    cls_prob: Optional[np.ndarray] = None,
    stage_true: Optional[np.ndarray] = None,
    stage_pred: Optional[np.ndarray] = None,
    voxel_spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
) -> dict:
    """Compute all metrics across segmentation, classification, and staging.

    Parameters
    ----------
    seg_pred : Optional[np.ndarray]
        Segmentation predictions.
    seg_target : Optional[np.ndarray]
        Segmentation ground truth.
    cls_true : Optional[np.ndarray]
        Classification ground truth.
    cls_pred : Optional[np.ndarray]
        Classification predictions.
    cls_prob : Optional[np.ndarray]
        Classification probabilities.
    stage_true : Optional[np.ndarray]
        Staging ground truth.
    stage_pred : Optional[np.ndarray]
        Staging predictions.
    voxel_spacing : tuple
        Voxel spacing in mm.

    Returns
    -------
    dict
        All metrics organized by task.
    """
    results = {}

    if seg_pred is not None and seg_target is not None:
        results["segmentation"] = {
            "dice": dice_coefficient(seg_pred, seg_target),
            "iou": iou_score(seg_pred, seg_target),
            "hd95": hausdorff_distance_95(seg_pred, seg_target, voxel_spacing),
            "asd": average_surface_distance(seg_pred, seg_target, voxel_spacing),
            "sensitivity": segmentation_sensitivity(seg_pred, seg_target),
            "specificity": segmentation_specificity(seg_pred, seg_target),
        }

    if cls_true is not None and cls_pred is not None:
        results["classification"] = compute_classification_metrics(cls_true, cls_pred, cls_prob)

    if stage_true is not None and stage_pred is not None:
        results["staging"] = compute_staging_metrics(stage_true, stage_pred)

    return results
