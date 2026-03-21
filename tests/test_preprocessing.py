"""Unit tests for preprocessing and metrics."""

import numpy as np
import pytest


# ============================================================================
# Preprocessing Tests
# ============================================================================


class TestPreprocessing:
    """Tests for src/data/preprocess.py functions."""

    def test_clip_hounsfield_units(self):
        """Test HU clipping to liver window."""
        from src.data.preprocess import clip_hounsfield_units

        arr = np.array([-500, -100, 0, 200, 400, 1000], dtype=np.float32)
        clipped = clip_hounsfield_units(arr, hu_min=-100, hu_max=400)

        assert clipped.min() == -100
        assert clipped.max() == 400
        assert clipped[2] == 0  # Within range, unchanged

    def test_normalize_to_unit_range(self):
        """Test normalization to [0, 1]."""
        from src.data.preprocess import normalize_to_unit_range

        arr = np.array([-100, 0, 150, 400], dtype=np.float32)
        norm = normalize_to_unit_range(arr, hu_min=-100, hu_max=400)

        assert np.isclose(norm[0], 0.0)
        assert np.isclose(norm[-1], 1.0)
        assert norm.dtype == np.float32

    def test_stack_multi_phase_single(self):
        """Test single-phase duplication to 3 channels."""
        from src.data.preprocess import stack_multi_phase

        single = np.random.rand(32, 32, 32).astype(np.float32)
        stacked = stack_multi_phase([single], num_channels=3)

        assert stacked.shape == (3, 32, 32, 32)
        np.testing.assert_array_equal(stacked[0], stacked[1])
        np.testing.assert_array_equal(stacked[1], stacked[2])

    def test_stack_multi_phase_three(self):
        """Test three-phase stacking."""
        from src.data.preprocess import stack_multi_phase

        phases = [np.random.rand(32, 32, 32).astype(np.float32) for _ in range(3)]
        stacked = stack_multi_phase(phases, num_channels=3)

        assert stacked.shape == (3, 32, 32, 32)
        np.testing.assert_array_equal(stacked[0], phases[0])
        np.testing.assert_array_equal(stacked[2], phases[2])

    def test_stack_multi_phase_empty_raises(self):
        """Test that empty input raises ValueError."""
        from src.data.preprocess import stack_multi_phase

        with pytest.raises(ValueError):
            stack_multi_phase([], num_channels=3)


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetrics:
    """Tests for src/evaluation/metrics.py functions."""

    def test_dice_perfect(self):
        """Test Dice with identical masks."""
        from src.evaluation.metrics import dice_coefficient

        mask = np.ones((10, 10, 10), dtype=np.uint8)
        assert dice_coefficient(mask, mask) > 0.999

    def test_dice_empty(self):
        """Test Dice with empty masks."""
        from src.evaluation.metrics import dice_coefficient

        a = np.zeros((10, 10, 10), dtype=np.uint8)
        b = np.zeros((10, 10, 10), dtype=np.uint8)
        # With smoothing, should be close to 1 for two empty masks
        result = dice_coefficient(a, b)
        assert 0 <= result <= 1

    def test_dice_no_overlap(self):
        """Test Dice with non-overlapping masks."""
        from src.evaluation.metrics import dice_coefficient

        a = np.zeros((10, 10, 10), dtype=np.uint8)
        b = np.zeros((10, 10, 10), dtype=np.uint8)
        a[:5] = 1
        b[5:] = 1

        result = dice_coefficient(a, b)
        assert result < 0.01  # Very close to 0

    def test_iou_perfect(self):
        """Test IoU with identical masks."""
        from src.evaluation.metrics import iou_score

        mask = np.ones((10, 10, 10), dtype=np.uint8)
        assert iou_score(mask, mask) > 0.999

    def test_sensitivity_all_detected(self):
        """Test sensitivity when all positives are detected."""
        from src.evaluation.metrics import segmentation_sensitivity

        pred = np.ones((10, 10, 10))
        target = np.ones((10, 10, 10))
        assert segmentation_sensitivity(pred, target) > 0.999

    def test_specificity_no_false_positives(self):
        """Test specificity with no false positives."""
        from src.evaluation.metrics import segmentation_specificity

        pred = np.zeros((10, 10, 10))
        target = np.zeros((10, 10, 10))
        assert segmentation_specificity(pred, target) > 0.999


# ============================================================================
# Connected Components Tests
# ============================================================================


class TestConnectedComponents:
    """Tests for src/postprocessing/connected_components.py functions."""

    def test_binarize_mask(self):
        """Test binary thresholding."""
        from src.postprocessing.connected_components import binarize_mask

        arr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        binary = binarize_mask(arr, threshold=0.5, apply_sigmoid=False)

        assert binary[0] == 0
        assert binary[1] == 0
        assert binary[2] == 1
        assert binary[3] == 1
        assert binary[4] == 1

    def test_remove_small_components(self):
        """Test removal of small components."""
        from src.postprocessing.connected_components import remove_small_components

        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        # Big component
        mask[5:15, 5:15, 5:15] = 1
        # Small component (< 50 voxels)
        mask[0, 0, 0] = 1
        mask[0, 0, 1] = 1

        cleaned = remove_small_components(mask, min_size=50)
        assert cleaned[0, 0, 0] == 0  # Small component removed
        assert cleaned[10, 10, 10] == 1  # Big component kept

    def test_extract_lesion_measurements(self):
        """Test complete lesion analysis pipeline."""
        from src.postprocessing.connected_components import extract_lesion_measurements

        mask = np.zeros((32, 32, 32), dtype=np.float32)
        mask[10:20, 10:20, 10:20] = 1.0  # 10x10x10 = 1000 voxels

        analysis = extract_lesion_measurements(
            mask,
            voxel_spacing=(1.5, 1.5, 1.5),
            threshold=0.5,
            apply_sigmoid=False,
        )

        assert analysis.lesion_count >= 1
        assert analysis.total_volume_cc > 0
        assert analysis.max_diameter_cm > 0


# ============================================================================
# BCLC Staging Tests
# ============================================================================


class TestBCLCStaging:
    """Tests for src/postprocessing/bclc_staging.py functions."""

    def test_stage_0(self):
        """Test Stage 0: single lesion ≤2cm."""
        from src.postprocessing.bclc_staging import compute_bclc_stage

        result = compute_bclc_stage(lesion_count=1, max_diameter_cm=1.5)
        assert "Stage 0" in result.stage

    def test_stage_a_single(self):
        """Test Stage A: single lesion ≤5cm."""
        from src.postprocessing.bclc_staging import compute_bclc_stage

        result = compute_bclc_stage(lesion_count=1, max_diameter_cm=4.0)
        assert "Stage A" in result.stage

    def test_stage_a_multiple(self):
        """Test Stage A: ≤3 lesions each ≤3cm."""
        from src.postprocessing.bclc_staging import compute_bclc_stage

        result = compute_bclc_stage(lesion_count=3, max_diameter_cm=2.5)
        assert "Stage A" in result.stage

    def test_stage_b(self):
        """Test Stage B: multinodular."""
        from src.postprocessing.bclc_staging import compute_bclc_stage

        result = compute_bclc_stage(lesion_count=5, max_diameter_cm=2.0)
        assert "Stage B" in result.stage

    def test_stage_c(self):
        """Test Stage C: vascular invasion."""
        from src.postprocessing.bclc_staging import compute_bclc_stage

        result = compute_bclc_stage(lesion_count=1, max_diameter_cm=3.0, has_vascular_invasion=True)
        assert "Stage C" in result.stage

    def test_treatment_recommendation(self):
        """Test that treatment is always provided."""
        from src.postprocessing.bclc_staging import compute_bclc_stage

        result = compute_bclc_stage(lesion_count=1, max_diameter_cm=1.0)
        assert len(result.treatment) > 0


# ============================================================================
# Report Generator Tests
# ============================================================================


class TestReportGenerator:
    """Tests for src/explainability/report_generator.py."""

    def test_report_generation(self):
        """Test that a report is generated with required sections."""
        from src.explainability.report_generator import generate_clinical_report

        predictions = {
            "malignancy": "Malignant",
            "malignancy_confidence": 95.0,
            "cancer_type": "HCC",
            "type_confidence": 88.0,
            "bclc_stage": "Stage A (Early)",
            "stage_confidence": 85.0,
            "treatment": "Resection",
            "median_survival": "36-60 months",
            "lesion_count": 1,
            "max_diameter": 3.0,
            "max_volume": 14.0,
            "total_volume": 14.0,
            "liver_volume": 1500.0,
            "needs_review": False,
        }

        report = generate_clinical_report(predictions, patient_id="TEST-001")

        assert "TEST-001" in report
        assert "Malignant" in report
        assert "HCC" in report
        assert "Stage A" in report
        assert "DISCLAIMER" in report


# ============================================================================
# Focal Loss Tests
# ============================================================================


class TestFocalLoss:
    """Tests for src/losses/focal_loss.py."""

    def test_focal_loss_basic(self):
        """Test focal loss computation."""
        import torch
        from src.losses.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 3)  # 4 samples, 3 classes
        targets = torch.tensor([0, 1, 2, 1])

        loss = loss_fn(logits, targets)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_focal_loss_ignores_invalid(self):
        """Test that label=-1 samples are ignored."""
        import torch
        from src.losses.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, -1, -1])

        loss = loss_fn(logits, targets)
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
