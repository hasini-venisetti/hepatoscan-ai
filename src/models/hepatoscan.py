"""HepatoScan AI — Full multi-task model assembly.

Combines Swin UNETR segmentation backbone with classification, staging,
and uncertainty estimation heads into a single end-to-end model.

One forward pass produces:
- liver_mask: (B, 1, H, W, D) — liver segmentation
- tumor_mask: (B, 1, H, W, D) — tumor segmentation
- benign_malignant_logits: (B, 2) — binary classification
- cancer_type_logits: (B, 4) — subtype classification
- bclc_stage_logits: (B, 5) — BCLC staging
- uncertainty_map: (B, 1, H, W, D) — only during eval with MC Dropout

Architecture:
    Input CT Volume (B, 3, 96, 96, 96)
    ├── Swin UNETR Encoder → bottleneck features (B, 768, 3, 3, 3)
    ├── Swin UNETR Decoder → segmentation masks (B, 2, 96, 96, 96)
    ├── ClassificationHead → binary + subtype logits
    └── StagingHead → BCLC stage logits
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from src.models.backbone import build_swin_unetr, get_encoder_output_dim
from src.models.classification_head import ClassificationHead
from src.models.staging_head import StagingHead
from src.models.uncertainty_head import MCDropoutWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BCLC_STAGE_NAMES = ["Stage 0 (Very Early)", "Stage A (Early)", "Stage B (Intermediate)",
                     "Stage C (Advanced)", "Stage D (Terminal)"]

CLASS_NAMES_BINARY = ["Benign", "Malignant"]
CLASS_NAMES_MALIGNANT = ["HCC", "ICC", "Colorectal Metastasis", "Other"]
CLASS_NAMES_BENIGN = ["Hemangioma", "Cyst", "FNH", "Adenoma"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class HepatoScanAI(nn.Module):
    """Multi-task liver lesion diagnosis model.

    Combines segmentation, classification, and staging into a single
    architecture with shared Swin UNETR encoder.

    Parameters
    ----------
    config : dict
        Model configuration dictionary. Expected keys:
        - model.img_size: (96, 96, 96)
        - model.in_channels: 3
        - model.out_channels: 2
        - model.feature_size: 48
        - model.pretrained: True
        - model.pretrained_path: str
        - model.use_checkpoint: True
        - model.dropout_rate: 0.4
        - model.mc_dropout_samples: 20
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        model_cfg = config.get("model", {})
        self.config = config

        # ---- Backbone: Swin UNETR ----
        self.img_size = tuple(model_cfg.get("img_size", [96, 96, 96]))
        self.in_channels = model_cfg.get("in_channels", 3)
        self.out_channels = model_cfg.get("out_channels", 2)
        self.feature_size = model_cfg.get("feature_size", 48)

        self.backbone = build_swin_unetr(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            feature_size=self.feature_size,
            pretrained=model_cfg.get("pretrained", True),
            pretrained_path=model_cfg.get("pretrained_path"),
            use_checkpoint=model_cfg.get("use_checkpoint", True),
        )

        # ---- Classification Head ----
        encoder_dim = get_encoder_output_dim(self.feature_size)
        dropout_rate = model_cfg.get("dropout_rate", 0.4)

        self.classification_head = ClassificationHead(
            in_features=encoder_dim,
            hidden_dim=512,
            dropout_rate=dropout_rate,
        )

        # ---- Staging Head ----
        staging_cfg = config.get("staging", {})
        self.staging_head = StagingHead(
            deep_dim=512,  # Must match ClassificationHead hidden_dim
            imaging_dim=staging_cfg.get("imaging_feature_dim", 5),
            radiomics_dim=staging_cfg.get("radiomics_feature_dim", 93),
            clinical_dim=staging_cfg.get("clinical_feature_dim", 2),
            num_stages=model_cfg.get("num_bclc_stages", 5),
        )

        # ---- MC Dropout for Uncertainty ----
        self.mc_dropout_samples = model_cfg.get("mc_dropout_samples", 20)

        # ---- Mode flags ----
        self._segmentation_only = False

        total_params = sum(p.numel() for p in self.parameters())
        logger.info("HepatoScanAI initialized: %.2fM parameters", total_params / 1e6)

    def set_segmentation_only(self, flag: bool = True) -> None:
        """Set model to segmentation-only mode (Phase 1 training).

        Parameters
        ----------
        flag : bool
            If True, forward() returns only segmentation output.
        """
        self._segmentation_only = flag
        logger.info("Segmentation-only mode: %s", flag)

    def _extract_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck features from Swin UNETR encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Encoder bottleneck features.
        """
        # Access the SwinViT encoder layers
        hidden_states = self.backbone.swinViT(x, self.backbone.normalize)
        # The last hidden state is the deepest encoder feature
        encoder_features = hidden_states[-1]
        return encoder_features

    def forward(
        self,
        x: torch.Tensor,
        imaging_features: Optional[torch.Tensor] = None,
        radiomics_features: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through all heads.

        Parameters
        ----------
        x : torch.Tensor
            Input CT volume. Shape (B, C, D, H, W).
        imaging_features : Optional[torch.Tensor]
            Imaging-derived features for staging. Shape (B, 5).
        radiomics_features : Optional[torch.Tensor]
            PyRadiomics features. Shape (B, 93).
        clinical_features : Optional[torch.Tensor]
            Clinical metadata. Shape (B, 2).

        Returns
        -------
        dict[str, torch.Tensor]
            All model outputs:
            - 'seg_logits': (B, out_channels, D, H, W)
            - 'binary_logits': (B, 2)
            - 'malignant_logits': (B, 4)
            - 'benign_logits': (B, 4)
            - 'stage_logits': (B, 5)
            - 'cls_features': (B, 512) — intermediate features
        """
        # Segmentation: full Swin UNETR forward
        seg_logits = self.backbone(x)  # (B, out_channels, D, H, W)

        outputs = {"seg_logits": seg_logits}

        if self._segmentation_only:
            return outputs

        # Extract encoder features for classification and staging
        encoder_features = self._extract_encoder_features(x)

        # Classification
        cls_outputs = self.classification_head(encoder_features)
        outputs["binary_logits"] = cls_outputs["binary_logits"]
        outputs["malignant_logits"] = cls_outputs["malignant_logits"]
        outputs["benign_logits"] = cls_outputs["benign_logits"]
        outputs["cls_features"] = cls_outputs["features"]

        # Staging (uses classification features + optional metadata)
        staging_outputs = self.staging_head(
            deep_features=cls_outputs["features"],
            imaging_features=imaging_features,
            radiomics_features=radiomics_features,
            clinical_features=clinical_features,
        )
        outputs["stage_logits"] = staging_outputs["stage_logits"]
        outputs["metadata_available"] = staging_outputs["metadata_available"]

        return outputs

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        imaging_features: Optional[torch.Tensor] = None,
        radiomics_features: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
        use_half: bool = False,
    ) -> dict[str, Any]:
        """Inference with probabilities and human-readable labels.

        Parameters
        ----------
        x : torch.Tensor
            Input CT volume.
        imaging_features : Optional[torch.Tensor]
            Imaging features for staging.
        radiomics_features : Optional[torch.Tensor]
            Radiomics features.
        clinical_features : Optional[torch.Tensor]
            Clinical metadata.
        use_half : bool
            Cast model to float16 for faster inference.

        Returns
        -------
        dict[str, Any]
            Human-readable predictions with probabilities.
        """
        if use_half:
            x = x.half()
            self.half()

        self.eval()
        outputs = self.forward(x, imaging_features, radiomics_features, clinical_features)

        # Convert logits to probabilities
        seg_probs = torch.sigmoid(outputs["seg_logits"])

        result = {
            "seg_probs": seg_probs,
            "liver_mask": (seg_probs[:, 0:1] > 0.5).float() if self.out_channels > 1 else seg_probs,
            "tumor_mask": (seg_probs[:, 1:2] > 0.5).float() if self.out_channels > 1 else seg_probs,
        }

        if not self._segmentation_only:
            binary_probs = torch.softmax(outputs["binary_logits"], dim=1)
            malignant_probs = torch.softmax(outputs["malignant_logits"], dim=1)
            benign_probs = torch.softmax(outputs["benign_logits"], dim=1)
            stage_probs = torch.softmax(outputs["stage_logits"], dim=1)

            # Human-readable labels
            binary_idx = binary_probs.argmax(dim=1).item()
            stage_idx = stage_probs.argmax(dim=1).item()

            result.update({
                "binary_probs": binary_probs,
                "malignant_probs": malignant_probs,
                "benign_probs": benign_probs,
                "stage_probs": stage_probs,
                "malignancy": CLASS_NAMES_BINARY[binary_idx],
                "malignancy_confidence": binary_probs.max().item() * 100,
                "cancer_type": (
                    CLASS_NAMES_MALIGNANT[malignant_probs.argmax(dim=1).item()]
                    if binary_idx == 1
                    else CLASS_NAMES_BENIGN[benign_probs.argmax(dim=1).item()]
                ),
                "bclc_stage": BCLC_STAGE_NAMES[stage_idx],
                "stage_confidence": stage_probs.max().item() * 100,
            })

        if use_half:
            self.float()

        return result

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """MC Dropout inference for uncertainty estimation.

        Parameters
        ----------
        x : torch.Tensor
            Input CT volume.
        num_samples : Optional[int]
            Number of MC samples. Default uses config value.
        **kwargs
            Additional features forwarded to self.forward().

        Returns
        -------
        dict[str, Any]
            Predictions with uncertainty estimates and review flags.
        """
        n = num_samples or self.mc_dropout_samples

        # Enable dropout during inference
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        seg_samples = []
        binary_samples = []
        stage_samples = []

        for _ in range(n):
            outputs = self.forward(x, **kwargs)
            seg_samples.append(torch.sigmoid(outputs["seg_logits"]))
            if "binary_logits" in outputs:
                binary_samples.append(torch.softmax(outputs["binary_logits"], dim=1))
            if "stage_logits" in outputs:
                stage_samples.append(torch.softmax(outputs["stage_logits"], dim=1))

        # Aggregate segmentation uncertainty
        seg_stack = torch.stack(seg_samples, dim=0)
        seg_mean = seg_stack.mean(dim=0)
        seg_std = seg_stack.std(dim=0)

        result = {
            "seg_probs": seg_mean,
            "seg_uncertainty": seg_std,
            "liver_mask": (seg_mean[:, 0:1] > 0.5).float(),
            "tumor_mask": (seg_mean[:, 1:2] > 0.5).float() if seg_mean.shape[1] > 1 else seg_mean,
        }

        if binary_samples:
            binary_stack = torch.stack(binary_samples, dim=0)
            result["binary_probs"] = binary_stack.mean(dim=0)
            result["binary_uncertainty"] = binary_stack.std(dim=0)

            max_binary_std = result["binary_uncertainty"].max().item()
            result["needs_review"] = max_binary_std > 0.2

        if stage_samples:
            stage_stack = torch.stack(stage_samples, dim=0)
            result["stage_probs"] = stage_stack.mean(dim=0)
            result["stage_uncertainty"] = stage_stack.std(dim=0)

        return result


def build_hepatoscan(config: dict) -> HepatoScanAI:
    """Factory function to build HepatoScanAI from config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    HepatoScanAI
        Initialized model.
    """
    model = HepatoScanAI(config)
    return model
