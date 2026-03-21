"""Hybrid BCLC staging head with deep + imaging + radiomics + clinical fusion.

Combines four independent feature streams:
1. Deep features — encoder output after global avg pool (512-dim)
2. Imaging features — lesion count, max diameter, volume, vessel proximity (5-dim)
3. Radiomics features — PyRadiomics shape + texture features (93-dim)
4. Clinical features — AFP level, treatment type (2-dim)

Outputs 5-class BCLC stage prediction (Stage 0/A/B/C/D).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DEEP_DIM = 512
DEFAULT_IMAGING_DIM = 5
DEFAULT_RADIOMICS_DIM = 93
DEFAULT_CLINICAL_DIM = 2
NUM_BCLC_STAGES = 5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Staging Head
# ---------------------------------------------------------------------------


class StagingHead(nn.Module):
    """Hybrid BCLC staging head with multi-stream feature fusion.

    Parameters
    ----------
    deep_dim : int
        Dimension of deep features from classification head. Default 512.
    imaging_dim : int
        Dimension of imaging-derived features. Default 5.
    radiomics_dim : int
        Dimension of PyRadiomics features. Default 93.
    clinical_dim : int
        Dimension of clinical metadata features. Default 2.
    num_stages : int
        Number of BCLC stages. Default 5 (0/A/B/C/D).
    dropout_rate : float
        Dropout rate for deep and radiomics branches. Default 0.3.
    """

    def __init__(
        self,
        deep_dim: int = DEFAULT_DEEP_DIM,
        imaging_dim: int = DEFAULT_IMAGING_DIM,
        radiomics_dim: int = DEFAULT_RADIOMICS_DIM,
        clinical_dim: int = DEFAULT_CLINICAL_DIM,
        num_stages: int = NUM_BCLC_STAGES,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        # Branch 1: Deep features
        self.deep_branch = nn.Sequential(
            nn.Linear(deep_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        deep_out = 256

        # Branch 2: Imaging features
        self.imaging_branch = nn.Sequential(
            nn.Linear(imaging_dim, 32),
            nn.ReLU(inplace=True),
        )
        imaging_out = 32

        # Branch 3: Radiomics features
        self.radiomics_branch = nn.Sequential(
            nn.Linear(radiomics_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        radiomics_out = 64

        # Branch 4: Clinical features
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 16),
            nn.ReLU(inplace=True),
        )
        clinical_out = 16

        # Fusion layer
        fusion_dim = deep_out + imaging_out + radiomics_out + clinical_out  # 368
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_stages),
        )

        # Flag for graceful degradation when metadata is missing
        self.imaging_dim = imaging_dim
        self.radiomics_dim = radiomics_dim
        self.clinical_dim = clinical_dim

        self._init_weights()

        logger.info(
            "StagingHead: deep=%d, imaging=%d, radiomics=%d, clinical=%d → fusion=%d → stages=%d",
            deep_dim, imaging_dim, radiomics_dim, clinical_dim, fusion_dim, num_stages,
        )

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        deep_features: torch.Tensor,
        imaging_features: Optional[torch.Tensor] = None,
        radiomics_features: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with graceful degradation for missing features.

        Parameters
        ----------
        deep_features : torch.Tensor
            Deep features from classification head. Shape (B, deep_dim).
        imaging_features : Optional[torch.Tensor]
            Imaging-derived features [lesion_count, max_diameter_cm,
            total_volume_cc, vessel_proximity_flag, num_lesions_gt_3cm].
            Shape (B, 5). If None, uses zeros.
        radiomics_features : Optional[torch.Tensor]
            PyRadiomics features. Shape (B, 93). If None, uses zeros.
        clinical_features : Optional[torch.Tensor]
            Clinical metadata [afp_level_log, treatment_type_encoded].
            Shape (B, 2). If None, uses zeros.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - 'stage_logits': (B, 5) — BCLC stage logits
            - 'metadata_available': bool — whether clinical data was provided
        """
        batch_size = deep_features.size(0)
        device = deep_features.device

        # Graceful degradation: use zeros if features not available
        if imaging_features is None:
            imaging_features = torch.zeros(batch_size, self.imaging_dim, device=device)
            metadata_available = False
        else:
            metadata_available = True

        if radiomics_features is None:
            radiomics_features = torch.zeros(batch_size, self.radiomics_dim, device=device)

        if clinical_features is None:
            clinical_features = torch.zeros(batch_size, self.clinical_dim, device=device)
            metadata_available = False

        # Process each branch
        deep_out = self.deep_branch(deep_features)
        imaging_out = self.imaging_branch(imaging_features)
        radiomics_out = self.radiomics_branch(radiomics_features)
        clinical_out = self.clinical_branch(clinical_features)

        # Concatenate and fuse
        fused = torch.cat([deep_out, imaging_out, radiomics_out, clinical_out], dim=1)
        stage_logits = self.fusion(fused)

        return {
            "stage_logits": stage_logits,
            "metadata_available": metadata_available,
        }

    def predict(
        self,
        deep_features: torch.Tensor,
        imaging_features: Optional[torch.Tensor] = None,
        radiomics_features: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Inference with probabilities and stage labels.

        Parameters
        ----------
        deep_features : torch.Tensor
            Deep features from classification head.
        imaging_features : Optional[torch.Tensor]
            Imaging-derived features.
        radiomics_features : Optional[torch.Tensor]
            PyRadiomics features.
        clinical_features : Optional[torch.Tensor]
            Clinical metadata features.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with stage probabilities and predicted stage index.
        """
        outputs = self.forward(deep_features, imaging_features, radiomics_features, clinical_features)

        stage_probs = torch.softmax(outputs["stage_logits"], dim=1)
        predicted_stage = torch.argmax(stage_probs, dim=1)

        return {
            "stage_probs": stage_probs,
            "predicted_stage": predicted_stage,
            "metadata_available": outputs["metadata_available"],
        }
