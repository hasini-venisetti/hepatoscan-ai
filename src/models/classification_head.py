"""Two-stage hierarchical classification head.

Stage 1: Binary classifier — Benign vs Malignant
Stage 2a: Multi-class malignant subtype — HCC / ICC / Colorectal Metastasis / Other
Stage 2b: Multi-class benign subtype — Hemangioma / Cyst / FNH / Adenoma

Architecture:
    Input: encoder feature map from Swin UNETR (last encoder layer)
    → 3D Global Average Pooling → flatten to feature vector
    → FC(feature_dim → 512) → BatchNorm → ReLU → Dropout(0.4)
    → Three parallel heads for binary, malignant subtype, benign subtype
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BINARY_CLASSES = 2       # benign, malignant
NUM_MALIGNANT_SUBTYPES = 4   # HCC, ICC, Colorectal Met, Other
NUM_BENIGN_SUBTYPES = 4      # Hemangioma, Cyst, FNH, Adenoma
DEFAULT_HIDDEN_DIM = 512
DEFAULT_DROPOUT = 0.4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------


class ClassificationHead(nn.Module):
    """Two-stage hierarchical classification head for lesion typing.

    Parameters
    ----------
    in_features : int
        Dimension of the input feature vector from the encoder.
    hidden_dim : int
        Intermediate hidden layer dimension. Default 512.
    num_binary : int
        Number of binary classification classes. Default 2.
    num_malignant : int
        Number of malignant subtype classes. Default 4.
    num_benign : int
        Number of benign subtype classes. Default 4.
    dropout_rate : float
        Dropout rate for regularization. Default 0.4.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_binary: int = NUM_BINARY_CLASSES,
        num_malignant: int = NUM_MALIGNANT_SUBTYPES,
        num_benign: int = NUM_BENIGN_SUBTYPES,
        dropout_rate: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # 3D Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Shared feature trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )

        # Stage 1: Binary head (benign vs malignant)
        self.binary_head = nn.Linear(hidden_dim, num_binary)

        # Stage 2a: Malignant subtype head
        self.malignant_head = nn.Linear(hidden_dim, num_malignant)

        # Stage 2b: Benign subtype head
        self.benign_head = nn.Linear(hidden_dim, num_benign)

        self._init_weights()

        logger.info(
            "ClassificationHead: in=%d, hidden=%d, binary=%d, malignant=%d, benign=%d",
            in_features, hidden_dim, num_binary, num_malignant, num_benign,
        )

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        encoder_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the classification head.

        Parameters
        ----------
        encoder_features : torch.Tensor
            Encoder output feature map with shape (B, C, D, H, W).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - 'binary_logits': (B, 2) — benign vs malignant
            - 'malignant_logits': (B, 4) — malignant subtype
            - 'benign_logits': (B, 4) — benign subtype
            - 'features': (B, hidden_dim) — intermediate features for staging head
        """
        # 3D Global Average Pooling: (B, C, D, H, W) → (B, C, 1, 1, 1)
        pooled = self.global_avg_pool(encoder_features)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)

        # Shared trunk
        features = self.trunk(pooled)  # (B, hidden_dim)

        # Three parallel heads
        binary_logits = self.binary_head(features)      # (B, 2)
        malignant_logits = self.malignant_head(features) # (B, 4)
        benign_logits = self.benign_head(features)       # (B, 4)

        return {
            "binary_logits": binary_logits,
            "malignant_logits": malignant_logits,
            "benign_logits": benign_logits,
            "features": features,
        }

    def predict(
        self,
        encoder_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Inference with probabilities instead of logits.

        Parameters
        ----------
        encoder_features : torch.Tensor
            Encoder output feature map.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with probability distributions instead of logits.
        """
        outputs = self.forward(encoder_features)

        return {
            "binary_probs": torch.softmax(outputs["binary_logits"], dim=1),
            "malignant_probs": torch.softmax(outputs["malignant_logits"], dim=1),
            "benign_probs": torch.softmax(outputs["benign_logits"], dim=1),
            "features": outputs["features"],
        }
