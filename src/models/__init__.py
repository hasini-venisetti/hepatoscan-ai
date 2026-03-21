"""HepatoScan AI — Model architectures.

Multi-task deep learning model for liver lesion diagnosis combining
Swin UNETR segmentation backbone with classification, staging, and
uncertainty estimation heads.
"""

from src.models.hepatoscan import HepatoScanAI
from src.models.backbone import build_swin_unetr
from src.models.classification_head import ClassificationHead
from src.models.staging_head import StagingHead
from src.models.uncertainty_head import MCDropoutWrapper

__all__ = [
    "HepatoScanAI",
    "build_swin_unetr",
    "ClassificationHead",
    "StagingHead",
    "MCDropoutWrapper",
]
