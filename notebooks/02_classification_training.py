"""HepatoScan AI — Kaggle Phase 2: Classification Training Notebook

Trains the classification head with frozen Swin UNETR encoder.
Requires Phase 1 checkpoint (phase1_best.pt).

Kaggle Notebook Setup:
1. Enable GPU
2. Upload phase1_best.pt as a Kaggle dataset
3. Add HCC-TACE-Seg dataset if available
4. Run all cells
"""

# ============================================================================
# Cell 1: Clone Repository (MUST be first cell)
# ============================================================================

# !git clone https://github.com/hepatoscan/hepatoscan-ai.git /kaggle/working/hepatoscan-ai
# %cd /kaggle/working/hepatoscan-ai
# !pip install -q -r requirements.txt

# ============================================================================
# Cell 2: Environment Setup
# ============================================================================

import os
import sys
import logging

sys.path.insert(0, "/kaggle/working/hepatoscan-ai")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# ============================================================================
# Cell 3: Configuration
# ============================================================================

import yaml
from pathlib import Path

with open("configs/base_config.yaml") as f:
    base_cfg = yaml.safe_load(f)
with open("configs/multitask_config.yaml") as f:
    mt_cfg = yaml.safe_load(f)

config = {**base_cfg}
for key, value in mt_cfg.items():
    if isinstance(value, dict) and key in config:
        config[key] = {**config[key], **value}
    else:
        config[key] = value

# Kaggle overrides
config["training"]["max_epochs"] = 50
config["training"]["batch_size"] = 2
config["training"]["learning_rate"] = 1e-3

# ============================================================================
# Cell 4: Training
# ============================================================================

from src.training.train_classification import train_classification

# Phase 1 checkpoint path (adjust based on your Kaggle dataset)
phase1_ckpt = "/kaggle/input/hepatoscan-phase1/phase1_best.pt"
if not Path(phase1_ckpt).exists():
    phase1_ckpt = "checkpoints/phase1_best.pt"

# Check for Phase 2 resume
resume_path = None
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob("phase2_epoch*.pt"))
    if checkpoints:
        resume_path = str(checkpoints[-1])
        print(f"Resuming from: {resume_path}")

summary = train_classification(
    config,
    phase1_checkpoint=phase1_ckpt,
    resume_path=resume_path,
)
print(f"\nPhase 2 Summary: {summary}")

# ============================================================================
# Cell 5: Save Results
# ============================================================================

import shutil
best_ckpt = Path("checkpoints/phase2_best.pt")
if best_ckpt.exists():
    shutil.copy(best_ckpt, "/kaggle/working/phase2_best.pt")
    print(f"Best checkpoint saved")

print("\n✅ Phase 2 complete! Download phase2_best.pt for Phase 3.")
