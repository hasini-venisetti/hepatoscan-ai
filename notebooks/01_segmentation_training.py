"""HepatoScan AI — Kaggle Phase 1: Segmentation Training Notebook

Run this notebook on Kaggle with GPU P100/T4 enabled.
Estimated runtime: 4-6 hours for 100 epochs.

Kaggle Notebook Setup:
1. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
2. Add LiTS dataset: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation
3. Run all cells
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

# Add project to path
sys.path.insert(0, "/kaggle/working/hepatoscan-ai")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("Phase1")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================================
# Cell 3: Configuration
# ============================================================================

import yaml
from pathlib import Path

# Load configs
with open("configs/base_config.yaml") as f:
    base_cfg = yaml.safe_load(f)
with open("configs/segmentation_config.yaml") as f:
    seg_cfg = yaml.safe_load(f)

# Merge configs
config = {**base_cfg}
for key, value in seg_cfg.items():
    if isinstance(value, dict) and key in config:
        config[key] = {**config[key], **value}
    else:
        config[key] = value

# Kaggle-specific overrides
config["data"]["data_dir"] = "/kaggle/input/liver-tumor-segmentation"
config["data"]["processed_dir"] = "/kaggle/working/processed"
config["training"]["max_epochs"] = 100
config["training"]["batch_size"] = 2
config["training"]["save_every_n_epochs"] = 10

print("Configuration loaded successfully")
print(f"  img_size: {config['model']['img_size']}")
print(f"  max_epochs: {config['training']['max_epochs']}")
print(f"  batch_size: {config['training']['batch_size']}")

# ============================================================================
# Cell 4: Data Preprocessing
# ============================================================================

# from src.data.preprocess import preprocess_dataset
# preprocess_dataset(
#     input_dir=config["data"]["data_dir"],
#     output_dir=config["data"]["processed_dir"],
# )
# print("Preprocessing complete")

# ============================================================================
# Cell 5: Training
# ============================================================================

from src.training.train_segmentation import train_segmentation

# Check for existing checkpoint (Kaggle session resume)
resume_path = None
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob("phase1_epoch*.pt"))
    if checkpoints:
        resume_path = str(checkpoints[-1])
        print(f"Resuming from: {resume_path}")

summary = train_segmentation(config, resume_path=resume_path)
print(f"\nTraining Summary: {summary}")

# ============================================================================
# Cell 6: Save Results
# ============================================================================

# Copy best checkpoint to Kaggle output
import shutil
best_ckpt = Path("checkpoints/phase1_best.pt")
if best_ckpt.exists():
    shutil.copy(best_ckpt, "/kaggle/working/phase1_best.pt")
    print(f"Best checkpoint saved: {best_ckpt}")
else:
    print("No best checkpoint found")

print("\n✅ Phase 1 complete! Download phase1_best.pt for Phase 2.")
