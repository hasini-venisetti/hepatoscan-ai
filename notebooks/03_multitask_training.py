"""HepatoScan AI — Kaggle Phase 3: Multi-task Joint Training Notebook

End-to-end multi-task training with uncertainty-weighted loss.
Requires Phase 2 checkpoint (phase2_best.pt).

Kaggle Notebook Setup:
1. Enable GPU (preferably A100 or T4 x2)
2. Upload phase2_best.pt as a Kaggle dataset
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

sys.path.insert(0, "/kaggle/working/hepatoscan-ai")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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

# Phase 3 overrides — very low LR for fine-tuning
config["training"]["max_epochs"] = 100
config["training"]["learning_rate"] = 1e-5
config["training"]["batch_size"] = 2

# ============================================================================
# Cell 4: Training
# ============================================================================

from src.training.train_multitask import train_multitask

phase2_ckpt = "/kaggle/input/hepatoscan-phase2/phase2_best.pt"
if not Path(phase2_ckpt).exists():
    phase2_ckpt = "checkpoints/phase2_best.pt"

# Auto-resume
resume_path = None
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob("phase3_epoch*.pt"))
    if checkpoints:
        resume_path = str(checkpoints[-1])
        print(f"Resuming from: {resume_path}")

summary = train_multitask(config, phase2_checkpoint=phase2_ckpt, resume_path=resume_path)
print(f"\nPhase 3 Summary: {summary}")

# ============================================================================
# Cell 5: Evaluation
# ============================================================================

# from src.evaluation.metrics import compute_all_metrics
# from src.evaluation.benchmark import generate_comparison_table
#
# results = compute_all_metrics(...)
# print(generate_comparison_table(results))

# ============================================================================
# Cell 6: Save Final Model
# ============================================================================

import shutil
best_ckpt = Path("checkpoints/phase3_best.pt")
if best_ckpt.exists():
    shutil.copy(best_ckpt, "/kaggle/working/hepatoscan_final.pt")
    print("Final model saved: hepatoscan_final.pt")

print("\n✅ Phase 3 complete! HepatoScan AI is fully trained.")
print("Next: Upload hepatoscan_final.pt and run the Gradio demo.")
