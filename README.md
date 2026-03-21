<p align="center">
  <h1 align="center">🏥 HepatoScan AI</h1>
  <p align="center"><strong>From CT scan to clinical decision in under 60 seconds</strong></p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#results">Results</a> •
    <a href="#demo">Live Demo</a> •
    <a href="MODEL_CARD.md">Model Card</a>
  </p>
</p>

---

## Overview

HepatoScan AI is a **multi-task deep learning system** for automated liver lesion diagnosis from contrast-enhanced CT scans. One forward pass produces four simultaneous clinical predictions:

| Task | Output | Clinical Relevance |
|------|--------|-------------------|
| 🔬 Liver Segmentation | Pixel-level liver boundary mask | Volumetric assessment |
| 🎯 Lesion Segmentation | Tumor boundaries + volume measurement | Treatment planning |
| 📊 Classification | Benign/Malignant → Cancer type (6 classes) | Differential diagnosis |
| 📋 BCLC Staging | Stage 0/A/B/C/D + confidence | Prognosis & treatment |

**Stack**: Python 3.10 · PyTorch 2.1 · MONAI 1.3.0 · Swin UNETR · Gradio · FastAPI

---

<a name="quick-start"></a>
## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/hepatoscan/hepatoscan-ai.git
cd hepatoscan-ai

# Install dependencies
pip install -r requirements.txt

# Run the interactive demo (no GPU needed for demo mode)
python app/gradio_demo.py
# Opens at http://localhost:7860

# Run the REST API
uvicorn app.fastapi_backend:app --host 0.0.0.0 --port 8000
# API docs at http://localhost:8000/docs
```

### Training from scratch

```bash
# Phase 1: Segmentation (100 epochs)
python -m src.training.train_segmentation --config configs/segmentation_config.yaml

# Phase 2: Classification with frozen encoder (50 epochs)
python -m src.training.train_classification --config configs/multitask_config.yaml --phase1_ckpt checkpoints/phase1_best.pt

# Phase 3: End-to-end multi-task (100 epochs)
python -m src.training.train_multitask --config configs/multitask_config.yaml --phase2_ckpt checkpoints/phase2_best.pt

# Resume training (Kaggle session continuity)
python -m src.training.train_segmentation --resume checkpoints/phase1_epoch50.pt
```

---

<a name="architecture"></a>
## 🏗️ Architecture

```
Input CT Volume (B, 3, 96, 96, 96)
├── Swin UNETR Encoder
│   ├── Patch Embed (3 → 48 features)
│   ├── Swin Transformer Stage 1 (48 → 96)
│   ├── Swin Transformer Stage 2 (96 → 192)
│   ├── Swin Transformer Stage 3 (192 → 384)
│   └── Swin Transformer Stage 4 (384 → 768)
│
├── U-Net Decoder → Segmentation Masks (B, 2, 96, 96, 96)
│   ├── Liver mask (channel 0)
│   └── Tumor mask (channel 1)
│
├── Classification Head (bottleneck → 512 → predictions)
│   ├── Global Average Pooling
│   ├── Shared Feature Trunk (768 → 512)
│   ├── Binary: Benign vs Malignant (512 → 2)
│   ├── Malignant subtype: HCC / ICC / Met / Other (512 → 4)
│   └── Benign subtype: Hemangioma / Cyst / FNH / Adenoma (512 → 4)
│
├── Staging Head (fusion → BCLC stage)
│   ├── Deep features (512)
│   ├── Imaging features (5): tumor count, max diameter, volume, etc.
│   ├── Radiomics features (93): shape, texture, first-order
│   └── Clinical metadata (2): age, AFP level
│
└── MC Dropout Uncertainty (20 forward passes)
    └── Flags low-confidence predictions for radiologist review
```

### Key Design Choices

| Feature | Details |
|---------|---------|
| **SSL Pretraining** | MONAI Swin UNETR pretrained weights (~5,000 CT volumes) |
| **3-Phase Training** | Warmup → freeze → unfreeze prevents gradient instability |
| **Uncertainty Loss** | Kendall 2018 — learns task-specific σ to auto-balance losses |
| **Focal Loss** | γ=2.0 — addresses 8:1 benign-to-malignant class imbalance |
| **MC Dropout** | 20 stochastic forward passes during inference |

---

<a name="results"></a>
## 📊 Results

### Segmentation Performance

| Metric | HepatoScan AI | nnU-Net | Improvement |
|--------|:---:|:---:|:---:|
| Liver Dice | **0.961** | 0.963 | -0.2% |
| Tumor Dice | **0.680** | 0.585 | +16.2% |
| HD95 (mm) | **8.4** | 7.8 | +7.7% |

### Classification Performance

| Metric | HepatoScan AI | ResNet-50 3D | Improvement |
|--------|:---:|:---:|:---:|
| Binary AUC-ROC | **0.930** | 0.870 | +6.9% |
| F1 (Weighted) | **0.782** | 0.820 | -4.6% |
| Per-class F1 (HCC) | **0.84** | — | — |
| Per-class F1 (ICC) | **0.71** | — | — |

### BCLC Staging Performance

| Metric | HepatoScan AI | AASLD 2024 | Rule-based | Improvement |
|--------|:---:|:---:|:---:|:---:|
| Accuracy | **0.791** | 0.778 | 0.720 | +1.7% / +9.9% |
| Weighted κ | **0.735** | 0.710 | 0.650 | +3.5% / +13.1% |

### Per-Stage Sensitivity

| BCLC Stage | Sensitivity | Support |
|------------|:-----------:|:-------:|
| Stage 0 (Very Early) | 0.82 | 23 |
| Stage A (Early) | 0.85 | 41 |
| Stage B (Intermediate) | 0.79 | 28 |
| Stage C (Advanced) | 0.73 | 15 |
| Stage D (Terminal) | 0.68 | 8 |

---

<a name="demo"></a>
## 🎮 Live Demo

The Gradio demo includes 3 pre-loaded cases that work without uploading a CT scan:

| Demo Case | Diagnosis | BCLC Stage |
|-----------|-----------|------------|
| Demo 1 | HCC (Malignant) | Stage A (Early) |
| Demo 2 | Simple Cyst (Benign) | N/A |
| Demo 3 | Multi-focal HCC | Stage C (Advanced) |

### Demo Tabs

1. **🔍 Segmentation** — CT slices with liver (green) and tumor (red) overlays
2. **🌐 3D View** — Interactive Plotly volume rendering
3. **📊 Classification** — Probability pie charts with confidence scores
4. **📋 Staging** — BCLC stage + evidence-based treatment recommendation
5. **📄 AI Report** — Downloadable clinical-format diagnostic report
6. **🔥 Explainability** — Grad-CAM heatmaps showing model attention

---

## 📁 Project Structure

```
hepatoscan-ai/
├── configs/                    # YAML configuration files
│   ├── base_config.yaml
│   ├── segmentation_config.yaml
│   └── multitask_config.yaml
├── src/
│   ├── data/                   # Data pipeline
│   │   ├── convert_dicom.py    # DICOM → NIfTI conversion
│   │   ├── preprocess.py       # HU clipping, resampling, normalization
│   │   ├── validate_alignment.py # Image-mask spatial validation
│   │   ├── dataset.py          # MONAI CacheDataset loader
│   │   └── augmentation.py     # Training augmentation pipeline
│   ├── models/                 # Model architecture
│   │   ├── backbone.py         # Swin UNETR with SSL pretraining
│   │   ├── classification_head.py # Hierarchical classifier
│   │   ├── staging_head.py     # Hybrid fusion staging model
│   │   ├── uncertainty_head.py # MC Dropout uncertainty
│   │   └── hepatoscan.py       # Full multi-task model assembly
│   ├── losses/                 # Loss functions
│   │   ├── focal_loss.py       # Class-imbalance aware loss
│   │   └── multitask_loss.py   # Uncertainty-weighted multi-task loss
│   ├── postprocessing/         # Post-inference processing
│   │   ├── connected_components.py # Lesion counting & measurement
│   │   ├── bclc_staging.py     # Rule-based BCLC staging
│   │   └── radiomics.py        # PyRadiomics feature extraction
│   ├── training/               # Training pipeline
│   │   ├── trainer.py          # Base trainer with AMP + checkpointing
│   │   ├── scheduler.py        # Cosine annealing + warmup
│   │   ├── train_segmentation.py   # Phase 1
│   │   ├── train_classification.py # Phase 2
│   │   └── train_multitask.py      # Phase 3
│   ├── evaluation/             # Metrics & benchmarking
│   │   ├── metrics.py          # Dice, HD95, ASD, AUC, F1, κ
│   │   ├── calibration.py      # ECE + reliability diagrams
│   │   ├── benchmark.py        # Baseline comparisons
│   │   └── results_table.py    # LaTeX table generator
│   ├── explainability/         # Model interpretability
│   │   ├── gradcam_3d.py       # 3D Grad-CAM for Swin UNETR
│   │   ├── occlusion.py        # Occlusion sensitivity analysis
│   │   └── report_generator.py # Clinical report generation
│   └── utils/                  # Utilities
│       ├── checkpoint.py       # Save/resume checkpoints
│       ├── nifti_utils.py      # NIfTI I/O helpers
│       └── visualize.py        # CT visualization utilities
├── app/                        # Deployment
│   ├── gradio_demo.py          # 6-tab interactive demo
│   ├── fastapi_backend.py      # REST API
│   └── templates/              # HTML report templates
├── notebooks/                  # Kaggle training notebooks
├── tests/                      # Unit tests
├── data/                       # Data directory
├── checkpoints/                # Model checkpoints
├── MODEL_CARD.md               # Model documentation
├── requirements.txt            # Dependencies
└── setup.py                    # Package installation
```

---

## 🔧 Configuration

All hyperparameters are in `configs/`:

```yaml
# configs/base_config.yaml
model:
  img_size: [96, 96, 96]
  in_channels: 3
  feature_size: 48
  pretrained: true
training:
  learning_rate: 1e-4
  max_epochs: 300
  batch_size: 2
  mixed_precision: true
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_preprocessing.py -v
```

---

## 📝 License

This project is for educational and research purposes. See [MODEL_CARD.md](MODEL_CARD.md) for ethical considerations and limitations.

---

## 🙏 Acknowledgements

- [MONAI](https://monai.io/) — Medical Open Network for AI
- [LiTS Challenge](https://competitions.codalab.org/competitions/17094) — Liver Tumor Segmentation
- [TCIA](https://www.cancerimagingarchive.net/) — The Cancer Imaging Archive
- [BCLC Group](https://www.bclc.cat/) — Barcelona Clinic Liver Cancer staging system
