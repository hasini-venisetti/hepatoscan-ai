# HepatoScan AI — Datasets

## Overview

HepatoScan AI uses six public datasets for training, validation, and testing.
Each dataset serves a specific purpose in the multi-task pipeline.

## Primary Datasets

### 1. LiTS (Liver Tumor Segmentation Challenge)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)
- **Contents**: 201 multi-center CT volumes with expert liver + tumor segmentation masks
- **Format**: NIfTI (.nii.gz) — ready to use
- **Purpose**: Primary segmentation training (large-scale, multi-center diversity)

### 2. HCC-TACE-Seg
- **Source**: [TCIA](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/)
- **Contents**: 105 HCC patients, multi-phase CECT, tumor masks, BCLC staging labels
- **Format**: DICOM + DICOM-SEG → convert with `python -m src.data.convert_dicom`
- **Purpose**: Classification + staging training (ground truth BCLC labels)

### 3. 3D-IRCADb-01
- **Source**: [IRCAD](https://www.ircad.fr/research/3d-ircadb-01/)
- **Contents**: 20 CT scans with high-quality annotations (cysts, hemangiomas)
- **Format**: DICOM → convert with `python -m src.data.convert_dicom`
- **Purpose**: External validation + benign class training

### 4. Medical Segmentation Decathlon — Liver Task
- **Source**: [Decathlon](http://medicaldecathlon.com)
- **Contents**: 201 portal venous phase CT volumes with liver + tumor masks
- **Format**: NIfTI — ready to use
- **Purpose**: Additional segmentation training volume

### 5. CHAOS Challenge
- **Source**: [Zenodo](https://zenodo.org/record/3431873)
- **Contents**: 40 CT + 120 MRI abdominal volumes
- **Format**: DICOM → convert to NIfTI
- **Purpose**: Liver boundary validation, CT/MRI generalization test

### 6. LLD-MMRI (Supplementary)
- **Source**: GitHub search "LLD-MMRI liver lesion"
- **Contents**: 600+ multi-phase MRI with 8 lesion types
- **Format**: NIfTI
- **Purpose**: Rare benign class diversity (FNH, adenoma)

## Dataset → Task Mapping

| Dataset | Segmentation | Classification | Staging |
|---------|:---:|:---:|:---:|
| LiTS | ✅ | ❌ | ❌ |
| HCC-TACE-Seg | ✅ | ✅ | ✅ |
| 3D-IRCADb | ✅ | ✅ | ❌ |
| Decathlon | ✅ | ❌ | ❌ |
| CHAOS | ✅ (validation) | ❌ | ❌ |
| LLD-MMRI | ❌ | ✅ | ❌ |

## Preprocessing Pipeline

```bash
# 1. Convert DICOM datasets
python -m src.data.convert_dicom --input_dir data/raw/hcc_tace --output_dir data/processed --dataset hcc_tace
python -m src.data.convert_dicom --input_dir data/raw/ircadb --output_dir data/processed --dataset ircadb

# 2. Preprocess all NIfTI volumes
python -m src.data.preprocess --input_dir data/raw/lits --output_dir data/processed/images --mask_dir data/processed/masks

# 3. Validate alignment
python -m src.data.validate_alignment --images_dir data/processed/images --masks_dir data/processed/masks --strict
```
