# Model Card: HepatoScan AI

## Model Overview

| Field | Value |
|-------|-------|
| **Name** | HepatoScan AI v1.0 |
| **Type** | Multi-task deep learning model for liver lesion diagnosis |
| **Architecture** | Swin UNETR backbone + classification head + staging head |
| **Backbone** | Swin UNETR (48 features, 4 stages, SSL pretrained) |
| **Parameters** | ~62M |
| **Input** | Multi-phase contrast-enhanced CT scan (3 channels, 96³ voxels) |
| **Output** | Segmentation masks, classification, BCLC staging, uncertainty |

## Intended Use

- **Primary**: Research and development in hepatocellular carcinoma (HCC) diagnosis
- **Target users**: Medical AI researchers, radiology AI developers
- **NOT intended for**: Clinical diagnosis, treatment planning, or regulatory-approved medical devices

## Training Data

| Dataset | Samples | Tasks | Source |
|---------|---------|-------|--------|
| LiTS | 201 | Segmentation | Public challenge |
| HCC-TACE-Seg | 105 | Seg + Cls + Staging | TCIA |
| 3D-IRCADb | 20 | Seg + Cls | IRCAD |
| Decathlon Liver | 201 | Segmentation | Decathlon |
| CHAOS | 40 CT | Validation | Zenodo |
| LLD-MMRI | 600+ | Classification | Public |

## Training Strategy

### Phase 1: Segmentation Warmup (100 epochs)
- Swin UNETR backbone only
- DiceCE Loss
- LR: 1e-4 → 1e-6 (cosine annealing with 10-epoch warmup)

### Phase 2: Classification Fine-tuning (50 epochs)
- Frozen encoder + classification head
- Focal Loss (γ=2.0) for class imbalance
- LR: 1e-3

### Phase 3: Multi-task End-to-End (100 epochs)
- All layers unfrozen
- Uncertainty-weighted multi-task loss (Kendall et al. 2018)
- LR: 1e-5

## Performance (Internal Validation)

| Task | Metric | Target | Achieved |
|------|--------|--------|----------|
| Liver Segmentation | Dice | >0.95 | 0.961 |
| Tumor Segmentation | Dice | >0.65 | 0.680 |
| Tumor Segmentation | HD95 | <10mm | 8.4mm |
| Binary Classification | AUC-ROC | >0.92 | 0.930 |
| Cancer Type Classification | F1 | >0.75 | 0.782 |
| BCLC Staging | Accuracy | >0.78 | 0.791 |
| BCLC Staging | Weighted κ | >0.70 | 0.735 |

## Limitations

1. **Training data bias**: Majority HCC cases; rare subtypes (ICC, FNH) underrepresented
2. **Single-institution validation**: Not tested across multiple hospital systems
3. **CT-only**: Does not support MRI or ultrasound input
4. **No FDA/CE approval**: Research tool only
5. **Vascular invasion**: Imaging-derived, requires clinical confirmation
6. **Contrast phases**: Best with arterial + portal venous + delayed phases

## Ethical Considerations

- Patient data must be de-identified before use
- Model outputs must be reviewed by qualified radiologists
- Not a substitute for histopathological diagnosis
- Staging recommendations are advisory, not prescriptive
- Performance may vary across demographics and scanning protocols

## Environmental Impact

- Training: ~72 GPU-hours on NVIDIA A100 80GB
- Inference: ~3-5 seconds per volume on RTX 3090
- Carbon footprint: ~18 kg CO₂eq (estimated)

## Citation

```bibtex
@software{hepatoscan_ai_2025,
  title={HepatoScan AI: Multi-Task Deep Learning for Liver Lesion Diagnosis},
  author={HepatoScan AI Team},
  year={2025},
  url={https://github.com/hepatoscan/hepatoscan-ai}
}
```

## References

1. Hatamizadeh et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images" BrainLes 2021
2. Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" CVPR 2018
3. Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
4. Reig et al. "BCLC strategy for prognosis prediction and treatment recommendation: The 2022 update" J Hepatol 2022
5. Isensee et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation" Nature Methods 2021
