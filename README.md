# Title: Unsupervised-Image-Anomaly-Detection-and-Localization-Using-Autoencoder-and-U-Net-Architectures


## Architecture

```
Input Image
    │
    ▼
┌──────────────────────┐
│  Convolutional AE    │  ← trained on defect-free images only
│  + ResNet-18 feature │    (pixel + perceptual + feature-bank scoring)
│    bank scoring      │
└──────────┬───────────┘
           │ anomaly score > threshold?
    ┌──────┴──────┐
    │ No          │ Yes
    ▼             ▼
  PASS    ┌──────────────┐
          │    U-Net     │  ← trained on real defect masks with augmentation
          │ segmentation │
          └──────┬───────┘
                 │
          ┌──────┴──────┐
          │ area > 0.1% │
          ▼             ▼
       REVIEW        REJECT
     (borderline)  (defect localized)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **ConvAutoencoder** | Learns to reconstruct normal textures; defects produce high reconstruction error |
| **ResNet-18 Feature Extractor** | Frozen ImageNet features for perceptual loss and feature-bank distance scoring |
| **U-Net** | Pixel-level binary segmentation of defect regions |
| **VisualInspectionAgent** | Orchestrates the two-stage pipeline with threshold gating |
| **Grad-CAM** | Explainability — highlights which regions drove the segmentation prediction |

### Anomaly Scoring (3-signal fusion)

The detection gate combines three z-normalised signals:
1. **Pixel reconstruction error** — local max-pooled L1 distance
2. **Feature reconstruction error** — MSE in ResNet-18 feature space
3. **Feature-bank distance** — Mahalanobis-style distance from normal training distribution

Weights: `0.3 × pixel + 0.3 × feature + 1.4 × bank` (bank is strongest standalone signal).

---

## Project Structure

```
agentic-visual-inspection/
├── config.py              # All hyperparameters and paths
├── train.py               # End-to-end training script
├── evaluate.py            # Full held-out evaluation suite
├── app.py                 # Gradio interactive demo
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── __init__.py
│   └── dataset.py         # NormalImagesDataset, TestDataset, RealMaskDataset, ...
│
├── models/
│   ├── __init__.py
│   ├── autoencoder.py     # ConvAutoencoder
│   ├── unet.py            # U-Net segmentation model
│   └── feature_extractor.py  # Frozen ResNet-18 early layers
│
├── utils/
│   ├── __init__.py
│   ├── scoring.py         # Anomaly scoring (pixel, feature, bank, combined)
│   ├── metrics.py         # Threshold selection, pixel metrics, dataset scorers
│   ├── visualization.py   # Confusion matrix, ROC curve, figure rendering
│   └── gradcam.py         # U-Net Grad-CAM explainability
│
├── agent/
│   ├── __init__.py
│   └── inspector.py       # VisualInspectionAgent (PASS / REVIEW / REJECT)
│
├── outputs/               # Generated at runtime (gitignored)
│   ├── agentic_vision_checkpoint.pt
│   └── agent_quantitative_report.csv
│
└── notebooks/
    └── agentic-ai-wood.ipynb  # Original Kaggle notebook (single-file version)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the MVTec AD dataset

Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place the category folder so that `data/<category>/train/good/` exists. For the default `wood` category:

```
data/
└── wood/
    ├── train/good/         # defect-free training images
    ├── test/
    │   ├── good/           # normal test images
    │   ├── color/          # defect type 1
    │   ├── combined/       # defect type 2
    │   └── ...
    └── ground_truth/       # pixel-level masks
```

On **Kaggle**, add the dataset `ipythonx/mvtec-ad` via *Add Input* — the code auto-discovers it under `/kaggle/input`.

### 3. Train

```bash
python train.py
```

Trains the autoencoder (40 epochs) and U-Net (60 epochs), calibrates thresholds, and saves everything to `outputs/agentic_vision_checkpoint.pt`.

### 4. Evaluate

```bash
python evaluate.py
```

Runs the full held-out evaluation: classification reports, confusion matrices, ROC curves, per-image IoU/Dice CSV, and a leave-one-defect-type-out generalization check.

### 5. Interactive demo

```bash
python app.py --share
```

Opens a Gradio web UI where you can upload part images and see: predicted mask, affected-region overlay, pixel boundary, AE error heatmap, U-Net probability map, Grad-CAM, and 3D error surface.

---

## Evaluation Outputs

The pipeline produces paper-grade evidence at every stage:

| Stage | Metrics |
|-------|---------|
| **Autoencoder** | Reconstruction samples, train/val loss curve, classification report, confusion matrix, ROC + AUROC |
| **U-Net** | Predicted masks vs ground truth, train/val loss curve, pixel-level classification report, confusion matrix, pixel ROC + AUROC |
| **Agent** | Full visual analysis (8 panels), 3D error surface, per-image CSV (precision/recall/F1/IoU/Dice), image-level AUROC, leave-one-type-out generalization |

---

## Configuration

Edit `config.py` to change the category or training hyperparameters:

```python
CATEGORY    = "wood"     # any MVTec AD category
IMG_SIZE    = 128
AE_EPOCHS   = 40
UNET_EPOCHS = 60
LR          = 1e-3
```

---

## Kaggle Notebook

The original single-file Kaggle notebook (with all outputs) is in `notebooks/agentic-ai-wood.ipynb`. It runs end-to-end on a Kaggle GPU (T4/P100) with the dataset `ipythonx/mvtec-ad` added as input.

---

## License

This project is released for educational and research purposes. The MVTec AD dataset has its own [license terms](https://www.mvtec.com/company/research/datasets/mvtec-ad).
