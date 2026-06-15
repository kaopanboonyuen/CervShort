# 🔬 CervShort: Domain-Aware Shortcut Disruption for Robust Cervical Cancer Cytology Classification

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-CervShort-green.svg)](#)

**Author:** [Teerapong Panboonyuen](https://kaopanboonyuen.github.io/) (Kao Panboonyuen)  
**Affiliation:** Chulalongkorn University · Khon Kaen University, Thailand  
**Funding:** C2F Postdoctoral Fellowship, Chulalongkorn University; Talent Scholarship, KKU

</div>

---

## 📖 Abstract

Deep cervical cytology classifiers often exploit superficial cues such as illumination drift, slide preparation artifacts, or background debris rather than pathological cell morphology, leading to poor generalization across laboratories and imaging setups.

We introduce **CervShort**, a domain-aware shortcut disruption framework employing a **tri-path feature purification module (TFPM)** that:
1. Isolates cell-centric morphological descriptors via adaptive nucleus–cytoplasm segmentation
2. Suppresses artifact-sensitive representations through a degradation-invariance projector
3. Learns distribution-stable features using a prototype-alignment constraint

---

## 🏗️ Architecture Overview

```
Input Cytology Image
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Shortcut-Perturbation Module (SPM)      │
│         Generates controlled artifact perturbations  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│           Tri-Path Feature Purification Module       │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐  │
│  │  Path 1:     │ │  Path 2:     │ │  Path 3:    │  │
│  │  Morphology  │ │  Degradation │ │  Prototype  │  │
│  │  Encoding    │ │  Invariance  │ │  Alignment  │  │
│  │  (NC-Seg)    │ │  Projector   │ │  (Cross-Lab)│  │
│  └──────────────┘ └──────────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│        Cross-Branch Contrastive Alignment Loss       │
└─────────────────────────────────────────────────────┘
        │
        ▼
   Classification Output (NILM / ASC-US / LSIL / HSIL / SCC)
```

---

## 📂 Repository Structure

```
CervShort/
├── cervshort/
│   ├── __init__.py
│   ├── model.py                    # Main CervShort model
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── tfpm.py                 # Tri-Path Feature Purification Module
│   │   ├── morphology_encoder.py   # NC segmentation + masked pooling
│   │   ├── degradation_projector.py# Degradation-invariant projection head
│   │   ├── prototype_alignment.py  # Cross-domain prototype alignment
│   │   └── spm.py                  # Shortcut-Perturbation Module
│   ├── augmentation/
│   │   ├── __init__.py
│   │   └── artifact_shift.py       # Artifact-Shift Adversarial Augmentation
│   └── utils/
│       ├── __init__.py
│       ├── losses.py               # All loss functions
│       ├── metrics.py              # AUC, FPR95, causal metrics
│       └── visualization.py        # GradCAM + attention maps
├── configs/
│   ├── base.yaml                   # Base configuration
│   ├── resnet18.yaml
│   ├── resnet50.yaml
│   ├── densenet121.yaml
│   └── vit_l16.yaml
├── data/
│   ├── __init__.py
│   ├── dataset.py                  # CervicalCytologyDataset
│   └── transforms.py               # Preprocessing pipelines
├── scripts/
│   ├── train.py                    # Main training script
│   ├── evaluate.py                 # Full evaluation suite
│   ├── ablation.py                 # Ablation study (Table 3)
│   └── robustness_eval.py          # Robustness evaluation (Table 4)
├── experiments/
│   ├── reproduce_table1.sh         # Cross-center performance (Table 1)
│   ├── reproduce_table2.sh         # Causal metrics (Table 2)
│   ├── reproduce_table3.sh         # Ablation study (Table 3)
│   ├── reproduce_table4.sh         # Robustness eval (Table 4)
│   └── reproduce_table5.sh         # Cytotechnologist comparison (Table 5)
├── figures/
│   ├── plot_loss_curves.py         # Figure 2: Loss convergence
│   ├── plot_radar_chart.py         # Figure 3 (left): Radar chart
│   ├── plot_decay_curves.py        # Figure 3 (right): Decay curves
│   └── plot_tpr_curves.py          # Figure 7: TPR stability
├── tests/
│   ├── test_tfpm.py
│   ├── test_losses.py
│   └── test_metrics.py
├── environment.yml
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/kaopanboonyuen/CervShort.git
cd CervShort

# Create conda environment
conda env create -f environment.yml
conda activate cervshort

# Or pip install
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Preparation

The dataset comprises **25,412 digitized cytology patches** from 5 independent Thai laboratories, annotated following the Bethesda 2014 guidelines.

```
data/
└── cervical_cytology/
    ├── lab_A/
    │   ├── NILM/
    │   ├── ASC-US/
    │   ├── LSIL/
    │   ├── HSIL/
    │   └── SCC/
    ├── lab_B/ ...
    ├── lab_C/ ...
    ├── lab_D/ ...
    └── lab_E/ ...
```

Preprocess with Macenko color normalization:
```bash
python data/preprocess.py \
    --input_dir /path/to/raw_slides \
    --output_dir data/cervical_cytology \
    --patch_size 256 \
    --normalization macenko
```

### 3. Training

```bash
# Train CervShort with ViT-L/16 backbone
python scripts/train.py \
    --config configs/vit_l16.yaml \
    --backbone vit_l16 \
    --use_cervshort \
    --output_dir outputs/cervshort_vitl16

# Train baseline (no CervShort)
python scripts/train.py \
    --config configs/vit_l16.yaml \
    --backbone vit_l16 \
    --output_dir outputs/baseline_vitl16
```

### 4. Evaluation

```bash
# Full evaluation (all metrics)
python scripts/evaluate.py \
    --checkpoint outputs/cervshort_vitl16/best.pth \
    --config configs/vit_l16.yaml \
    --use_cervshort
```

---

## 📊 Reproducing All Tables and Figures

### Table 1: Cross-Center Diagnostic Performance

```bash
bash experiments/reproduce_table1.sh
```

Expected output:

| Backbone    | Variant      | Avg AUC↑ | Avg FPR↓ |
|-------------|-------------|----------|----------|
| ResNet18    | w/o CervS   | 82.7     | 30.0     |
| ResNet18    | w/ CervShort| **90.5** | **14.3** |
| ResNet50    | w/o CervS   | 85.1     | 27.8     |
| ResNet50    | w/ CervShort| **93.0** | **11.9** |
| DenseNet121 | w/o CervS   | 85.7     | 26.8     |
| DenseNet121 | w/ CervShort| **93.6** | **11.4** |
| ViT-L/16    | w/o CervS   | 87.2     | 25.2     |
| ViT-L/16    | w/ CervShort| **95.9** | **8.7**  |

### Table 2: Causal Consistency Metrics

```bash
bash experiments/reproduce_table2.sh
```

### Table 3: Ablation Study

```bash
bash experiments/reproduce_table3.sh
```

### Table 4: Robustness to Perturbations

```bash
bash experiments/reproduce_table4.sh
```

### Table 5: vs. Cytotechnologists

```bash
bash experiments/reproduce_table5.sh
```

---

## 📈 Reproducing Figures

```bash
# Figure 2: Loss convergence curves
python figures/plot_loss_curves.py \
    --baseline_log outputs/baseline_vitl16/train.log \
    --cervshort_log outputs/cervshort_vitl16/train.log

# Figure 3: Radar chart + decay curves
python figures/plot_radar_chart.py
python figures/plot_decay_curves.py

# Figure 7: Cross-domain TPR curves
python figures/plot_tpr_curves.py
```

---

## ⚙️ Configuration

All hyperparameters are in `configs/`. Key settings:

```yaml
# configs/base.yaml (excerpt)
training:
  optimizer: adamw
  lr: 3e-4           # ViTs; use 1e-3 for CNNs
  batch_size: 128
  epochs: 200
  warmup_epochs: 5
  weight_decay: 0.05
  scheduler: cosine

cervshort:
  lambda_morph: 0.5
  lambda_deg: 0.3
  lambda_proto: 0.2
  lambda_adv: 0.1
  beta_freq: 0.1
  proto_dim: 256
  proto_momentum: 0.97
```

---

## 🧪 Hardware

| Component | Specification       |
|-----------|---------------------|
| GPU       | NVIDIA T4 (16 GiB)  |
| CPU       | Intel Xeon 2.5 GHz  |
| RAM       | 32 GiB              |
| CUDA      | 7.5 (Compute Cap.)  |

Training time: ~14–18 hours per backbone.

---

## 📜 Citation

```bibtex
@article{panboonyuen2026cervshort,
  title     = {CervShort: Domain-Aware Shortcut Disruption for Robust Cervical Cancer Cytology Classification},
  author    = {Panboonyuen, Teerapong},
  year      = {2026},
  note      = {Funded by C2F Postdoctoral Fellowship, Chulalongkorn University}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<b>Teerapong Panboonyuen (Kao Panboonyuen)</b><br>
C2F Postdoctoral Fellow · Chulalongkorn University, Thailand<br>
<a href="https://kaopanboonyuen.github.io/">kaopanboonyuen.github.io</a>
</div>
