# CervShort  
**Domain-Aware Shortcut Disruption for Robust Cervical Cytology Classification**

<p align="center">
  <strong>Author:</strong> Teerapong Panboonyuen &nbsp;•&nbsp;
  <em>Under Review 2026</em>
</p>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/Paper-Under%20Review-blue.svg" alt="Paper">
  </a>
  <a href="https://kaopanboonyuen.github.io/CervShort/">
    <img src="https://img.shields.io/badge/Project-Website-brightgreen.svg" alt="Project">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-Research%20Only-lightgrey.svg" alt="License">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Status-Code%20Release%20Pending-orange.svg" alt="Status">
  </a>
</p>

<p align="center">
  <img src="img/CervShort_MAIN.png" width="90%">
</p>

---

## 🔬 Overview

Deep learning systems for cervical cytology often achieve high in-domain accuracy while silently relying on **non-pathological shortcuts**, such as staining variations, illumination drift, slide preparation artifacts, or background debris. These hidden dependencies can severely degrade performance when models are deployed across hospitals, scanners, or laboratories.

**CervShort** is a domain-aware training framework designed to **explicitly disrupt shortcut learning** in cervical cancer cytology classifiers. Instead of encouraging models to fit dataset-specific cues, CervShort promotes **morphology-grounded, distribution-stable representations** that generalize across clinical centers and imaging conditions.

---

## ✨ Key Ideas

CervShort introduces a principled strategy for shortcut mitigation tailored to cytology data:

### 1. Tri-Path Feature Purification

The network disentangles competing sources of information by learning three complementary feature streams:

* **Cell-centric morphology path** focuses on nucleus–cytoplasm structure and cellular shape.
* **Artifact-suppression path** reduces sensitivity to acquisition-dependent degradations.
* **Distribution-stable path** enforces alignment across staining and illumination variations.

### 2. Artifact-Shift Augmentation

To expose hidden shortcuts during training, CervShort simulates **realistic lab-dependent perturbations** and treats them as adversarial signals, encouraging the model to ignore spurious correlations.

### 3. Robustness-Oriented Learning Objective

Prototype alignment and invariance constraints are applied to promote **cross-center consistency** without sacrificing diagnostic sensitivity.

---

## 🧠 Why It Matters

* Improves **generalization across hospitals and scanners**
* Reduces **artifact-driven misclassification**
* Produces **interpretable attention maps** grounded in cellular morphology
* Addresses reliability and fairness concerns in **population-scale cervical cancer screening**

CervShort is designed with **real clinical deployment** in mind, where robustness is as critical as accuracy.

---

## 📊 Experimental Highlights

* Evaluated on large-scale cervical cytology datasets with multi-center variation
* Consistently outperforms standard ERM and augmentation-based baselines under domain shift
* Demonstrates strong resilience to staining, illumination, and background artifacts

(Full experimental details are provided in the accompanying paper.)

---

## 📁 Repository Structure (Planned)

```
CervShort/
├── img/                    # Figures and visualizations
│   └── CervShort_MAIN.png
├── configs/                # Training and evaluation configurations
├── models/                 # Network architectures
├── datasets/               # Dataset loaders and preprocessing
├── training/               # Training loops and loss definitions
├── evaluation/             # Robustness and cross-domain evaluation
└── README.md
```

---

## 📌 Paper & Citation

This repository accompanies the submission:

> **CervShort: Domain-Aware Shortcut Disruption for Robust Cervical Cancer Cytology Classification**

A BibTeX entry will be released upon acceptance.

---

## ⚠️ Clinical Disclaimer

This code is intended **for research purposes only** and is **not a certified medical device**. Any clinical deployment requires regulatory approval and independent validation.

---

## 👤 Author

**Teerapong Panboonyuen**

- 📧 [teerapong.panboonyuen@gmail.com](mailto:teerapong.panboonyuen@gmail.com)
- 🔗 Project page: [https://kaopanboonyuen.github.io/CervShort/](https://kaopanboonyuen.github.io/CervShort/)
- 🆔 ORCID: 0000-0001-8464-4476

---

## 🌱 Status

🚧 Code release in progress
⭐ Star the repository to receive updates

---