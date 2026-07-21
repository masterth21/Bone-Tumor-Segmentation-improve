# 🦴 NaLaFormer-TransUNet: Vision Linear Attention for Bone Tumor Segmentation

[![Paper - NaLaFormer](https://img.shields.io/badge/Paper-NaLaFormer_(arXiv_2025/2026)-red.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2506.21137)
[![Dataset - BTXRD](https://img.shields.io/badge/Dataset-BTXRD_(Nature_2024)-blue.svg?style=for-the-badge&logo=nature)](https://www.nature.com/articles/s41597-024-04311-y)
[![Base Code - UNet 3+](https://img.shields.io/badge/Base-UNet_3+_Repo-black.svg?style=for-the-badge&logo=github)](https://github.com/hamidriasat/UNet-3-Plus)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

Official implementation of **NaLaFormer-TransUNet** — integrating **Norm×Direction: Q Norm-Aware Vision Linear Attention** into a Hybrid CNN-Transformer architecture for high-precision **Bone Tumor Segmentation & Classification (Benign vs. Malignant)** on the **BTXRD Dataset**.

---

## 📑 Table of Contents

- [✨ Key Innovations](#-key-innovations)
- [🧮 Mathematical Formulation](#-mathematical-formulation)
- [🏗️ Architecture Overview](#%EF%B8%8F-architecture-overview)
- [📊 Dataset & Segmentation Classes](#-dataset--segmentation-classes)
- [🚀 Quick Start](#-quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Dataset Preparation](#2-dataset-preparation)
  - [3. Training](#3-training)
- [📁 Repository Structure](#-repository-structure)
- [⚙️ Configuration Reference](#%EF%B8%8F-configuration-reference)
- [📚 References & Citation](#-references--citation)
- [📜 License](#-license)

---

## ✨ Key Innovations

Standard Vision Transformers (ViT) and Softmax Attention suffer from **$\mathcal{O}(N^2)$ quadratic complexity**, making high-resolution medical image segmentation computationally prohibitive. While standard Linear Attentions achieve $\mathcal{O}(N)$ complexity, they remove Query normalization, destroying query-norm-entropy correlation ("spikiness").

**NaLaFormer** ([arXiv:2506.21137](https://arxiv.org/abs/2506.21137)) solves this with three core contributions:

1. **Q Norm-Aware Linear Attention**: Restores missing query-norm dependency in linear space without Softmax.
2. **Cosine Direction Feature Mapping ($\phi_c$)**: Enforces strict non-negativity by decomposing vectors into magnitude and cosine direction encodings.
3. **Gated Activation Branch ($\text{ACT}[G]$)**: Dynamically gates features via $f(x) = \lambda \cdot (\tau + \tanh(x))$.
4. **Sub-quadratic Efficiency**: Reduces bottleneck attention complexity from $\mathcal{O}(N^2 \cdot d)$ to **$\mathcal{O}(N \cdot d^2)$**.

---

## 🧮 Mathematical Formulation

### 1. Feature Maps ($\phi_q, \phi_k, \phi_c$)
The input Query ($Q$) and Key ($K$) vectors are transformed into non-negative feature maps via norm-direction decomposition:

$$\phi_q(q) = |d(q)^{\|q\|}| \odot \big[\cos(\text{dir}(q));\, \sin(\text{dir}(q))\big]$$

$$\phi_k(k) = |k^\lambda| \odot \big[\cos(\text{dir}(k));\, \sin(\text{dir}(k))\big]$$

where $\text{dir}(x) = \frac{x}{\|x\| + \epsilon}$ and $\lambda$ is a learnable scaling parameter.

### 2. Linear Attention Computation
Rather than computing the $N \times N$ matrix $Q K^T$, associativity is exploited:

$$S = (K')^T V \in \mathbb{R}^{2d \times d}$$

$$\text{Attn}(Q, K, V) = \frac{Q' \cdot S}{Q' \cdot \sum K'}$$

### 3. Gated Output Projection
$$\text{Output} = \text{LayerNorm}\Big(\text{Attn} \odot \text{ACT}[G]\Big) W_O$$

---

## 🏗️ Architecture Overview

```
                          NaLaFormer-TransUNet Pipeline
                          
   Input Image (384×384×3)
             │
   ┌─────────┴─────────┐
   │ ResNet50V2 Encoder│  ───► Skip s1 (192×192×64)   ────────┐
   │    (Backbone)     │  ───► Skip s2 (96×96×256)    ──────┐ │
   └─────────┬─────────┘  ───► Skip s3 (48×48×512)    ────┐ │ │
             │            ───► Skip s4 (24×24×1024)   ──┐ │ │ │
      CNN Out (12×12×2048)                              │ │ │ │
             │                                          │ │ │ │
  ┌──────────▼──────────┐                               │ │ │ │
  │ NaLaFormer Bottleneck│                               │ │ │ │
  │  (Linear Attention) │ ◄── N × NaLaFormer Blocks     │ │ │ │
  └──────────┬──────────┘                               │ │ │ │
             │                                          │ │ │ │
   ┌─────────▼─────────┐                                │ │ │ │
   │  U-Net Decoder    │ ◄──────────────────────────────┴─┴─┴─┘
   │ + Skip Connection │
   └─────────┬─────────┘
             │
   Output Mask (384×384×3)  [Background / Benign / Malignant]
```

---

## 📊 Dataset & Segmentation Classes

The pipeline is configured for the **[BTXRD Dataset](https://www.nature.com/articles/s41597-024-04311-y)** (Scientific Data / Nature) with **3-Class Fine-Grained Segmentation**:

| Class ID | Pixel Value (Visual Mask) | Class Category | Original BTXRD Labels Included |
| :---: | :---: | :--- | :--- |
| **0** | `0` (Black) | **Background** | Non-tumor tissue / Background |
| **1** | `128` (Gray) | **Benign Tumor (BT)** | `osteochondroma`, `multiple osteochondromas`, `simple bone cyst`, `giant cell tumor`, `synovial osteochondroma`, `osteofibroma`, `other bt` |
| **2** | `255` (White) | **Malignant Tumor (MT)** | `osteosarcoma`, `other mt` |

*Note: Mask generation uses **polygon contours** exclusively to guarantee boundary precision.*

---

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and install required dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/nalaformer-transunet.git
cd nalaformer-transunet

# Install dependencies
pip install tensorflow hydra-core
```

### 2. Dataset Preparation

Convert LabelMe JSON annotations to polygon multi-class masks and split into Train/Val sets (80/20):

```bash
python prepare_data.py
```

*Output dataset structure:*
```
D:/TumorBone/data/BTXRD/split/
├── train/
│   ├── images/  (1,493 images)
│   └── mask/    (1,493 masks)
└── val/
    ├── images/  (374 images)
    └── mask/    (374 masks)
```

### 3. Training

Launch model training with automatic mixed precision (AMP) and learning rate scheduling:

```bash
python train.py
```

---

## 📁 Repository Structure

```
.
├── configs/
│   └── config.yaml               # Centralized Hydra configuration file
├── models/
│   ├── nalaformer_attention.py   # NaLaFormer Q Norm-Aware Attention core implementation
│   ├── model.py                  # Model factory & NaLaFormer-TransUNet builder
│   ├── backbones.py              # VGG & ResNet backbone extractors
│   ├── unet3plus.py              # UNet 3+ implementation
│   └── unet3plus_deep_supervision.py
├── data_generators/
│   ├── data_generator.py         # Multi-backend generator dispatcher
│   └── tf_data_generator.py      # TF sequence data generator with Data Augmentation
├── data_preparation/
│   └── verify_data.py            # Dataset integrity verification script
├── losses/
│   ├── unet_loss.py              # Class-weighted Dice Loss
│   └── loss.py                  # Focal, SSIM, and IoU loss functions
├── prepare_data.py               # LabelMe JSON -> 3-Class Polygon Mask converter & Splitter
├── train.py                      # Main training pipeline script
└── README.md                     # Project documentation
```

---

## ⚙️ Configuration Reference

Key hyperparameter controls in [`configs/config.yaml`](file:///d:/TumorBone/code/Unet3+/unet-/configs/config.yaml):

```yaml
MODEL:
  TYPE: "nalaformer_transunet"
  NALAFORMER:
    D_MODEL: 256       # Hidden dimension size
    DEPTH: 4           # Stacked NaLaFormer block count
    NUM_HEADS: 8       # Multi-head attention count
    FF_EXPANSION: 4    # Feed-forward expansion factor (4x)
    DROPOUT_RATE: 0.1  # Dropout rate

INPUT:
  HEIGHT: 320
  WIDTH: 320
  CHANNELS: 3

OUTPUT:
  CLASSES: 3          # [0: Background, 1: Benign, 2: Malignant]

HYPER_PARAMETERS:
  EPOCHS: 100
  BATCH_SIZE: 16
  LEARNING_RATE: 5e-5

OPTIMIZATION:
  AMP: True            # Automatic Mixed Precision
  XLA: True            # Accelerated Linear Algebra JIT
```

---

## 📚 References & Citation

1. **NaLaFormer (Attention Mechanism)**:
   > *Norm×Direction: Restoring the Missing Query Norm in Vision Linear Attention*  
   > Paper: [arXiv:2506.21137](https://arxiv.org/abs/2506.21137)

2. **BTXRD Dataset**:
   > *Bone Tumor X-Ray Dataset for Medical Image Segmentation and Classification*  
   > Scientific Data (Nature, 2024). DOI: [10.1038/s41597-024-04311-y](https://www.nature.com/articles/s41597-024-04311-y)

3. **UNet 3+ Baseline**:
   > *UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation*  
   > Codebase reference: [hamidriasat/UNet-3-Plus](https://github.com/hamidriasat/UNet-3-Plus)

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
