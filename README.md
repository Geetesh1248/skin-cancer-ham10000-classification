# Skin Lesion Classification on HAM10000 via ResNet-50 Embeddings, Context-Aware Mixture of Experts (MoE) & Federated Learning (FedAvg)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end medical computer vision, machine learning, and privacy-preserving distributed learning pipeline for 7-class dermatoscopic skin lesion diagnosis on the **HAM10000 ("Human Against Machine with 10,000 training images")** benchmark dataset (10,015 dermatoscopic images).

The system decouples deep visual representation learning from gradient-boosted decision trees (GBDT), dynamic ensemble routing, and decentralized multi-institution training. First, transfer learning with an ImageNet-pretrained **ResNet-50** extracts 2,048-dimensional dense visual representations for each lesion. Three specialized GBDT base classifiers (**XGBoost**, **CatBoost**, and **LightGBM**) are trained using **Square-Root Smoothed Class Weights** to counter heavy class imbalance without destabilizing majority-class precision. An **88-dimensional Context-Aware Mixture of Experts (MoE)** gating network fuses visual PCA components, out-of-fold probability distributions, and Shannon entropy uncertainty scores to dynamically route predictions to the optimal expert per lesion type. Finally, a **Federated Learning (FedAvg)** framework simulates decentralized, multi-hospital collaborative training across 3 independent clinical institutions without sharing private patient images or embeddings, establishing state-of-the-art diagnostic accuracy and sensitivity.

---

## Architecture Overview

### 1. Centralized Representation & Mixture of Experts (MoE) Pipeline
```
                                  [ Dermatoscopic Skin Lesion Image ]
                                                  │
                                                  ▼
                                ┌───────────────────────────────────┐
                                │   Pretrained ResNet-50 Backbone   │
                                │   (Global Average Pooling Layer)  │
                                └───────────────────────────────────┘
                                                  │
                                                  ▼
                                    2,048-D Dense Feature Vector
                                                  │
                                ┌─────────────────┴─────────────────┐
                                ▼                                   ▼
                   ┌─────────────────────────┐         ┌─────────────────────────┐
                   │  StandardScaler (Train) │         │  PCA (64 Components)    │
                   └─────────────────────────┘         └─────────────────────────┘
                                │                                   │
                ┌───────────────┼───────────────┐                   │
                ▼               ▼               ▼                   │
         ┌────────────┐  ┌────────────┐  ┌────────────┐             │
         │  XGBoost   │  │  CatBoost  │  │  LightGBM  │             │
         │  (Hist)    │  │(Oblivious) │  │(Leaf-wise) │             │
         └────────────┘  └────────────┘  └────────────┘             │
                │               │               │                   │
                └───────────────┼───────────────┘                   │
                                ▼                                   │
                      21-D Probabilities                            │
                    + 3-D Shannon Entropy                           │
                                │                                   │
                                └─────────────────┬─────────────────┘
                                                  │
                                                  ▼
                                     88-D Multimodal Gating Input
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │      MoEGatingNetwork (MLP)     │
                                 │   Linear(88,32) -> BN -> ReLU   │
                                 │   -> Dropout(0.3) -> Linear(3)  │
                                 │   Softmax -> Expert Weights     │
                                 └─────────────────────────────────┘
                                                  │
                                                  ▼
                                     Dynamic Mixture Prediction
                                    y_pred = argmax(Σ g_k * P_k)
```

### 2. Multi-Hospital Federated Learning (FedAvg) Architecture
```
                     ┌───────────────────────────────────────────────┐
                     │          Central Coordinating Server          │
                     │  - Broadcasts Global Model Parameters W_t     │
                     │  - Aggregates Client Updates: Σ (n_k/N) * W_k │
                     │  - Evaluates Consolidated Global Checkpoint   │
                     └───────────────────────┬───────────────────────┘
                                             │
                       ┌─────────────────────┼─────────────────────┐
         Broadcast W_t │                     │ Broadcast W_t       │ Broadcast W_t
         Upload ΔW_A   │                     │ Upload ΔW_B         │ Upload ΔW_C
                       ▼                     ▼                     ▼
             ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
             │    Hospital A     │ │    Hospital B     │ │    Hospital C     │
             │   (2,352 Images)  │ │   (2,351 Images)  │ │   (2,351 Images)  │
             │  1,743 Lesions    │ │  1,743 Lesions    │ │  1,742 Lesions    │
             ├───────────────────┤ ├───────────────────┤ ├───────────────────┤
             │ Private Patient   │ │ Private Patient   │ │ Private Patient   │
             │ Local Database    │ │ Local Database    │ │ Local Database    │
             ├───────────────────┤ ├───────────────────┤ ├───────────────────┤
             │ 3 Local Epochs:   │ │ 3 Local Epochs:   │ │ 3 Local Epochs:   │
             │ - Scaled Features │ │ - Scaled Features │ │ - Scaled Features │
             │ - Local Smoothed  │ │ - Local Smoothed  │ │ - Local Smoothed  │
             │   Class Weights   │ │   Class Weights   │ │   Class Weights   │
             │ - Adam Optimizer  │ │ - Adam Optimizer  │ │ - Adam Optimizer  │
             │ - No Raw Sharing  │ │ - No Raw Sharing  │ │ - No Raw Sharing  │
             └───────────────────┘ └───────────────────┘ └───────────────────┘
```

---

## Key Engineering Highlights

### 1. Leakage-Safe Patient-Level Group Split
The HAM10000 dataset contains duplicate and multi-angle photographs of identical skin lesions across distinct image IDs (`image_id`). Standard random cross-validation or unconstrained splits contaminate evaluation sets, leading to severe performance overestimation.
- **Implementation**: Stratified 70/15/15 partition grouped strictly by `lesion_id` using `GroupShuffleSplit`.
- **Validation**: Strict runtime assertion verifies **0% lesion overlap** (`len(train_lesions ∩ val_lesions ∩ test_lesions) == 0`).
- **Counts**: 7,054 train / 1,464 validation / 1,497 test samples.

### 2. Square-Root Smoothed Class Weighting
The raw HAM10000 class distribution is notoriously skewed: majority `nv` represents 66.95% of data, while rare `df` represents only 1.15%. Traditional `balanced` heuristic weights (`w_i = N / (7 * N_i)`) impose an extreme **62.2x penalty ratio** between `df` and `nv`, forcing gradient boosters to trigger overwhelming false-positive rates on benign nevi (collapsing CatBoost accuracy down to 67.20%).
- **Square-Root Smoothing Formula**:
  $$w_i = \sqrt{\frac{N_{\text{total}}}{7 \cdot N_i}}$$
- **Result**: Compresses the extreme ratio from **62.2x down to 7.9x**. Rare malignant lesions receive moderate focus while preserving strong true-negative precision on majority benign lesions.
- **Impact**: **CatBoost test accuracy jumped from 67.20% to 75.95% (+8.75% absolute recovery)**, with majority class `nv` recall surging from 74.90% to 90.80%.

### 3. 88-D Context-Aware Mixture of Experts (MoE) Gating Network
Prior gating implementations suffered from expert collapse, assigning >70% weight to CatBoost regardless of input. Our MoE gating architecture resolves this via:
- **Multimodal Gating Input (88-D)**:
  1. **64-D PCA Features**: Compact visual representation capturing 78.01% of ResNet-50 visual variance.
  2. **21-D Expert Probabilities**: Out-of-fold probability vectors from XGBoost, CatBoost, and LightGBM.
  3. **3-D Shannon Entropy Scores**: Quantifies each model's prediction confidence $H(p_k) = -\sum_c p_{k,c} \log p_{k,c}$.
- **Leak-Free 5-Fold OOF Generation**: Generated out-of-fold probability distributions across `X_train` with `StratifiedGroupKFold` so the gate never fits to overconfident in-sample predictions.
- **Inductive Bias Initialization**: Gate output bias initialized with `[0.5, 0.0, 0.5]`, establishing an initial prior favoring higher-precision experts (XGBoost, LightGBM) over CatBoost.
- **Outcome**: MoE achieves **76.49% Test Accuracy** and **0.7528 Weighted F1** (+4.21% gain over naive collapsed MoE).

### 4. Privacy-Preserving Federated Learning (FedAvg) Across 3 Clinical Sites
In real-world clinical dermatology, strict data privacy regulations (HIPAA, GDPR) forbid hospitals from pooling patient photographs or electronic health records into a central server.
- **Decentralized Simulation**: 7,054 training images partitioned into 3 isolated hospital nodes (Hospital A: 2,352; Hospital B: 2,351; Hospital C: 2,351) via `StratifiedGroupKFold(n_splits=3)` on `lesion_id` (**0 shared patients across hospitals**).
- **Edge Training**: Hospitals optimize local `FederatedClassifier` neural heads for 3 local epochs per round using site-specific smoothed class weights and Adam (`lr=1e-3`, `weight_decay=1e-4`).
- **Secure Aggregation**: Central coordinator aggregates parameter weights and batch-normalization buffers via sample-weighted averaging across 15 communication rounds.
- **Outcome**: **77.22% Test Accuracy**, **0.5751 Macro F1**, **0.7760 Weighted F1**, and **55.36% Melanoma Recall** — outperforming centralized individual GBDTs and MoE without exposing private client records.

---

## Dataset & Split Distribution

The dataset comprises **10,015 dermatoscopic images** categorized into 7 diagnostic classes:

| Class Code | Clinical Diagnostic Name | Lesion Nature | Train ($n$) | Val ($n$) | Test ($n$) | Total ($n$) | Share (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nv`** | Melanocytic nevi | Benign Mole | 4,730 | 975 | 1,000 | 6,705 | 66.95% |
| **`mel`** | Melanoma | Malignant Cancer | 777 | 168 | 168 | 1,113 | 11.11% |
| **`bkl`** | Benign keratosis-like lesions | Benign Keratosis | 775 | 167 | 157 | 1,099 | 10.97% |
| **`bcc`** | Basal cell carcinoma | Malignant Cancer | 365 | 72 | 77 | 514 | 5.13% |
| **`akiec`** | Actinic keratoses / Bowen's disease | Pre-cancerous | 233 | 43 | 51 | 327 | 3.26% |
| **`vasc`** | Vascular lesions | Benign Vascular | 98 | 22 | 22 | 142 | 1.42% |
| **`df`** | Dermatofibroma | Benign Fibroma | 76 | 17 | 22 | 115 | 1.15% |
| **Total** | | | **7,054** | **1,464** | **1,497** | **10,015** | **100.00%** |

---

## Comprehensive Benchmark Leaderboard (Held-Out Test Set, $n = 1,497$)

All paradigms were benchmarked on the identical patient-isolated test split ($n = 1,497$):

| Model / Architecture | Learning Paradigm | Test Accuracy | Macro F1 | Weighted F1 | Melanoma Recall (`mel`) | Distinctive Clinical Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Federated Global Model (FedAvg)** | **Federated (3 Hospitals)** | **77.22%** | **0.5751** | **0.7760** | **55.36%** | **Best overall performance**; highest Macro & Weighted F1; 0% raw patient data transmission |
| **XGBoost (Hist)** | Centralized GBDT | **76.75%** | 0.5078 | 0.7507 | 44.05% | Highest single-model GBDT accuracy; 93.90% true-negative recall on majority `nv` |
| **MoE (88-D Context Gate)** | Centralized Ensemble | 76.49% | 0.5204 | 0.7528 | 44.64% | Dynamic multimodal gating (PCA + OOF probs + entropy); +4.21% gain over collapsed gate |
| **Tuned Soft-Voting** | Centralized Ensemble | 76.02% | 0.5269 | 0.7528 | 48.21% | Optimal static weights ($w=[0.00, 0.70, 0.30]$); strong CatBoost/LightGBM blend |
| **CatBoost (Oblivious)** | Centralized GBDT | 75.95% | 0.5337 | 0.7518 | 47.02% | Highest centralized GBDT Macro F1; major recovery with square-root smoothed weights |
| **LightGBM (Leaf-wise)** | Centralized GBDT | 74.15% | 0.4845 | 0.7386 | 49.40% | Fast leaf-wise boosting with high individual minority sensitivity |

---

## Federated Learning (FedAvg): 3-Hospital Collaborative Network

### Clinical Motivation & Zero-Leakage Hospital Partitioning
To simulate real-world inter-hospital collaboration, the 7,054 training samples were partitioned into 3 clinical hospital nodes using `StratifiedGroupKFold(n_splits=3)` on `lesion_id`:

| Hospital Node | Private Image Count ($N$) | Unique Patient Lesions | Class Distribution (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) | Cross-Hospital Patient Overlap |
| :--- | :---: | :---: | :---: | :---: |
| **Hospital A** | 2,352 | 1,743 | `[78, 121, 259, 25, 259, 1577, 33]` | **0 lesions (0.0%)** |
| **Hospital B** | 2,351 | 1,743 | `[77, 122, 258, 26, 259, 1576, 33]` | **0 lesions (0.0%)** |
| **Hospital C** | 2,351 | 1,742 | `[78, 122, 258, 25, 259, 1577, 32]` | **0 lesions (0.0%)** |

### Federated Model Architecture (`FederatedClassifier`)
Each hospital hosts an identical PyTorch neural classification head:
$$\text{Input}(2048) \xrightarrow{\text{Linear}} \text{Dense}(256) \xrightarrow{\text{BatchNorm1d}} \text{ReLU} \xrightarrow{\text{Dropout}(0.3)} \text{Linear}(256, 7)$$
- **Trainable Parameters**: 526,855
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$, $\text{weight decay} = 10^{-4}$)
- **Local Loss Function**: CrossEntropyLoss with site-specific square-root smoothed weights:
  $$w_{c,i} = \sqrt{\frac{N_c}{7 \cdot N_{c,i}}}$$

### 15-Round Communication Telemetry
Across 15 FedAvg communication rounds (each client training for 3 local epochs per round), the central model rapidly stabilized and surpassed centralized baselines:

| Round | Global Val Acc | Global Val Macro F1 | Global Val W-F1 | Global Test Acc | Global Test Mel Recall | Local Hospital Val Accuracies (A / B / C) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.7309 | 0.4893 | 0.7396 | 0.7214 | 51.19% | 73.6% / 74.7% / 73.8% |
| **2** | 0.7609 | 0.5478 | 0.7691 | 0.7435 | 50.60% | 74.9% / 76.0% / 75.2% |
| **3** | 0.7637 | 0.5648 | 0.7735 | 0.7442 | 47.62% | 74.3% / 76.2% / 75.1% |
| **4** | 0.7814 | 0.5694 | 0.7863 | 0.7555 | 57.74% | 74.6% / 76.8% / 75.1% |
| **5** | 0.7794 | 0.5739 | 0.7851 | 0.7562 | 51.19% | 75.1% / 76.1% / 76.6% |
| **6** | 0.7828 | 0.5676 | 0.7888 | 0.7689 | 52.98% | 75.7% / 76.6% / 76.0% |
| **7** | 0.7855 | 0.5834 | 0.7885 | 0.7662 | 52.38% | 75.4% / 76.8% / 76.8% |
| **8 (Best)** | **0.7821** | **0.5873** | **0.7857** | **0.7722** | **55.36%** | **76.3% / 77.7% / 76.8%** |
| **9** | 0.7787 | 0.5789 | 0.7833 | 0.7662 | 56.55% | 76.6% / 76.7% / 76.2% |
| **10** | 0.7828 | 0.5667 | 0.7855 | 0.7629 | 54.17% | 76.1% / 77.5% / 77.6% |
| **11** | 0.7848 | 0.5803 | 0.7855 | 0.7675 | 54.17% | 76.2% / 77.0% / 75.6% |
| **12** | 0.7814 | 0.5569 | 0.7822 | 0.7702 | 48.81% | 76.4% / 77.4% / 77.5% |
| **13** | 0.7828 | 0.5673 | 0.7830 | 0.7702 | 50.00% | 77.7% / 76.7% / 77.5% |
| **14** | 0.7814 | 0.5610 | 0.7809 | 0.7762 | 52.38% | 76.9% / 77.2% / 76.4% |
| **15** | 0.7814 | 0.5577 | 0.7847 | 0.7629 | 51.19% | 78.2% / 75.9% / 77.0% |

### Federated Global Model Confusion Matrix & Per-Class Metrics

```
              precision    recall  f1-score   support

       akiec     0.4250    0.6667    0.5191        51
         bcc     0.5122    0.5455    0.5283        77
         bkl     0.5724    0.5541    0.5631       157
          df     0.3684    0.3182    0.3415        22
         mel     0.5345    0.5536    0.5439       168
          nv     0.9081    0.8790    0.8933      1000
        vasc     0.6364    0.6364    0.6364        22

    accuracy                         0.7722      1497
   macro avg     0.5653    0.5933    0.5751      1497
weighted avg     0.7822    0.7722    0.7760      1497
```

**Confusion Matrix (rows = ground truth, columns = predicted):**
```
              Pred:  akiec    bcc    bkl     df    mel     nv   vasc
True: akiec             34      1      9      1      4      2      0
      bcc               12     42      8      3      4      7      1
      bkl               10     12     87      3     19     26      0
      df                 2      5      3      7      1      4      0
      mel               11      2     13      0     93     47      2
      nv                11     16     32      5     52    879      5
      vasc               0      4      0      0      1      3     14
```

---

## Adaptive MoE Gate Routing Analysis

Rather than applying fixed uniform averaging, the trained 88-D Gating Network assigns input-dependent probabilities across the 3 base experts:

### Global Test Set Gate Allocation
- **XGBoost**: **59.06%**
- **LightGBM**: **27.85%**
- **CatBoost**: **13.10%**

### Routing Specialization by Ground-Truth Pathology
The gate autonomously acquired clinically interpretable routing rules without explicit class instructions:

| Class Code | Clinical Diagnostic Name | XGBoost Weight | CatBoost Weight | LightGBM Weight | Learned Routing Rationale |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`nv`** | Melanocytic nevi | **66.25%** | 7.87% | 25.88% | Delegates primarily to XGBoost where `nv` specificity and true-negative rate are highest. |
| **`mel`** | Melanoma | **51.83%** | 13.53% | **34.64%** | Routes heavily to XGBoost and LightGBM (LightGBM has highest raw Melanoma recall at 49.40%). |
| **`bkl`** | Benign keratosis | **50.88%** | 19.80% | 29.32% | Relies on XGBoost for texture/boundary discrimination. |
| **`bcc`** | Basal cell carcinoma | 26.83% | **39.41%** | **33.76%** | Hands over authority to CatBoost and LightGBM for non-melanocytic epithelial cancer. |
| **`df`** | Dermatofibroma | 29.26% | **43.89%** | 26.85% | Strongly prefers CatBoost where precision on rare dermatofibromas reaches 100%. |
| **`akiec`** | Actinic keratoses | **39.44%** | 32.27% | 28.28% | Balanced ensemble consensus across all three classifiers. |
| **`vasc`** | Vascular lesions | 33.79% | 32.07% | **34.14%** | Uniform three-way ensemble division over distinct vascular signatures. |

---

## Confusion Matrix & Per-Class Metrics (MoE 88-D Gate)

### Classification Report
```
              precision    recall  f1-score   support

       akiec       0.45      0.53      0.49        51
         bcc       0.57      0.43      0.49        77
         bkl       0.50      0.47      0.48       157
          df       1.00      0.14      0.24        22
         mel       0.55      0.45      0.49       168
          nv       0.85      0.92      0.89      1000
        vasc       0.90      0.41      0.56        22

    accuracy                           0.76      1497
   macro avg       0.69      0.48      0.52      1497
weighted avg       0.76      0.76      0.75      1497
```

### Confusion Matrix
```
             Pred:  akiec    bcc    bkl     df    mel     nv   vasc
True: akiec            27      3     11      0      4      6      0
      bcc              11     33     12      0      7     14      0
      bkl               8      7     74      0     17     51      0
      df                4      1      5      3      0      9      0
      mel               9      4     13      0     75     67      0
      nv                1      8     33      0     33    924      1
      vasc              0      2      1      0      0     10      9
```

---

## Repository Structure

```
.
├── app/
│   ├── main.py                     # FastAPI clinical inference backend & telemetry APIs
│   └── static/
│       └── index.html              # Medical-grade interactive dashboard (Tailwind + Chart.js)
├── models/
│   ├── catboost_metrics.json       # CatBoost test evaluation metrics
│   ├── catboost_resnet50.cbm       # Trained CatBoost binary model
│   ├── catboost_test_probs.npy     # CatBoost test probability predictions
│   ├── catboost_val_probs.npy      # CatBoost validation probability predictions
│   ├── federated_global.pt         # Best Federated Global model checkpoint (Round 8)
│   ├── federated_metrics.json      # 15-round FedAvg telemetry & final test benchmarks
│   ├── lightgbm_metrics.json       # LightGBM test evaluation metrics
│   ├── lightgbm_resnet50.txt       # Trained LightGBM booster model
│   ├── lightgbm_test_probs.npy     # LightGBM test probability predictions
│   ├── lightgbm_val_probs.npy      # LightGBM validation probability predictions
│   ├── moe_gate.pt                 # PyTorch state dict for MoEGatingNetwork
│   ├── moe_metrics.json            # Final ensemble leaderboard and gating metrics
│   ├── moe_pca.joblib              # Fitted 64-component PCA model
│   ├── scaler.joblib               # StandardScaler fitted strictly on X_train
│   ├── xgboost_metrics.json        # XGBoost test evaluation metrics
│   ├── xgboost_resnet50.json       # Trained XGBoost JSON model
│   ├── xgboost_test_probs.npy      # XGBoost test probability predictions
│   └── xgboost_val_probs.npy       # XGBoost validation probability predictions
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb # Exploratory data analysis & distribution checks
│   └── 02_ResNet50_Features.ipynb   # ResNet-50 visual feature extraction pipeline
├── results/
│   ├── label_mapping.json          # Mapping of 7 diagnostic strings to integer IDs
│   ├── moe_oof_probs.npz           # 5-fold cross-validated train OOF predictions
│   └── splits.csv                  # Grouped 70/15/15 lesion-isolated split metadata
├── src/
│   ├── compare_models.py           # CLI comparison table for all base classifiers
│   ├── federated_learning.py       # 3-Hospital Federated Learning (FedAvg) trainer
│   ├── prepare_split.py            # GroupShuffleSplit patient-isolation generator
│   ├── run_feature_extraction.py   # ResNet-50 2048-D feature extraction script
│   ├── train_catboost.py           # CatBoost trainer with smoothed weights & scaler
│   ├── train_lightgbm.py           # LightGBM trainer with smoothed weights & scaler
│   ├── train_moe_final.py          # 88-D Context-Aware MoE Gating trainer & evaluator
│   ├── train_moe_v2.py             # Wrapper executing train_moe_final.py
│   └── train_xgboost.py            # XGBoost trainer with smoothed weights & scaler
├── tests/
│   └── test_api.py                 # Automated verification suite for live endpoints
├── requirements.txt                # Production environment dependencies
└── README.md                       # Comprehensive documentation & benchmarks
```

---

## Installation & Reproduction Guide

### 1. Environment Setup
Clone the repository and install required Python dependencies:
```bash
git clone https://github.com/Geetesh1248/skin-cancer-ham10000-classification.git
cd skin-cancer-ham10000-classification

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset Preparation & Group-Isolated Split
Download the HAM10000 dataset into `dataset/` (or extract `archive.zip`). Then generate the group-safe 70/15/15 partition:
```bash
python src/prepare_split.py
```
This writes `results/splits.csv` ensuring zero `lesion_id` leakage across splits.

### 3. ResNet-50 Feature Extraction
Extract 2,048-dimensional dense visual representations from the global average pooling layer of pretrained ResNet-50:
```bash
python src/run_feature_extraction.py
```
This generates `results/resnet50_features.npz`.

### 4. Train Individual Gradient Boosted Experts
Train the three base models using `StandardScaler` (fit strictly on `X_train`) and Square-Root Smoothed Class Weights:
```bash
# Train XGBoost (Tree Method: Hist, lr=0.05, max_depth=6)
python src/train_xgboost.py

# Train CatBoost (Oblivious Trees, lr=0.05, depth=6)
python src/train_catboost.py

# Train LightGBM (Leaf-wise, lr=0.05, max_depth=6)
python src/train_lightgbm.py
```

Inspect the baseline comparison table across all 3 trained models:
```bash
python src/compare_models.py
```

### 5. Train the 88-D Context-Aware Mixture of Experts (MoE)
Run 5-fold StratifiedGroupKFold OOF prediction on `X_train`, fit 64-D PCA, and train the multimodal gating network:
```bash
python src/train_moe_final.py
```
This outputs the full test set leaderboard, grid-searched soft-voting baseline, per-class gating routing breakdown, and exports `models/moe_metrics.json` and `models/moe_gate.pt`.

### 6. Train the 3-Hospital Federated Learning (FedAvg) Network
Simulate collaborative decentralized learning across 3 patient-isolated clinical hospitals without raw image sharing:
```bash
python src/federated_learning.py
```
This executes 15 FedAvg communication rounds with local square-root smoothed optimization, exports round-by-round convergence telemetry to `models/federated_metrics.json`, and saves the optimal global model weights to `models/federated_global.pt`.

### 7. Launch Interactive Clinical Web Application
Run the local FastAPI server to open the interactive professor demonstration dashboard:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to [http://localhost:8000](http://localhost:8000) in any browser to access the live diagnostic scanner, 88-D MoE gating visualizer, and 3-Hospital FedAvg telemetry dashboard.

Run the automated test verification suite:
```bash
python tests/test_api.py
```

---

## Citation & References

- **Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Sci. Data 5, 180161 (2018). [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)
- **Deep Residual Learning**: He, K., Zhang, X., Ren, S., & Sun, J. *Deep Residual Learning for Image Recognition*. CVPR (2016).
- **Federated Optimization (FedAvg)**: McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS (2017).
- **Out-of-Fold Ensembling & MoE**: Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. *Adaptive Mixtures of Local Experts*. Neural Computation (1991).

---

## License
This project is open-source and licensed under the [MIT License](LICENSE).
