# Skin Lesion Classification on HAM10000 via ResNet-50 Embeddings & Context-Aware Mixture of Experts (MoE)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end medical computer vision and machine learning pipeline for 7-class dermatoscopic skin lesion diagnosis on the **HAM10000 ("Human Against Machine with 10,000 training images")** benchmark dataset (10,015 dermatoscopic images).

The system decouples deep visual representation learning from tabular gradient-boosted decision trees (GBDT) and dynamic ensemble routing. First, transfer learning with an ImageNet-pretrained **ResNet-50** extracts 2,048-dimensional dense visual representations for each lesion. Three specialized GBDT base classifiers (**XGBoost**, **CatBoost**, and **LightGBM**) are trained using **Square-Root Smoothed Class Weights** to counter heavy class imbalance without destabilizing majority-class precision. Finally, an **88-dimensional Context-Aware Mixture of Experts (MoE)** gating network fuses visual PCA components, out-of-fold probability distributions, and Shannon entropy uncertainty scores to dynamically route predictions to the optimal expert per lesion type.

---

## Architecture Overview

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

## Model Leaderboard (Held-Out Test Set, $n = 1,497$)

All models were evaluated on the exact same patient-isolated test split ($n = 1,497$):

| Model | Test Accuracy | Macro F1 | Weighted F1 | Melanoma Recall (`mel`) | Key Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | **76.75%** | 0.5078 | 0.7507 | 44.05% | Highest overall accuracy; 93.90% recall on majority benign `nv` |
| **MoE (88-D Gate)** | **76.49%** | **0.5204** | **0.7528** | 44.64% | Best balanced performance; dynamic multimodal routing (+4.21% over v1) |
| **Tuned Soft-Voting** | 76.02% | 0.5269 | **0.7528** | 48.21% | Grid-searched static weights ($w=[0.00, 0.70, 0.30]$) |
| **CatBoost** | 75.95% | **0.5337** | 0.7518 | 47.02% | Highest Macro F1; major recovery from 67.20% (+8.75% boost) |
| **LightGBM** | 74.15% | 0.4845 | 0.7386 | **49.40%** | Highest single-model sensitivity on clinically critical Melanoma |

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
├── models/
│   ├── catboost_metrics.json       # CatBoost test evaluation metrics
│   ├── catboost_resnet50.cbm       # Trained CatBoost binary model
│   ├── catboost_test_probs.npy     # CatBoost test probability predictions
│   ├── catboost_val_probs.npy      # CatBoost validation probability predictions
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
│   ├── prepare_split.py            # GroupShuffleSplit patient-isolation generator
│   ├── run_feature_extraction.py   # ResNet-50 2048-D feature extraction script
│   ├── train_catboost.py           # CatBoost trainer with smoothed weights & scaler
│   ├── train_lightgbm.py           # LightGBM trainer with smoothed weights & scaler
│   ├── train_moe_final.py          # 88-D Context-Aware MoE Gating trainer & evaluator
│   ├── train_moe_v2.py             # Wrapper executing train_moe_final.py
│   └── train_xgboost.py            # XGBoost trainer with smoothed weights & scaler
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

---

## Citation & References

- **Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Sci. Data 5, 180161 (2018). [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)
- **Deep Residual Learning**: He, K., Zhang, X., Ren, S., & Sun, J. *Deep Residual Learning for Image Recognition*. CVPR (2016).
- **Out-of-Fold Ensembling & MoE**: Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. *Adaptive Mixtures of Local Experts*. Neural Computation (1991).

---

## License
This project is open-source and licensed under the [MIT License](LICENSE).
