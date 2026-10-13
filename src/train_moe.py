"""
train_moe.py
────────────
Mixture of Experts (MoE) for HAM10000 7-class skin lesion classification.

Experts:
  1. XGBoost  (ResNet50 features)
  2. CatBoost (ResNet50 features)
  3. LightGBM (ResNet50 features)

Gate:
  - Input: 2048-d ResNet50 feature vector
  - Architecture: Linear(2048, 3) + Softmax(dim=-1)
  - Trained ONLY on validation set (split 80/20 by lesion_id via GroupShuffleSplit)
  - Loss: NLL with balanced class weights
  - Optimization: Adam (lr=1e-3) with early stopping on gate-holdout loss

Evaluation:
  - Compares all 3 base experts, simple average baseline, and MoE on test set.
"""

import copy
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -- Paths ---------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH       = RESULTS_DIR / "resnet50_features.npz"
LABEL_MAP_PATH = RESULTS_DIR / "label_mapping.json"
SPLITS_CSV     = RESULTS_DIR / "splits.csv"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_resnet50.json"
CAT_MODEL_PATH = MODELS_DIR / "catboost_resnet50.cbm"
LGB_MODEL_PATH = MODELS_DIR / "lightgbm_resnet50.txt"
METRICS_OUT    = MODELS_DIR / "moe_metrics.json"

# -- 1. Load Data --------------------------------------------------------------
print("=" * 70)
print("Mixture of Experts (MoE)  |  HAM10000  |  7-Class Classification")
print("=" * 70)

data = np.load(NPZ_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]
print(f"Data shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
idx2cls = {v: k for k, v in label_map.items()}
class_names = [idx2cls[i] for i in range(len(idx2cls))]
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# -- 2. Load Trained Expert Models ---------------------------------------------
print("\nLoading expert models...")
print(f"  [1] XGBoost:  {XGB_MODEL_PATH}")
xgb_model = XGBClassifier()
xgb_model.load_model(str(XGB_MODEL_PATH))

print(f"  [2] CatBoost: {CAT_MODEL_PATH}")
cb_model = CatBoostClassifier()
cb_model.load_model(str(CAT_MODEL_PATH))

print(f"  [3] LightGBM: {LGB_MODEL_PATH}")
lgb_booster = lgb.Booster(model_file=str(LGB_MODEL_PATH))

# -- 3. Get Predict Probabilities from Experts (Frozen) ------------------------
print("\nExtracting probability distributions from experts...")

def get_expert_probs(X):
    p_xgb = xgb_model.predict_proba(X)
    p_cb  = cb_model.predict_proba(X)
    p_lgb = lgb_booster.predict(X)
    # Stack along axis 1 -> shape: (N, 3, 7)
    return np.stack([p_xgb, p_cb, p_lgb], axis=1)

P_val_experts  = get_expert_probs(X_val)
P_test_experts = get_expert_probs(X_test)
print(f"  Val expert probs shape : {P_val_experts.shape}")
print(f"  Test expert probs shape: {P_test_experts.shape}")

# -- 4. Gate Training Split (Val Set 80/20 Group-Aware on lesion_id) -----------
splits_df = pd.read_csv(SPLITS_CSV)
val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
assert len(val_df) == len(X_val), f"Mismatch between splits.csv val rows ({len(val_df)}) and X_val ({len(X_val)})"

gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
gate_train_idx, gate_holdout_idx = next(gss.split(val_df, groups=val_df["lesion_id"]))

train_lesions   = set(val_df.iloc[gate_train_idx]["lesion_id"])
holdout_lesions = set(val_df.iloc[gate_holdout_idx]["lesion_id"])
overlap = train_lesions & holdout_lesions
assert len(overlap) == 0, f"Data leakage in gate split: {len(overlap)} shared lesions!"

print("\nGate dataset split (80/20 GroupShuffleSplit on val set):")
print(f"  Gate Train   : {len(gate_train_idx):5d} samples, {len(train_lesions):5d} unique lesions")
print(f"  Gate Holdout : {len(gate_holdout_idx):5d} samples, {len(holdout_lesions):5d} unique lesions")
print(f"  Lesion overlap: 0 (verified zero leakage)")

# Slice gate train and holdout
X_gtrain   = torch.tensor(X_val[gate_train_idx], dtype=torch.float32)
P_gtrain   = torch.tensor(P_val_experts[gate_train_idx], dtype=torch.float32)
y_gtrain   = torch.tensor(y_val[gate_train_idx], dtype=torch.long)

X_gholdout = torch.tensor(X_val[gate_holdout_idx], dtype=torch.float32)
P_gholdout = torch.tensor(P_val_experts[gate_holdout_idx], dtype=torch.float32)
y_gholdout = torch.tensor(y_val[gate_holdout_idx], dtype=torch.long)

# -- 5. Gate Model Architecture & Loss -----------------------------------------
class Gate(nn.Module):
    def __init__(self, in_features=2048, num_experts=3):
        super().__init__()
        self.linear = nn.Linear(in_features, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

gate = Gate(in_features=2048, num_experts=3)

# Balanced class weights computed on training set
cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
class_weights_t = torch.tensor(cw, dtype=torch.float32)
nll_criterion = nn.NLLLoss(weight=class_weights_t)

optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)

# -- 6. Train Gate with Early Stopping -----------------------------------------
print("\nTraining Gate on validation set (Adam lr=1e-3, early stopping on gate-holdout loss)...")
print("-" * 70)

epochs = 200
patience = 25
best_holdout_loss = float("inf")
best_epoch = 0
best_state = None
patience_counter = 0

for epoch in range(1, epochs + 1):
    gate.train()
    optimizer.zero_grad()

    # Forward on gate-train
    w_train = gate(X_gtrain)  # (N, 3)
    # Mixture probabilities: sum_k w_k * P_k -> (N, 7)
    P_mixture_train = torch.sum(w_train.unsqueeze(-1) * P_gtrain, dim=1)
    log_p_train = torch.log(torch.clamp(P_mixture_train, min=1e-12, max=1.0))
    loss_train = nll_criterion(log_p_train, y_gtrain)

    loss_train.backward()
    optimizer.step()

    # Evaluate on gate-holdout
    gate.eval()
    with torch.no_grad():
        w_holdout = gate(X_gholdout)
        P_mixture_holdout = torch.sum(w_holdout.unsqueeze(-1) * P_gholdout, dim=1)
        log_p_holdout = torch.log(torch.clamp(P_mixture_holdout, min=1e-12, max=1.0))
        loss_holdout = nll_criterion(log_p_holdout, y_gholdout).item()

    if loss_holdout < best_holdout_loss:
        best_holdout_loss = loss_holdout
        best_epoch = epoch
        best_state = copy.deepcopy(gate.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 20 == 0 or epoch == 1 or patience_counter == patience:
        print(f"  Epoch {epoch:3d}/{epochs:3d} | Train Loss: {loss_train.item():.4f} | "
              f"Holdout Loss: {loss_holdout:.4f} | Best: {best_holdout_loss:.4f} (ep {best_epoch})")

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch} (best epoch: {best_epoch}, best loss: {best_holdout_loss:.4f})")
        break

# Restore best weights
gate.load_state_dict(best_state)
gate.eval()

# Check average gate weights on validation set
with torch.no_grad():
    all_val_weights = gate(torch.tensor(X_val, dtype=torch.float32)).numpy()
mean_weights = all_val_weights.mean(axis=0)
print(f"\nLearned average gate weights across val set:")
print(f"  XGBoost:  {mean_weights[0]:.4f}")
print(f"  CatBoost: {mean_weights[1]:.4f}")
print(f"  LightGBM: {mean_weights[2]:.4f}")

# -- 7. Predictions on Test Set ------------------------------------------------
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

# (A) Individual experts
y_pred_xgb = np.argmax(P_test_experts[:, 0, :], axis=1)
y_pred_cb  = np.argmax(P_test_experts[:, 1, :], axis=1)
y_pred_lgb = np.argmax(P_test_experts[:, 2, :], axis=1)

# (B) Simple average baseline: (P_xgb + P_cb + P_lgb) / 3
P_baseline_test = np.mean(P_test_experts, axis=1)  # (N, 7)
y_pred_baseline = np.argmax(P_baseline_test, axis=1)

# (C) Mixture of Experts
with torch.no_grad():
    gate_test_weights = gate(torch.tensor(X_test, dtype=torch.float32)).numpy()  # (N, 3)
P_moe_test = np.sum(gate_test_weights[:, :, None] * P_test_experts, axis=1)    # (N, 7)
y_pred_moe = np.argmax(P_moe_test, axis=1)

# Helper function to compute metrics
def compute_metrics(y_true, y_pred, name):
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    rep         = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm          = confusion_matrix(y_true, y_pred)
    mel_recall  = rep["mel"]["recall"]
    return {
        "name": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "mel_recall": round(mel_recall, 4),
        "report": rep,
        "cm": cm,
    }

results_xgb      = compute_metrics(y_test, y_pred_xgb, "XGBoost")
results_cb       = compute_metrics(y_test, y_pred_cb, "CatBoost")
results_lgb      = compute_metrics(y_test, y_pred_lgb, "LightGBM")
results_baseline = compute_metrics(y_test, y_pred_baseline, "Simple Average")
results_moe      = compute_metrics(y_test, y_pred_moe, "MoE (Gated)")

all_results = [results_xgb, results_cb, results_lgb, results_baseline, results_moe]

# Summary Comparison Table
print(f"\n{'Model':18s} | {'Accuracy':8s} | {'Macro F1':8s} | {'Weighted F1':11s} | {'Mel Recall':10s}")
print("-" * 65)
for r in all_results:
    print(f"{r['name']:18s} | {r['accuracy']:8.4f} | {r['macro_f1']:8.4f} | {r['weighted_f1']:11.4f} | {r['mel_recall']:10.4f}")

# Detailed MoE Breakdown
print("\n" + "=" * 70)
print("MoE Detailed Classification Report (Test Set):")
print("=" * 70)
print(classification_report(y_test, y_pred_moe, target_names=class_names))

print("MoE Per-Class Recall (Sensitivity):")
for cls in class_names:
    rec  = results_moe["report"][cls]["recall"]
    prec = results_moe["report"][cls]["precision"]
    f1   = results_moe["report"][cls]["f1-score"]
    supp = int(results_moe["report"][cls]["support"])
    print(f"  {cls:6s}  recall={rec:.4f}  precision={prec:.4f}  f1={f1:.4f}  support={supp:4d}")

print("\nMoE Confusion Matrix (rows=true, cols=pred):")
header = f"{'':8s}" + "".join(f"{c:>7s}" for c in class_names)
print(header)
cm_moe = results_moe["cm"]
for i, cls in enumerate(class_names):
    row = f"{cls:8s}" + "".join(f"{cm_moe[i,j]:>7d}" for j in range(num_classes))
    print(row)

# Mean gating weight assigned per ground-truth class
print("\nMean Expert Gating Weight by Ground-Truth Class (Test Set):")
print(f"{'Class':8s} | {'XGBoost':8s} | {'CatBoost':8s} | {'LightGBM':8s}")
print("-" * 40)
for i, cls in enumerate(class_names):
    mask = (y_test == i)
    mean_w = gate_test_weights[mask].mean(axis=0)
    print(f"{cls:8s} | {mean_w[0]:8.4f} | {mean_w[1]:8.4f} | {mean_w[2]:8.4f}")

# -- 8. Save Metrics to models/moe_metrics.json --------------------------------
metrics_export = {
    "comparison": [
        {
            "model": r["name"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "mel_recall": r["mel_recall"],
        }
        for r in all_results
    ],
    "moe_detailed": {
        "accuracy": results_moe["accuracy"],
        "macro_f1": results_moe["macro_f1"],
        "weighted_f1": results_moe["weighted_f1"],
        "mel_recall": results_moe["mel_recall"],
        "per_class": {
            cls: {
                "precision": round(results_moe["report"][cls]["precision"], 4),
                "recall":    round(results_moe["report"][cls]["recall"], 4),
                "f1":        round(results_moe["report"][cls]["f1-score"], 4),
                "support":   int(results_moe["report"][cls]["support"]),
            }
            for cls in class_names
        },
        "confusion_matrix": cm_moe.tolist(),
        "mean_gate_weights": {
            "all_test": [round(float(x), 4) for x in gate_test_weights.mean(axis=0)],
            "by_class": {
                cls: [round(float(x), 4) for x in gate_test_weights[y_test == i].mean(axis=0)]
                for i, cls in enumerate(class_names)
            }
        }
    }
}

with open(METRICS_OUT, "w") as f:
    json.dump(metrics_export, f, indent=2)
print(f"\nSaved metrics: {METRICS_OUT}")

# Also save the gate model weights
GATE_MODEL_PATH = MODELS_DIR / "moe_gate.pt"
torch.save(gate.state_dict(), str(GATE_MODEL_PATH))
print(f"Saved gate model weights: {GATE_MODEL_PATH}")
print("\n[DONE] MoE training and evaluation complete.")
