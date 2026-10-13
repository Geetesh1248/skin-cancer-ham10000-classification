"""
train_moe_v2.py
───────────────
Mixture of Experts v2 (MoE v2) for HAM10000 7-class skin lesion classification.

Improvements over v1:
  1. Train set expert probabilities are generated via 5-fold StratifiedGroupKFold
     on lesion_id (out-of-fold / OOF predictions) to prevent the gate from fitting
     to overconfident / overfit in-sample predictions.
  2. Gate operates on concatenated expert probabilities (21 features: 3 experts x 7 classes),
     learning which expert to trust based on the consensus / disagreement patterns.
  3. Gate architecture: Linear(21, 3) + Softmax(dim=-1).
  4. Trained on full train OOF predictions with Adam (lr=1e-3), balanced NLL loss,
     entropy bonus (lambda=0.1) to discourage collapse onto a single expert,
     and early stopping monitored on the validation set.
  5. Evaluates and compares: XGBoost, CatBoost, LightGBM, Simple Average, and MoE v2.
"""

import copy
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from xgboost import XGBClassifier

# Reproducibility
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
OOF_CACHE_PATH = RESULTS_DIR / "moe_v2_oof_probs.npz"

XGB_MODEL_PATH = MODELS_DIR / "xgboost_resnet50.json"
CAT_MODEL_PATH = MODELS_DIR / "catboost_resnet50.cbm"
LGB_MODEL_PATH = MODELS_DIR / "lightgbm_resnet50.txt"

METRICS_OUT    = MODELS_DIR / "moe_v2_metrics.json"
GATE_MODEL_OUT = MODELS_DIR / "moe_v2_gate.pt"

# Best iteration counts identified from early-stopped full runs
XGB_N_ESTIMATORS = 148
CB_ITERATIONS    = 185
LGB_N_ESTIMATORS = 52

# -- 1. Load Data & Label Map --------------------------------------------------
print("=" * 70)
print("Mixture of Experts v2 (MoE v2)  |  HAM10000  |  OOF + 21-d Gate")
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

# Load splits to access lesion_id
splits_df = pd.read_csv(SPLITS_CSV)
train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
val_df   = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
assert len(train_df) == len(X_train), "Mismatch between train_df and X_train"
assert len(val_df)   == len(X_val),   "Mismatch between val_df and X_val"

# -- 2. Load Saved Full Experts for Val and Test -------------------------------
print("\nLoading saved full expert models (for Val and Test sets)...")
xgb_saved = XGBClassifier()
xgb_saved.load_model(str(XGB_MODEL_PATH))

cb_saved = CatBoostClassifier()
cb_saved.load_model(str(CAT_MODEL_PATH))

lgb_saved = lgb.Booster(model_file=str(LGB_MODEL_PATH))

print("Generating predictions for Val and Test sets from saved experts...")
# Val
p_val_xgb = xgb_saved.predict_proba(X_val)
p_val_cb  = cb_saved.predict_proba(X_val)
p_val_lgb = lgb_saved.predict(X_val)
P_val_experts = np.stack([p_val_xgb, p_val_cb, p_val_lgb], axis=1)  # (N_val, 3, 7)
X_gate_val = np.concatenate([p_val_xgb, p_val_cb, p_val_lgb], axis=1)  # (N_val, 21)

# Test
p_test_xgb = xgb_saved.predict_proba(X_test)
p_test_cb  = cb_saved.predict_proba(X_test)
p_test_lgb = lgb_saved.predict(X_test)
P_test_experts = np.stack([p_test_xgb, p_test_cb, p_test_lgb], axis=1)  # (N_test, 3, 7)
X_gate_test = np.concatenate([p_test_xgb, p_test_cb, p_test_lgb], axis=1)  # (N_test, 21)

print(f"  Val gate input shape : {X_gate_val.shape}")
print(f"  Test gate input shape: {X_gate_test.shape}")

# -- 3. Build Out-of-Fold (OOF) Probabilities for Train Set --------------------
if OOF_CACHE_PATH.exists():
    print(f"\n[CACHE] Loading precomputed OOF probabilities from {OOF_CACHE_PATH}...")
    oof_data = np.load(OOF_CACHE_PATH)
    P_oof_xgb = oof_data["P_oof_xgb"]
    P_oof_cb  = oof_data["P_oof_cb"]
    P_oof_lgb = oof_data["P_oof_lgb"]
    print("  Loaded cached OOF successfully.")
else:
    print(f"\nComputing 5-fold StratifiedGroupKFold OOF probabilities on TRAIN set (lesion_id)...")
    print(f"  XGBoost n_estimators : {XGB_N_ESTIMATORS}")
    print(f"  CatBoost iterations  : {CB_ITERATIONS}")
    print(f"  LightGBM n_estimators: {LGB_N_ESTIMATORS}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    P_oof_xgb = np.zeros((len(X_train), num_classes), dtype=np.float32)
    P_oof_cb  = np.zeros((len(X_train), num_classes), dtype=np.float32)
    P_oof_lgb = np.zeros((len(X_train), num_classes), dtype=np.float32)

    t_cv_start = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups=train_df["lesion_id"]), 1):
        t_fold_start = time.time()
        print(f"\n--- Fold {fold}/5 [Train: {len(trn_idx)}, Val: {len(val_idx)}] ---")

        X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        # Verify zero lesion leakage in fold
        trn_lesions = set(train_df.iloc[trn_idx]["lesion_id"])
        val_lesions = set(train_df.iloc[val_idx]["lesion_id"])
        assert len(trn_lesions & val_lesions) == 0, f"Data leakage in fold {fold}!"

        # Balanced weights for this fold
        sw_tr = compute_sample_weight("balanced", y=y_tr)
        cw_tr = [len(y_tr) / (num_classes * (y_tr == c).sum()) for c in range(num_classes)]

        # 1. LightGBM
        t0 = time.time()
        model_lgb = LGBMClassifier(
            n_estimators=LGB_N_ESTIMATORS,
            learning_rate=0.1,
            num_leaves=31,
            objective="multiclass",
            num_class=num_classes,
            random_state=42 + fold,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb.fit(X_tr, y_tr)
        P_oof_lgb[val_idx] = model_lgb.predict_proba(X_va)
        print(f"  LightGBM trained in {time.time()-t0:.1f}s")

        # 2. XGBoost
        t0 = time.time()
        model_xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.1,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
        )
        model_xgb.fit(X_tr, y_tr, sample_weight=sw_tr)
        P_oof_xgb[val_idx] = model_xgb.predict_proba(X_va)
        print(f"  XGBoost trained in  {time.time()-t0:.1f}s")

        # 3. CatBoost
        t0 = time.time()
        model_cb = CatBoostClassifier(
            iterations=CB_ITERATIONS,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42 + fold,
            verbose=0,
            class_weights=cw_tr,
            thread_count=-1,
        )
        model_cb.fit(X_tr, y_tr)
        P_oof_cb[val_idx] = model_cb.predict_proba(X_va)
        print(f"  CatBoost trained in {time.time()-t0:.1f}s")
        print(f"  Fold {fold} finished in {time.time()-t_fold_start:.1f}s")

    print(f"\nAll 5 folds completed in {time.time()-t_cv_start:.1f}s total.")
    np.savez_compressed(
        OOF_CACHE_PATH,
        P_oof_xgb=P_oof_xgb,
        P_oof_cb=P_oof_cb,
        P_oof_lgb=P_oof_lgb,
    )
    print(f"Saved OOF probabilities to {OOF_CACHE_PATH}")

# Construct gate training inputs (21 features: 3 experts x 7 classes)
X_gate_train = np.concatenate([P_oof_xgb, P_oof_cb, P_oof_lgb], axis=1)  # (7054, 21)
P_train_experts = np.stack([P_oof_xgb, P_oof_cb, P_oof_lgb], axis=1)     # (7054, 3, 7)
print(f"\nGate Train Input shape: {X_gate_train.shape}")

# -- 4. Gate Model Architecture & Training -------------------------------------
class GateV2(nn.Module):
    """
    Linear(21, 3) + Softmax gate network.
    Takes 21 concatenated expert class probabilities and predicts sample-level
    weight distribution over the 3 experts.
    """
    def __init__(self, in_features=21, num_experts=3):
        super().__init__()
        self.linear = nn.Linear(in_features, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

gate = GateV2(in_features=21, num_experts=3)

# Balanced class weights for NLL
cw_full = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
class_weights_t = torch.tensor(cw_full, dtype=torch.float32)
nll_criterion = nn.NLLLoss(weight=class_weights_t)

optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)

# Prepare PyTorch Tensors
X_gtrain_t  = torch.tensor(X_gate_train, dtype=torch.float32)
P_gtrain_t  = torch.tensor(P_train_experts, dtype=torch.float32)
y_gtrain_t  = torch.tensor(y_train, dtype=torch.long)

X_gval_t    = torch.tensor(X_gate_val, dtype=torch.float32)
P_gval_t    = torch.tensor(P_val_experts, dtype=torch.float32)
y_gval_t    = torch.tensor(y_val, dtype=torch.long)

print("\nTraining Gate on OOF Train predictions (Adam lr=1e-3, lambda_entropy=0.1, val early stopping)...")
print("-" * 70)

epochs = 300
patience = 35
best_val_loss = float("inf")
best_epoch = 0
best_state = None
patience_counter = 0
lambda_entropy = 0.1

for epoch in range(1, epochs + 1):
    gate.train()
    optimizer.zero_grad()

    # Forward on Train OOF
    w_train = gate(X_gtrain_t)  # (N_train, 3)
    P_mixture_train = torch.sum(w_train.unsqueeze(-1) * P_gtrain_t, dim=1)  # (N_train, 7)
    log_p_train = torch.log(torch.clamp(P_mixture_train, min=1e-12, max=1.0))
    nll_train = nll_criterion(log_p_train, y_gtrain_t)

    # Entropy bonus: H(w) = - sum(w * log(w))
    entropy_train = - torch.sum(w_train * torch.log(torch.clamp(w_train, min=1e-12, max=1.0)), dim=-1)
    mean_entropy_train = torch.mean(entropy_train)

    # Total loss = NLL - lambda * Entropy (encourages diverse weights)
    loss_train = nll_train - lambda_entropy * mean_entropy_train

    loss_train.backward()
    optimizer.step()

    # Evaluate on Validation set
    gate.eval()
    with torch.no_grad():
        w_val = gate(X_gval_t)
        P_mixture_val = torch.sum(w_val.unsqueeze(-1) * P_gval_t, dim=1)
        log_p_val = torch.log(torch.clamp(P_mixture_val, min=1e-12, max=1.0))
        nll_val = nll_criterion(log_p_val, y_gval_t)
        entropy_val = - torch.sum(w_val * torch.log(torch.clamp(w_val, min=1e-12, max=1.0)), dim=-1)
        mean_entropy_val = torch.mean(entropy_val)
        val_loss = (nll_val - lambda_entropy * mean_entropy_val).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(gate.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 25 == 0 or epoch == 1 or patience_counter == patience:
        print(f"  Epoch {epoch:3d}/{epochs:3d} | Train Loss: {loss_train.item():.4f} (NLL: {nll_train.item():.4f}, Ent: {mean_entropy_train.item():.4f}) | "
              f"Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f} (ep {best_epoch})")

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch} (best epoch: {best_epoch}, best val loss: {best_val_loss:.4f})")
        break

# Restore best weights
gate.load_state_dict(best_state)
gate.eval()

# Check learned weights on val set
with torch.no_grad():
    val_weights = gate(X_gval_t).numpy()
mean_val_weights = val_weights.mean(axis=0)
print(f"\nLearned average gate weights across Val set:")
print(f"  XGBoost:  {mean_val_weights[0]:.4f}")
print(f"  CatBoost: {mean_val_weights[1]:.4f}")
print(f"  LightGBM: {mean_val_weights[2]:.4f}")

# -- 5. Evaluation on Held-Out Test Set ----------------------------------------
print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

# (A) Individual experts
y_pred_xgb = np.argmax(P_test_experts[:, 0, :], axis=1)
y_pred_cb  = np.argmax(P_test_experts[:, 1, :], axis=1)
y_pred_lgb = np.argmax(P_test_experts[:, 2, :], axis=1)

# (B) Simple average baseline: (P_xgb + P_cb + P_lgb) / 3
P_baseline_test = np.mean(P_test_experts, axis=1)
y_pred_baseline = np.argmax(P_baseline_test, axis=1)

# (C) MoE v2 (Gated)
with torch.no_grad():
    gate_test_weights = gate(torch.tensor(X_gate_test, dtype=torch.float32)).numpy()  # (N_test, 3)
P_moe_v2_test = np.sum(gate_test_weights[:, :, None] * P_test_experts, axis=1)     # (N_test, 7)
y_pred_moe_v2 = np.argmax(P_moe_v2_test, axis=1)

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

res_xgb      = compute_metrics(y_test, y_pred_xgb, "XGBoost")
res_cb       = compute_metrics(y_test, y_pred_cb, "CatBoost")
res_lgb      = compute_metrics(y_test, y_pred_lgb, "LightGBM")
res_baseline = compute_metrics(y_test, y_pred_baseline, "Simple Average")
res_moe_v2   = compute_metrics(y_test, y_pred_moe_v2, "MoE v2 (21-d Gate)")

all_results = [res_xgb, res_cb, res_lgb, res_baseline, res_moe_v2]

# Summary Comparison Table
print(f"\n{'Model':22s} | {'Accuracy':8s} | {'Macro F1':8s} | {'Weighted F1':11s} | {'Mel Recall':10s}")
print("-" * 69)
for r in all_results:
    print(f"{r['name']:22s} | {r['accuracy']:8.4f} | {r['macro_f1']:8.4f} | {r['weighted_f1']:11.4f} | {r['mel_recall']:10.4f}")

# Mean gate weight per expert on test
mean_test_weights = gate_test_weights.mean(axis=0)
print("\nMean Gate Weights across Test Set:")
print(f"  XGBoost:  {mean_test_weights[0]:.4f}")
print(f"  CatBoost: {mean_test_weights[1]:.4f}")
print(f"  LightGBM: {mean_test_weights[2]:.4f}")

# Mean gate weights per ground-truth class on test
print("\nMean Expert Gating Weight by Ground-Truth Class (Test Set):")
print(f"{'Class':8s} | {'XGBoost':8s} | {'CatBoost':8s} | {'LightGBM':8s}")
print("-" * 40)
for i, cls in enumerate(class_names):
    mask = (y_test == i)
    mean_w = gate_test_weights[mask].mean(axis=0)
    print(f"{cls:8s} | {mean_w[0]:8.4f} | {mean_w[1]:8.4f} | {mean_w[2]:8.4f}")

# MoE v2 Detailed Classification Report
print("\n" + "=" * 70)
print("MoE v2 Detailed Classification Report (Test Set):")
print("=" * 70)
print(classification_report(y_test, y_pred_moe_v2, target_names=class_names))

print("MoE v2 Per-Class Recall (Sensitivity):")
for cls in class_names:
    rec  = res_moe_v2["report"][cls]["recall"]
    prec = res_moe_v2["report"][cls]["precision"]
    f1   = res_moe_v2["report"][cls]["f1-score"]
    supp = int(res_moe_v2["report"][cls]["support"])
    print(f"  {cls:6s}  recall={rec:.4f}  precision={prec:.4f}  f1={f1:.4f}  support={supp:4d}")

print("\nMoE v2 Confusion Matrix (rows=true, cols=pred):")
header = f"{'':8s}" + "".join(f"{c:>7s}" for c in class_names)
print(header)
cm_v2 = res_moe_v2["cm"]
for i, cls in enumerate(class_names):
    row = f"{cls:8s}" + "".join(f"{cm_v2[i,j]:>7d}" for j in range(num_classes))
    print(row)

# -- 6. Save Metrics & Model ---------------------------------------------------
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
    "mean_gate_weights": {
        "xgboost":  round(float(mean_test_weights[0]), 4),
        "catboost": round(float(mean_test_weights[1]), 4),
        "lightgbm": round(float(mean_test_weights[2]), 4),
        "by_class": {
            cls: [round(float(x), 4) for x in gate_test_weights[y_test == i].mean(axis=0)]
            for i, cls in enumerate(class_names)
        }
    },
    "moe_v2_detailed": {
        "accuracy": res_moe_v2["accuracy"],
        "macro_f1": res_moe_v2["macro_f1"],
        "weighted_f1": res_moe_v2["weighted_f1"],
        "mel_recall": res_moe_v2["mel_recall"],
        "per_class": {
            cls: {
                "precision": round(res_moe_v2["report"][cls]["precision"], 4),
                "recall":    round(res_moe_v2["report"][cls]["recall"], 4),
                "f1":        round(res_moe_v2["report"][cls]["f1-score"], 4),
                "support":   int(res_moe_v2["report"][cls]["support"]),
            }
            for cls in class_names
        },
        "confusion_matrix": cm_v2.tolist(),
    }
}

with open(METRICS_OUT, "w") as f:
    json.dump(metrics_export, f, indent=2)
print(f"\nSaved metrics: {METRICS_OUT}")

torch.save(gate.state_dict(), str(GATE_MODEL_OUT))
print(f"Saved gate model weights: {GATE_MODEL_OUT}")
print("\n[DONE] MoE v2 execution complete.")
