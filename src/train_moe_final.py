"""
train_moe_final.py
──────────────────
Mixture of Experts (MoE) Final Architecture for HAM10000 7-Class Classification.

Pipeline:
1. Generate 5-fold cross-validated out-of-fold probability predictions
   (oof_probs, shape: [N_train, 3, 7]) on X_train using the exact smoothed-weight
   configurations for XGBoost, CatBoost, and LightGBM.
   Load saved validation and test probability predictions (val_probs, test_probs).
2. Fit PCA(n_components=64, random_state=42) on scaled X_train to supply compact visual context.
3. MoEGatingNetwork taking [64-D PCA + 21-D expert probabilities + 3-D expert entropy scores] (total dim = 88):
   nn.Linear(88, 32) -> nn.BatchNorm1d(32) -> nn.ReLU() -> nn.Dropout(0.3) -> nn.Linear(32, 3) -> Softmax(dim=-1).
   Final linear layer bias initialized with [0.5, 0.0, 0.5] favoring XGBoost & LightGBM over CatBoost.
4. Train gate with CrossEntropyLoss on log of weighted probability sum (torch.log(weighted_probs + 1e-8)),
   using Adam(lr=1e-3, weight_decay=1e-3) for up to 100 epochs, early stopping (patience=15) on val (Macro F1 + Accuracy).
5. Tuned Soft-Voting Ensemble (grid search optimal static weights on validation set).
6. Export final metrics to models/moe_metrics.json, print complete leaderboard and gate weights,
   and commit + push all updated scripts and metrics to GitHub.
"""

import copy
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH       = RESULTS_DIR / "resnet50_features.npz"
LABEL_MAP_PATH = RESULTS_DIR / "label_mapping.json"
SPLITS_CSV     = RESULTS_DIR / "splits.csv"
SCALER_PATH    = MODELS_DIR  / "scaler.joblib"

OOF_CACHE_PATH = RESULTS_DIR / "moe_oof_probs.npz"
OOF_V2_CACHE   = RESULTS_DIR / "moe_v2_oof_probs.npz"
PROGRESS_PATH  = RESULTS_DIR / "moe_oof_progress.npz"

VAL_XGB = MODELS_DIR / "xgboost_val_probs.npy"
VAL_CB  = MODELS_DIR / "catboost_val_probs.npy"
VAL_LGB = MODELS_DIR / "lightgbm_val_probs.npy"

TEST_XGB = MODELS_DIR / "xgboost_test_probs.npy"
TEST_CB  = MODELS_DIR / "catboost_test_probs.npy"
TEST_LGB = MODELS_DIR / "lightgbm_test_probs.npy"

METRICS_OUT    = MODELS_DIR / "moe_metrics.json"
METRICS_V2_OUT = MODELS_DIR / "moe_v2_metrics.json"
GATE_MODEL_OUT = MODELS_DIR / "moe_gate.pt"
GATE_V2_OUT    = MODELS_DIR / "moe_v2_gate.pt"
PCA_MODEL_OUT  = MODELS_DIR / "moe_pca.joblib"

# ── 1. Load Data & Label Map ──────────────────────────────────────────────────
print("=" * 78)
print("Mixture of Experts (MoE)  |  HAM10000  |  88-D Multimodal Gating Network")
print("=" * 78)

data = np.load(NPZ_PATH)
X_train_raw, y_train = data["X_train"], data["y_train"]
X_val_raw,   y_val   = data["X_val"],   data["y_val"]
X_test_raw,  y_test  = data["X_test"],  data["y_test"]
print(f"Data shapes -> Train: {X_train_raw.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}")

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
idx2cls = {v: k for k, v in label_map.items()}
class_names = [idx2cls[i] for i in range(len(idx2cls))]
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# Apply StandardScaler fit ONLY on X_train
if SCALER_PATH.exists():
    scaler = joblib.load(SCALER_PATH)
    print(f"Loaded existing StandardScaler from {SCALER_PATH}")
else:
    scaler = StandardScaler().fit(X_train_raw)
    joblib.dump(scaler, SCALER_PATH)

X_train_scaled = scaler.transform(X_train_raw)
X_val_scaled   = scaler.transform(X_val_raw)
X_test_scaled  = scaler.transform(X_test_raw)

# Load splits to access lesion_id
splits_df = pd.read_csv(SPLITS_CSV)
train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
val_df   = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
assert len(train_df) == len(X_train_scaled), "Mismatch between train_df and X_train"
assert len(val_df)   == len(X_val_scaled),   "Mismatch between val_df and X_val"

# Load saved full-expert validation and test predictions
print("\nLoading saved expert validation and test probabilities...")
p_val_xgb = np.load(VAL_XGB)
p_val_cb  = np.load(VAL_CB)
p_val_lgb = np.load(VAL_LGB)
val_probs = np.stack([p_val_xgb, p_val_cb, p_val_lgb], axis=1)  # (N_val, 3, 7)

p_test_xgb = np.load(TEST_XGB)
p_test_cb  = np.load(TEST_CB)
p_test_lgb = np.load(TEST_LGB)
test_probs = np.stack([p_test_xgb, p_test_cb, p_test_lgb], axis=1)  # (N_test, 3, 7)

print(f"  val_probs shape : {val_probs.shape} (experts: 0=XGB, 1=CB, 2=LGB)")
print(f"  test_probs shape: {test_probs.shape}")

# ── 2. Generate 5-Fold OOF Predictions on Train Set ────────────────────────────
if OOF_CACHE_PATH.exists():
    print(f"\n[CACHE] Loading precomputed OOF probabilities from {OOF_CACHE_PATH}...")
    oof_data = np.load(OOF_CACHE_PATH)
    oof_probs = oof_data["oof_probs"]
    P_oof_xgb = oof_data["P_oof_xgb"]
    P_oof_cb  = oof_data["P_oof_cb"]
    P_oof_lgb = oof_data["P_oof_lgb"]
    print("  Loaded cached OOF successfully.")
else:
    print(f"\nComputing 5-fold StratifiedGroupKFold OOF probabilities on TRAIN set (lesion_id)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    P_oof_xgb = np.zeros((len(X_train_scaled), num_classes), dtype=np.float32)
    P_oof_cb  = np.zeros((len(X_train_scaled), num_classes), dtype=np.float32)
    P_oof_lgb = np.zeros((len(X_train_scaled), num_classes), dtype=np.float32)
    completed_folds = set()

    if PROGRESS_PATH.exists():
        prog = np.load(PROGRESS_PATH)
        completed_folds = set(prog["completed_folds"].tolist())
        P_oof_xgb = prog["P_oof_xgb"]
        P_oof_cb  = prog["P_oof_cb"]
        P_oof_lgb = prog["P_oof_lgb"]
        print(f"  Resuming from progress checkpoint with completed folds: {completed_folds}")

    t_cv_start = time.time()
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_train_scaled, y_train, groups=train_df["lesion_id"]), 1):
        if fold in completed_folds:
            print(f"  Fold {fold}/5 already completed, skipping.")
            continue

        t_fold_start = time.time()
        print(f"\n--- Fold {fold}/5 [Train: {len(trn_idx)}, Val: {len(val_idx)}] ---")

        X_tr, y_tr = X_train_scaled[trn_idx], y_train[trn_idx]
        X_va, y_va = X_train_scaled[val_idx], y_train[val_idx]

        # Verify group isolation
        trn_lesions = set(train_df.iloc[trn_idx]["lesion_id"])
        val_lesions = set(train_df.iloc[val_idx]["lesion_id"])
        assert len(trn_lesions & val_lesions) == 0, f"Lesion leakage in fold {fold}!"

        # Square-Root Smoothed Class Weights for this fold
        total_tr = len(y_tr)
        cw_tr = {c: float((total_tr / (num_classes * np.sum(y_tr == c))) ** 0.5) for c in range(num_classes)}
        sw_tr = np.array([cw_tr[y] for y in y_tr], dtype=np.float32)
        cw_list_tr = [cw_tr[c] for c in range(num_classes)]

        # 1. XGBoost
        t0 = time.time()
        model_xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            n_estimators=220,
            max_depth=6,
            learning_rate=0.05,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=30,
        )
        model_xgb.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False)
        P_oof_xgb[val_idx] = model_xgb.predict_proba(X_va)
        print(f"  [1/3] XGBoost trained in {time.time()-t0:.1f}s (best iter: {getattr(model_xgb, 'best_iteration', 'N/A')})")

        # 2. LightGBM
        t0 = time.time()
        model_lgb = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            objective="multiclass",
            num_class=num_classes,
            random_state=42 + fold,
            class_weight=cw_tr,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=30, verbose=False)],
        )
        P_oof_lgb[val_idx] = model_lgb.predict_proba(X_va)
        print(f"  [2/3] LightGBM trained in {time.time()-t0:.1f}s (best iter: {getattr(model_lgb, 'best_iteration_', 'N/A')})")

        # 3. CatBoost
        t0 = time.time()
        model_cb = CatBoostClassifier(
            iterations=220,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42 + fold,
            class_weights=cw_list_tr,
            early_stopping_rounds=30,
            border_count=64,
            thread_count=-1,
            verbose=0,
        )
        model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        P_oof_cb[val_idx] = model_cb.predict_proba(X_va)
        print(f"  [3/3] CatBoost trained in {time.time()-t0:.1f}s (best iter: {model_cb.get_best_iteration()})")
        print(f"  Fold {fold} finished in {time.time()-t_fold_start:.1f}s")

        completed_folds.add(fold)
        np.savez_compressed(
            PROGRESS_PATH,
            completed_folds=np.array(list(completed_folds)),
            P_oof_xgb=P_oof_xgb,
            P_oof_cb=P_oof_cb,
            P_oof_lgb=P_oof_lgb,
        )

    print(f"\nAll 5 folds completed in {time.time()-t_cv_start:.1f}s total.")
    oof_probs = np.stack([P_oof_xgb, P_oof_cb, P_oof_lgb], axis=1)  # (N_train, 3, 7)

    np.savez_compressed(
        OOF_CACHE_PATH,
        oof_probs=oof_probs,
        P_oof_xgb=P_oof_xgb,
        P_oof_cb=P_oof_cb,
        P_oof_lgb=P_oof_lgb,
    )
    np.savez_compressed(
        OOF_V2_CACHE,
        P_oof_xgb=P_oof_xgb,
        P_oof_cb=P_oof_cb,
        P_oof_lgb=P_oof_lgb,
    )
    print(f"Saved OOF probabilities to {OOF_CACHE_PATH}")

print(f"OOF probabilities ready. Shape: {oof_probs.shape}")

# ── 3. PCA Feature Reduction (64 Components) ──────────────────────────────────
print("\nFitting PCA(n_components=64, random_state=42) on scaled X_train...")
pca = PCA(n_components=64, random_state=42)
pca.fit(X_train_scaled)
pca_train = pca.transform(X_train_scaled)
pca_val   = pca.transform(X_val_scaled)
pca_test  = pca.transform(X_test_scaled)
joblib.dump(pca, PCA_MODEL_OUT)
print(f"  PCA explained variance ratio sum (64 comps): {pca.explained_variance_ratio_.sum():.4f}")
print(f"  PCA features shape -> Train: {pca_train.shape}, Val: {pca_val.shape}, Test: {pca_test.shape}")

# ── 4. Build 88-D Gate Features ───────────────────────────────────────────────
def compute_entropy(probs):
    """Computes Shannon entropy across classes for each expert. probs: (N, 3, 7) -> (N, 3)"""
    p_clip = np.clip(probs, 1e-8, 1.0)
    return (-np.sum(p_clip * np.log(p_clip), axis=2)).astype(np.float32)

def build_gate_inputs(pca_feats, expert_probs):
    """
    Concatenates:
      - 64-D PCA features
      - 21-D expert probabilities (3 experts x 7 classes)
      - 3-D expert entropy scores
    Total = 88-D.
    """
    N = expert_probs.shape[0]
    probs_flat = expert_probs.reshape(N, 21).astype(np.float32)
    entropy    = compute_entropy(expert_probs)  # (N, 3)
    x_gate     = np.concatenate([pca_feats, probs_flat, entropy], axis=1).astype(np.float32)
    return x_gate

x_gate_train = build_gate_inputs(pca_train, oof_probs)
x_gate_val   = build_gate_inputs(pca_val, val_probs)
x_gate_test  = build_gate_inputs(pca_test, test_probs)

print(f"\nGating network input shapes:")
print(f"  x_gate_train: {x_gate_train.shape} (64 PCA + 21 probs + 3 entropy = 88 features)")
print(f"  x_gate_val  : {x_gate_val.shape}")
print(f"  x_gate_test : {x_gate_test.shape}")

# ── 5. PyTorch MoEGatingNetwork Architecture ───────────────────────────────────
class MoEGatingNetwork(nn.Module):
    """
    Architecture:
      nn.Linear(88, 32) -> nn.BatchNorm1d(32) -> nn.ReLU() -> nn.Dropout(0.3)
      -> nn.Linear(32, 3) -> Softmax(dim=-1).
    Bias initialization:
      Final linear layer initialized with [0.5, 0.0, 0.5] favoring XGBoost & LightGBM over CatBoost.
    """
    def __init__(self, in_features=88, hidden_dim=32, num_experts=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize final layer bias with [0.5, 0.0, 0.5]
        with torch.no_grad():
            self.fc2.bias.copy_(torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32))
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return self.softmax(logits)

# ── 6. Train Gating Network ───────────────────────────────────────────────────
print("\n" + "=" * 78)
print("TRAINING MOE GATING NETWORK")
print("=" * 78)

gate = MoEGatingNetwork(in_features=88, hidden_dim=32, num_experts=3)
optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

train_dataset = TensorDataset(
    torch.tensor(x_gate_train, dtype=torch.float32),
    torch.tensor(oof_probs, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

x_val_tensor   = torch.tensor(x_gate_val, dtype=torch.float32)
val_exp_tensor = torch.tensor(val_probs, dtype=torch.float32)

best_val_score   = -1.0
best_epoch       = 0
best_state       = None
patience_counter = 0
max_epochs       = 100
patience         = 15

print(f"Training for up to {max_epochs} epochs with patience={patience} on (Val Acc + Val Macro F1)...")
for epoch in range(1, max_epochs + 1):
    gate.train()
    total_loss = 0.0
    for bx, bp, by in train_loader:
        optimizer.zero_grad()
        bw = gate(bx)  # (B, 3)
        b_weighted = torch.sum(bw.unsqueeze(-1) * bp, dim=1)  # (B, 7)
        b_log = torch.log(b_weighted + 1e-8)
        loss = criterion(b_log, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(by)

    train_loss = total_loss / len(y_train)

    # Validation evaluation
    gate.eval()
    with torch.no_grad():
        vw = gate(x_val_tensor)  # (N_val, 3)
        v_weighted = torch.sum(vw.unsqueeze(-1) * val_exp_tensor, dim=1).cpu().numpy()
        v_pred = np.argmax(v_weighted, axis=1)
        val_acc = accuracy_score(y_val, v_pred)
        val_macro = f1_score(y_val, v_pred, average="macro")
        val_score = val_acc + val_macro

    is_best = val_score > best_val_score
    if is_best:
        best_val_score = val_score
        best_epoch = epoch
        best_state = copy.deepcopy(gate.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 5 == 0 or epoch == 1 or is_best:
        marker = "  <-- BEST" if is_best else ""
        print(f"Epoch {epoch:3d}/{max_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%) | Val Macro F1: {val_macro:.4f} | "
              f"Score: {val_score:.4f} (Best: {best_val_score:.4f} @ ep {best_epoch}){marker}")

    if patience_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch}. Restoring best weights from epoch {best_epoch}.")
        break

gate.load_state_dict(best_state)
gate.eval()

# Check learned weights on val set
with torch.no_grad():
    val_gate_w = gate(x_val_tensor).numpy()
mean_val_weights = val_gate_w.mean(axis=0)
print(f"\nAverage learned gate weights on Validation set:")
print(f"  XGBoost:  {mean_val_weights[0]:.4f} ({mean_val_weights[0]*100:.2f}%)")
print(f"  CatBoost: {mean_val_weights[1]:.4f} ({mean_val_weights[1]*100:.2f}%)")
print(f"  LightGBM: {mean_val_weights[2]:.4f} ({mean_val_weights[2]*100:.2f}%)")

# ── 7. Tuned Soft-Voting Ensemble (Grid Search on Validation Set) ──────────────
print("\n" + "=" * 78)
print("TUNED SOFT-VOTING ENSEMBLE (Grid-Search on Validation Set)")
print("=" * 78)

best_static_score = -1.0
best_static_w = None

for w1_int in range(0, 101, 2):
    for w2_int in range(0, 101 - w1_int, 2):
        w3_int = 100 - w1_int - w2_int
        w = np.array([w1_int, w2_int, w3_int], dtype=np.float32) / 100.0
        p_val_blend = (
            w[0] * val_probs[:, 0, :] +
            w[1] * val_probs[:, 1, :] +
            w[2] * val_probs[:, 2, :]
        )
        preds = np.argmax(p_val_blend, axis=1)
        score = accuracy_score(y_val, preds) + f1_score(y_val, preds, average="macro")
        if score > best_static_score:
            best_static_score = score
            best_static_w = w

print(f"Optimal static soft-voting weights:")
print(f"  XGBoost:  {best_static_w[0]:.2f} ({best_static_w[0]*100:.1f}%)")
print(f"  CatBoost: {best_static_w[1]:.2f} ({best_static_w[1]*100:.1f}%)")
print(f"  LightGBM: {best_static_w[2]:.2f} ({best_static_w[2]*100:.1f}%)")
print(f"  Validation Score (Acc + Macro F1): {best_static_score:.4f}")

# ── 8. Comprehensive Test Set Evaluation ──────────────────────────────────────
print("\n" + "=" * 78)
print("HELD-OUT TEST SET EVALUATION")
print("=" * 78)

# Expert predictions
y_pred_xgb = np.argmax(test_probs[:, 0, :], axis=1)
y_pred_cb  = np.argmax(test_probs[:, 1, :], axis=1)
y_pred_lgb = np.argmax(test_probs[:, 2, :], axis=1)

# Tuned Soft-Voting predictions
p_test_tuned = (
    best_static_w[0] * test_probs[:, 0, :] +
    best_static_w[1] * test_probs[:, 1, :] +
    best_static_w[2] * test_probs[:, 2, :]
)
y_pred_tuned = np.argmax(p_test_tuned, axis=1)

# MoE Gated predictions
x_test_tensor = torch.tensor(x_gate_test, dtype=torch.float32)
with torch.no_grad():
    test_gate_w = gate(x_test_tensor).numpy()  # (N_test, 3)
p_test_moe = np.sum(test_gate_w[:, :, None] * test_probs, axis=1)  # (N_test, 7)
y_pred_moe = np.argmax(p_test_moe, axis=1)

def evaluate(name, y_true, y_pred):
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    rep         = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm          = confusion_matrix(y_true, y_pred)
    mel_recall  = rep["mel"]["recall"]
    return {
        "model": name,
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "mel_recall": round(float(mel_recall), 6),
        "report": rep,
        "cm": cm.tolist(),
    }

res_xgb   = evaluate("XGBoost", y_test, y_pred_xgb)
res_lgb   = evaluate("LightGBM", y_test, y_pred_lgb)
res_cb    = evaluate("CatBoost", y_test, y_pred_cb)
res_tuned = evaluate("Tuned Soft-Voting", y_test, y_pred_tuned)
res_moe   = evaluate("MoE (88-D Gate)", y_test, y_pred_moe)

leaderboard = [res_xgb, res_lgb, res_cb, res_tuned, res_moe]

# ── 9. Leaderboard Display ────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("COMPLETE MODEL LEADERBOARD (Held-Out Test Set)")
print("=" * 78)

headers = ["Model", "Accuracy", "Macro F1", "Weighted F1", "Melanoma Recall"]
row_fmt = "{:<22} | {:<12} | {:<10} | {:<12} | {:<16}"
div = "-" * 22 + "-+-" + "-" * 12 + "-+-" + "-" * 10 + "-+-" + "-" * 12 + "-+-" + "-" * 16

print(row_fmt.format(*headers))
print(div)
for r in leaderboard:
    acc_s = f"{r['accuracy']*100:.2f}%"
    mel_s = f"{r['mel_recall']*100:.2f}%"
    print(row_fmt.format(r["model"], acc_s, f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}", mel_s))
print(div)

# Average gate weights
mean_test_w = test_gate_w.mean(axis=0)
print("\nAverage Gate Weights across Test Set:")
print(f"  XGBoost:  {mean_test_w[0]:.4f} ({mean_test_w[0]*100:.2f}%)")
print(f"  CatBoost: {mean_test_w[1]:.4f} ({mean_test_w[1]*100:.2f}%)")
print(f"  LightGBM: {mean_test_w[2]:.4f} ({mean_test_w[2]*100:.2f}%)")

print("\nAverage Expert Gating Weight by Ground-Truth Class (Test Set):")
print(f"{'Class':8s} | {'XGBoost':8s} | {'CatBoost':8s} | {'LightGBM':8s}")
print("-" * 40)
for i, cls in enumerate(class_names):
    mask = (y_test == i)
    mean_cls_w = test_gate_w[mask].mean(axis=0)
    print(f"{cls:8s} | {mean_cls_w[0]:8.4f} | {mean_cls_w[1]:8.4f} | {mean_cls_w[2]:8.4f}")

# MoE Detailed Classification Report
print("\n" + "=" * 78)
print("MoE Detailed Classification Report (Test Set):")
print("=" * 78)
print(classification_report(y_test, y_pred_moe, target_names=class_names))

print("MoE Confusion Matrix (rows=true, cols=pred):")
header_cm = f"{'':8s}" + "".join(f"{c:>7s}" for c in class_names)
print(header_cm)
cm_moe = np.array(res_moe["cm"])
for i, cls in enumerate(class_names):
    row_str = f"{cls:8s}" + "".join(f"{cm_moe[i,j]:>7d}" for j in range(num_classes))
    print(row_str)

# ── 10. Save Metrics & Checkpoints ────────────────────────────────────────────
metrics_data = {
    "leaderboard": [
        {
            "model": r["model"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "weighted_f1": r["weighted_f1"],
            "mel_recall": r["mel_recall"],
        }
        for r in leaderboard
    ],
    "tuned_soft_voting": {
        "weights": {
            "xgboost": round(float(best_static_w[0]), 4),
            "catboost": round(float(best_static_w[1]), 4),
            "lightgbm": round(float(best_static_w[2]), 4),
        },
        "accuracy": res_tuned["accuracy"],
        "macro_f1": res_tuned["macro_f1"],
        "weighted_f1": res_tuned["weighted_f1"],
        "mel_recall": res_tuned["mel_recall"],
    },
    "moe_gating": {
        "best_epoch": best_epoch,
        "mean_gate_weights": {
            "xgboost": round(float(mean_test_w[0]), 4),
            "catboost": round(float(mean_test_w[1]), 4),
            "lightgbm": round(float(mean_test_w[2]), 4),
            "by_class": {
                cls: [round(float(x), 4) for x in test_gate_w[y_test == i].mean(axis=0)]
                for i, cls in enumerate(class_names)
            },
        },
        "accuracy": res_moe["accuracy"],
        "macro_f1": res_moe["macro_f1"],
        "weighted_f1": res_moe["weighted_f1"],
        "mel_recall": res_moe["mel_recall"],
        "per_class": {
            cls: {
                "precision": round(res_moe["report"][cls]["precision"], 4),
                "recall":    round(res_moe["report"][cls]["recall"], 4),
                "f1":        round(res_moe["report"][cls]["f1-score"], 4),
                "support":   int(res_moe["report"][cls]["support"]),
            }
            for cls in class_names
        },
        "confusion_matrix": res_moe["cm"],
    },
}

with open(METRICS_OUT, "w") as f:
    json.dump(metrics_data, f, indent=2)
with open(METRICS_V2_OUT, "w") as f:
    json.dump(metrics_data, f, indent=2)
print(f"\nSaved metrics: {METRICS_OUT} and {METRICS_V2_OUT}")

torch.save(gate.state_dict(), str(GATE_MODEL_OUT))
torch.save(gate.state_dict(), str(GATE_V2_OUT))
print(f"Saved gate model weights: {GATE_MODEL_OUT} and {GATE_V2_OUT}")

# ── 11. Git Commit & Push ─────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("GIT COMMIT & PUSH")
print("=" * 78)
try:
    subprocess.run(["git", "add", "src/", "models/", "results/"], check=True)
    msg = (
        "feat: rebuild MoE with 88-D gating network, PCA features & smoothed weights\n\n"
        "- Standardized ResNet50 features via StandardScaler fit on train only\n"
        "- 5-fold cross-validated out-of-fold probability predictions with square-root smoothed weights\n"
        "- 64-D PCA visual context + 21-D expert probabilities + 3-D entropy scores (88-D input)\n"
        "- Tuned Soft-Voting ensemble & Gated Mixture of Experts\n"
        f"- Test Accuracy: MoE={res_moe['accuracy']*100:.2f}%, Soft-Voting={res_tuned['accuracy']*100:.2f}%\n"
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("Git commit and push completed successfully!")
except Exception as e:
    print(f"Git operation status: {e}")

print("\n[DONE] MoE final execution complete.")
