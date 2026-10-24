"""
train_catboost.py
─────────────────
CatBoost classifier on ResNet50 2048-d features for HAM10000 7-class skin
lesion classification.

Pipeline:
1. Load results/resnet50_features.npz
2. StandardScaler fit ONLY on X_train, transforming X_train, X_val, X_test
3. Square-Root Smoothed Class Weights:
     w_i = (Total_Samples / (7 * count_i)) ** 0.5
4. Train CatBoostClassifier:
     iterations=500, learning_rate=0.05, depth=6, early_stopping_rounds=40
     monitoring validation MultiClass loss
5. Save model, validation and test predict_proba to disk, and update metrics JSON
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH       = RESULTS_DIR / "resnet50_features.npz"
LABEL_MAP_PATH = RESULTS_DIR / "label_mapping.json"
MODEL_OUT      = MODELS_DIR  / "catboost_resnet50.cbm"
METRICS_OUT    = MODELS_DIR  / "catboost_metrics.json"

VAL_PROBS_OUT  = MODELS_DIR  / "catboost_val_probs.npy"
TEST_PROBS_OUT = MODELS_DIR  / "catboost_test_probs.npy"
VAL_PROBS_RES  = RESULTS_DIR / "catboost_val_probs.npy"
TEST_PROBS_RES = RESULTS_DIR / "catboost_test_probs.npy"
PROBS_NPZ_RES  = RESULTS_DIR / "catboost_probs.npz"

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("CatBoost  |  HAM10000  |  ResNet50 features")
print("=" * 60)

data = np.load(NPZ_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]
print(f"Loaded raw shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
idx2cls = {v: k for k, v in label_map.items()}
class_names = [idx2cls[i] for i in range(len(idx2cls))]
n_classes = len(class_names)
print(f"Classes ({n_classes}): {class_names}")

# ── 2. StandardScaler fit ONLY on X_train ─────────────────────────────────────
print("\nApplying StandardScaler (fit ONLY on X_train)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
print(f"  X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
print(f"  X_val   mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
print(f"  X_test  mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")

# ── 3. Square-Root Smoothed Class Weights ─────────────────────────────────────
total_samples = len(y_train)
class_weights = {}
print(f"\nClass distribution in train & Square-Root Smoothed Weights (Total={total_samples}):")
for i, cls in enumerate(class_names):
    count_i = int(np.sum(y_train == i))
    w_i = float((total_samples / (n_classes * count_i)) ** 0.5)
    class_weights[i] = w_i
    print(f"  Class {i} ({cls:6s}): count={count_i:5d}  weight={w_i:.4f}")

cw_list = [class_weights[i] for i in range(n_classes)]

# ── 4. Train Model ────────────────────────────────────────────────────────────
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    verbose=50,
    class_weights=cw_list,
    use_best_model=True,
    early_stopping_rounds=40,
)

print("\nTraining CatBoost (500 iters max, early_stopping_rounds=40, lr=0.05, depth=6)...")
print("-" * 60)
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
)
print("-" * 60)
best_iter = getattr(model, "get_best_iteration", lambda: None)()
if best_iter is not None:
    print(f"Best iteration: {best_iter}")

# ── 5. Probability predictions & Evaluation ───────────────────────────────────
val_probs  = model.predict_proba(X_val)
test_probs = model.predict_proba(X_test)
y_pred     = np.argmax(test_probs, axis=1)

# Save probability predictions
np.save(VAL_PROBS_OUT, val_probs)
np.save(TEST_PROBS_OUT, test_probs)
np.save(VAL_PROBS_RES, val_probs)
np.save(TEST_PROBS_RES, test_probs)
np.savez_compressed(PROBS_NPZ_RES, val_probs=val_probs, test_probs=test_probs)
print(f"\nSaved predict_proba arrays ->")
print(f"  Val probs  : {VAL_PROBS_OUT} (shape: {val_probs.shape})")
print(f"  Test probs : {TEST_PROBS_OUT} (shape: {test_probs.shape})")

acc         = accuracy_score(y_test, y_pred)
macro_f1    = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
report      = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
cm          = confusion_matrix(y_test, y_pred)
mel_recall  = report["mel"]["recall"]

print("\n" + "=" * 60)
print("TEST SET RESULTS  --  CatBoost")
print("=" * 60)
print(f"  Accuracy        : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Macro F1        : {macro_f1:.4f}")
print(f"  Weighted F1     : {weighted_f1:.4f}")
print(f"  Melanoma recall : {mel_recall:.4f} ({mel_recall*100:.2f}%)  <- clinically critical")

print("\nPer-class recall (sensitivity):")
for cls in class_names:
    print(f"  {cls:6s}  recall={report[cls]['recall']:.4f}  "
          f"precision={report[cls]['precision']:.4f}  "
          f"f1={report[cls]['f1-score']:.4f}  "
          f"support={int(report[cls]['support'])}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix (rows=true, cols=pred):")
header = f"{'':8s}" + "".join(f"{c:>7s}" for c in class_names)
print(header)
for i, cls in enumerate(class_names):
    row = f"{cls:8s}" + "".join(f"{cm[i,j]:>7d}" for j in range(len(class_names)))
    print(row)

# ── Save model ────────────────────────────────────────────────────────────────
model.save_model(str(MODEL_OUT))
print(f"\nModel saved : {MODEL_OUT}")

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics = {
    "model": "CatBoost",
    "accuracy": round(acc, 6),
    "macro_f1": round(macro_f1, 6),
    "weighted_f1": round(weighted_f1, 6),
    "mel_recall": round(mel_recall, 6),
    "best_iteration": int(best_iter) if best_iter is not None else None,
    "per_class": {
        cls: {
            "precision": round(report[cls]["precision"], 6),
            "recall":    round(report[cls]["recall"], 6),
            "f1":        round(report[cls]["f1-score"], 6),
            "support":   int(report[cls]["support"]),
        }
        for cls in class_names
    },
    "confusion_matrix": cm.tolist(),
}
with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved : {METRICS_OUT}")
print("\n[DONE] CatBoost training complete.")
