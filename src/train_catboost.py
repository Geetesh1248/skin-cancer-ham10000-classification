"""
train_catboost.py
─────────────────
CatBoost classifier on ResNet50 2048-d features for HAM10000 7-class skin
lesion classification.

Class imbalance strategy: built-in class_weights parameter
  weight_i = total_samples / (n_classes * count_i)
"""

import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH       = RESULTS_DIR / "resnet50_features.npz"
LABEL_MAP_PATH = RESULTS_DIR / "label_mapping.json"
MODEL_OUT      = MODELS_DIR  / "catboost_resnet50.cbm"
METRICS_OUT    = MODELS_DIR  / "catboost_metrics.json"

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("CatBoost  |  HAM10000  |  ResNet50 features")
print("=" * 60)

data = np.load(NPZ_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]
print(f"Train : {X_train.shape}  Val : {X_val.shape}  Test : {X_test.shape}")

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
idx2cls = {v: k for k, v in label_map.items()}
class_names = [idx2cls[i] for i in range(len(idx2cls))]
print(f"Classes: {class_names}")

# ── Balanced class weights: total / (n_classes * count_i) ─────────────────────
n_classes      = len(class_names)
total_samples  = len(y_train)
class_weights  = {}
print("\nComputed class weights (balanced):")
for i, cls in enumerate(class_names):
    count = (y_train == i).sum()
    w = total_samples / (n_classes * count)
    class_weights[i] = w
    print(f"  {cls:6s}  count={count:5d}  weight={w:.4f}")

# CatBoost expects class_weights as a list indexed by class label
cw_list = [class_weights[i] for i in range(n_classes)]

# -- Model ---------------------------------------------------------------------
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=42,
    verbose=50,
    class_weights=cw_list,
    use_best_model=True,
    early_stopping_rounds=30,
)

print("\nTraining CatBoost (500 iters max, depth=6, lr=0.1, early_stopping_rounds=30)...")
print("-" * 60)
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
)
print("-" * 60)
if hasattr(model, "get_best_iteration"):
    print(f"Best iteration: {model.get_best_iteration()}")

# ── Evaluate on TEST set ──────────────────────────────────────────────────────
y_pred = model.predict(X_test).flatten()

acc         = accuracy_score(y_test, y_pred)
macro_f1    = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
report      = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
cm          = confusion_matrix(y_test, y_pred)
mel_recall  = report["mel"]["recall"]

print("\n" + "=" * 60)
print("TEST SET RESULTS  --  CatBoost")
print("=" * 60)
print(f"  Accuracy        : {acc:.4f}")
print(f"  Macro F1        : {macro_f1:.4f}")
print(f"  Weighted F1     : {weighted_f1:.4f}")
print(f"  Melanoma recall : {mel_recall:.4f}  <- clinically critical")

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
