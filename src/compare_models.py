"""
compare_models.py
─────────────────
Displays comparison table for XGBoost, CatBoost, and LightGBM models on HAM10000.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

files = {
    "XGBoost": MODELS_DIR / "xgboost_metrics.json",
    "CatBoost": MODELS_DIR / "catboost_metrics.json",
    "LightGBM": MODELS_DIR / "lightgbm_metrics.json",
}

print("\n" + "=" * 78)
print("BASE MODELS PERFORMANCE COMPARISON (Square-Root Smoothed Weights + Scaler)")
print("=" * 78)

headers = ["Model", "Accuracy", "Macro F1", "Weighted F1", "Melanoma Recall", "Best Iter"]
row_fmt = "{:<12} | {:<10} | {:<10} | {:<12} | {:<17} | {:<10}"
divider = "-" * 12 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 12 + "-+-" + "-" * 17 + "-+-" + "-" * 10

print(row_fmt.format(*headers))
print(divider)

for model_name, path in files.items():
    if not path.exists():
        print(f"{model_name:<12} | [NOT FOUND: {path.name}]")
        continue
    with open(path) as f:
        data = json.load(f)
    acc = f"{data.get('accuracy', 0.0):.4f} ({data.get('accuracy', 0.0)*100:.2f}%)"
    macro_f1 = f"{data.get('macro_f1', 0.0):.4f}"
    weighted_f1 = f"{data.get('weighted_f1', 0.0):.4f}"
    mel_recall = f"{data.get('mel_recall', 0.0):.4f} ({data.get('mel_recall', 0.0)*100:.2f}%)"
    best_iter = str(data.get("best_iteration", "N/A"))
    print(row_fmt.format(model_name, acc, macro_f1, weighted_f1, mel_recall, best_iter))

print(divider)
print("=" * 78 + "\n")
