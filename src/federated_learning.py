"""
federated_learning.py
─────────────────────
Federated Learning (FedAvg) simulation for HAM10000 7-class skin lesion classification.

Scenario:
  Simulate 3 independent hospital clients (Hospital A, Hospital B, Hospital C)
  participating in collaborative federated training without sharing raw dermatoscopic
  images or patient visual embeddings.

Key Engineering Design:
  1. Strict Patient-Level Isolation:
     Partition 7,054 training samples into 3 disjoint client subsets using
     StratifiedGroupKFold(n_splits=3) on lesion_id, guaranteeing 0% patient/lesion leakage.
  2. Neural Classification Head:
     FederatedClassifier(nn.Module):
       nn.Linear(2048, 256) -> nn.BatchNorm1d(256) -> nn.ReLU() -> nn.Dropout(0.3) -> nn.Linear(256, 7)
  3. Decentralized Training with Square-Root Smoothed Weights:
     Each client computes local square-root smoothed class weights:
       w_i = sqrt(N_total / (7 * N_i))
     and trains for 3 local epochs per round using CrossEntropyLoss and Adam(lr=1e-3, weight_decay=1e-4).
  4. Server-Side Federated Averaging (FedAvg):
     Aggregates client state_dicts via sample-weighted parameter & buffer averaging across
     15 communication rounds.
  5. Comprehensive Evaluation & Telemetry:
     Tracks global model validation and test performance per round, records local hospital
     validation accuracies, saves the best global checkpoint to models/federated_global.pt,
     and exports complete telemetry to models/federated_metrics.json.
"""

import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH = RESULTS_DIR / "resnet50_features.npz"
SPLITS_CSV = RESULTS_DIR / "splits.csv"
LABEL_MAP_PATH = RESULTS_DIR / "label_mapping.json"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

MODEL_OUT = MODELS_DIR / "federated_global.pt"
METRICS_OUT = MODELS_DIR / "federated_metrics.json"


# ── Model Architecture ────────────────────────────────────────────────────────
class FederatedClassifier(nn.Module):
    """
    Neural classification head for federated learning:
      Linear(2048, 256) -> BatchNorm1d(256) -> ReLU() -> Dropout(0.3) -> Linear(256, 7)
    """

    def __init__(
        self,
        in_features: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 7,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


# ── Helper Evaluation Functions ───────────────────────────────────────────────
def evaluate_model(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    class_names: List[str],
    batch_size: int = 256,
) -> Dict:
    """
    Evaluates a PyTorch model on a dataset and returns loss, accuracy,
    macro F1, weighted F1, melanoma recall, per-class metrics, and confusion matrix.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    avg_loss = total_loss / len(y_true)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    mel_recall = float(report["mel"]["recall"]) if "mel" in report else 0.0

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "mel_recall": round(mel_recall, 4),
        "report": report,
        "confusion_matrix": cm,
    }


def local_train(
    client_model: nn.Module,
    X_client: torch.Tensor,
    y_client: torch.Tensor,
    class_weights: torch.Tensor,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> nn.Module:
    """
    Trains a local hospital client model for a fixed number of epochs.
    """
    client_model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        client_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    dataset = TensorDataset(X_client, y_client)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = client_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return client_model


def federated_averaging(
    client_states: Dict[str, Dict[str, torch.Tensor]],
    client_sample_counts: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """
    Performs FedAvg parameter and buffer aggregation weighted by client sample count.
    Cleanly handles floating point tensors (averaging) and integer buffers (e.g. num_batches_tracked).
    """
    total_samples = sum(client_sample_counts.values())
    client_names = list(client_states.keys())
    sample_weights = {
        name: client_sample_counts[name] / total_samples for name in client_names
    }

    reference_state = client_states[client_names[0]]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key, ref_tensor in reference_state.items():
        if torch.is_floating_point(ref_tensor):
            aggregated_state[key] = sum(
                sample_weights[name] * client_states[name][key].to(device)
                for name in client_names
            )
        else:
            # For integer buffers like bn.num_batches_tracked, use the first client's value
            aggregated_state[key] = ref_tensor.clone().to(device)

    return aggregated_state


# ── Main Federated Learning Routine ───────────────────────────────────────────
def main():
    print("=" * 80)
    print("HAM10000 Skin Lesion Classification  |  Federated Learning (FedAvg)")
    print("=" * 80)
    print(f"Executing on compute device: {device}")

    # 1. Load Data
    print(f"\n[1/6] Loading visual feature representations and splits metadata...")
    data = np.load(NPZ_PATH)
    X_train_raw = data["X_train"]
    y_train = data["y_train"]
    X_val_raw = data["X_val"]
    y_val = data["y_val"]
    X_test_raw = data["X_test"]
    y_test = data["y_test"]

    print(
        f"  Raw feature matrix shapes -> Train: {X_train_raw.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}"
    )

    splits_df = pd.read_csv(SPLITS_CSV)
    train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
    assert len(train_df) == len(
        X_train_raw
    ), f"Mismatch: splits.csv train rows ({len(train_df)}) vs X_train ({len(X_train_raw)})"

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    idx2cls = {v: k for k, v in label_map.items()}
    class_names = [idx2cls[i] for i in range(len(idx2cls))]
    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")

    # 2. Standardize Features using fitted scaler.joblib
    print(f"\n[2/6] Standardizing features using pre-fitted {SCALER_PATH.name}...")
    scaler = joblib.load(SCALER_PATH)
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    print(f"  Standardized X_train: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    print(f"  Standardized X_val  : mean={X_val.mean():.4f}, std={X_val.std():.4f}")
    print(f"  Standardized X_test : mean={X_test.mean():.4f}, std={X_test.std():.4f}")

    # Convert validation and test sets to PyTorch tensors
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 3. Simulate 3 Independent Hospital Clients
    print(
        f"\n[3/6] Simulating 3 independent hospital clients via StratifiedGroupKFold on lesion_id..."
    )
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    hospital_names = ["Hospital A", "Hospital B", "Hospital C"]
    clients: Dict[str, Dict] = {}

    for fold_idx, (_, client_idx) in enumerate(
        sgkf.split(X_train, y_train, groups=train_df["lesion_id"])
    ):
        h_name = hospital_names[fold_idx]
        h_y = y_train[client_idx]
        h_X = X_train[client_idx]
        h_lesions = set(train_df.iloc[client_idx]["lesion_id"])
        h_counts = [int(np.sum(h_y == c)) for c in range(num_classes)]
        n_samples = len(client_idx)

        # Square-root smoothed class weights: w_i = sqrt(N_total / (7 * N_i))
        # Compresses extreme weight disparity and maintains true-negative precision
        smoothed_weights = [
            float(np.sqrt(n_samples / (num_classes * max(h_counts[c], 1))))
            for c in range(num_classes)
        ]

        clients[h_name] = {
            "indices": client_idx,
            "X_tensor": torch.tensor(h_X, dtype=torch.float32),
            "y_tensor": torch.tensor(h_y, dtype=torch.long),
            "lesion_ids": h_lesions,
            "n_samples": n_samples,
            "counts": h_counts,
            "class_weights": torch.tensor(smoothed_weights, dtype=torch.float32),
        }

    # Verify zero lesion leakage across hospital clients
    for i in range(len(hospital_names)):
        for j in range(i + 1, len(hospital_names)):
            h1, h2 = hospital_names[i], hospital_names[j]
            overlap = clients[h1]["lesion_ids"] & clients[h2]["lesion_ids"]
            assert (
                len(overlap) == 0
            ), f"CRITICAL: Patient leakage detected between {h1} and {h2}: {len(overlap)} shared lesions!"

    print("  Zero patient-level leakage confirmed across all 3 hospitals:")
    for name, c in clients.items():
        print(
            f"    - {name:10s}: {c['n_samples']:5d} images | {len(c['lesion_ids']):5d} unique lesions | Class counts: {c['counts']}"
        )

    # 4. Initialize Global Model
    print(f"\n[4/6] Initializing FederatedClassifier neural head...")
    global_model = FederatedClassifier(
        in_features=2048, hidden_dim=256, num_classes=num_classes, dropout_rate=0.3
    ).to(device)

    total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # Initial zero-round evaluation (untrained)
    init_eval = evaluate_model(global_model, X_val_t, y_val_t, class_names)
    print(
        f"  Pre-training Val Accuracy: {init_eval['accuracy']:.4f}, Val Macro F1: {init_eval['macro_f1']:.4f}"
    )

    # 5. Federated Averaging (FedAvg) Training Loop
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    print(
        f"\n[5/6] Starting FedAvg over {NUM_ROUNDS} communication rounds ({LOCAL_EPOCHS} local epochs per round)..."
    )
    print("-" * 88)
    print(
        f"{'Round':^7} | {'Val Acc':^9} | {'Val Macro F1':^12} | {'Val W-F1':^10} | {'Test Acc':^9} | {'Test Mel Rec':^12} | {'Local Val Accs (A / B / C)':^26}"
    )
    print("-" * 88)

    telemetry: List[Dict] = []
    best_val_macro_f1 = -1.0
    best_round = -1
    best_global_state = None

    client_sample_counts = {name: c["n_samples"] for name, c in clients.items()}

    for round_num in range(1, NUM_ROUNDS + 1):
        client_states: Dict[str, Dict[str, torch.Tensor]] = {}
        local_val_accs: Dict[str, float] = {}

        # Broadcast global weights to each hospital client
        current_global_weights = copy.deepcopy(global_model.state_dict())

        for name, client in clients.items():
            # Clone global model for local client
            local_model = FederatedClassifier(
                in_features=2048,
                hidden_dim=256,
                num_classes=num_classes,
                dropout_rate=0.3,
            ).to(device)
            local_model.load_state_dict(copy.deepcopy(current_global_weights))

            # Train locally for LOCAL_EPOCHS
            local_model = local_train(
                client_model=local_model,
                X_client=client["X_tensor"],
                y_client=client["y_tensor"],
                class_weights=client["class_weights"],
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )

            # Evaluate local model on validation set before aggregation
            local_eval = evaluate_model(local_model, X_val_t, y_val_t, class_names)
            local_val_accs[name] = local_eval["accuracy"]

            # Save state dict for aggregation (kept in CPU memory)
            client_states[name] = {
                k: v.cpu().clone() for k, v in local_model.state_dict().items()
            }

        # Central Server Aggregation via FedAvg
        aggregated_weights = federated_averaging(
            client_states, client_sample_counts
        )
        global_model.load_state_dict(aggregated_weights)

        # Evaluate aggregated global model on validation and test sets
        val_metrics = evaluate_model(global_model, X_val_t, y_val_t, class_names)
        test_metrics = evaluate_model(global_model, X_test_t, y_test_t, class_names)

        # Check for best model checkpoint
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_round = round_num
            best_global_state = copy.deepcopy(global_model.state_dict())

        round_log = {
            "round": round_num,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_mel_recall": val_metrics["mel_recall"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_mel_recall": test_metrics["mel_recall"],
            "local_val_accuracies": local_val_accs,
        }
        telemetry.append(round_log)

        h_acc_str = f"{local_val_accs['Hospital A']:.3f} / {local_val_accs['Hospital B']:.3f} / {local_val_accs['Hospital C']:.3f}"
        print(
            f"{round_num:^7d} | {val_metrics['accuracy']:^9.4f} | {val_metrics['macro_f1']:^12.4f} | "
            f"{val_metrics['weighted_f1']:^10.4f} | {test_metrics['accuracy']:^9.4f} | "
            f"{test_metrics['mel_recall']:^12.4f} | {h_acc_str:^26}"
        )

    print("-" * 88)
    print(
        f"FedAvg complete. Best checkpoint achieved at Round {best_round} (Val Macro F1 = {best_val_macro_f1:.4f})."
    )

    # 6. Final Evaluation with Best Model Checkpoint
    print(
        f"\n[6/6] Restoring best global model checkpoint (Round {best_round}) for final test set evaluation..."
    )
    global_model.load_state_dict(best_global_state)
    torch.save(best_global_state, MODEL_OUT)
    print(f"  Saved best model checkpoint -> {MODEL_OUT}")

    final_test = evaluate_model(global_model, X_test_t, y_test_t, class_names)

    print("\n" + "=" * 80)
    print("FEDERATED GLOBAL MODEL: HELD-OUT TEST SET RESULTS (N = 1,497)")
    print("=" * 80)
    print(f"  Test Accuracy         : {final_test['accuracy']:.4f} ({final_test['accuracy']*100:.2f}%)")
    print(f"  Test Macro F1         : {final_test['macro_f1']:.4f}")
    print(f"  Test Weighted F1      : {final_test['weighted_f1']:.4f}")
    print(f"  Melanoma Recall (mel) : {final_test['mel_recall']:.4f} ({final_test['mel_recall']*100:.2f}%)")

    print("\nPer-Class Classification Report:")
    print(
        classification_report(
            y_test,
            torch.argmax(global_model(X_test_t.to(device)), dim=-1).cpu().numpy(),
            target_names=class_names,
            digits=4,
        )
    )

    print("Confusion Matrix (rows=True, cols=Pred):")
    header = f"{'':8s}" + "".join(f"{c:>7s}" for c in class_names)
    print(header)
    for i, cls in enumerate(class_names):
        row = f"{cls:8s}" + "".join(
            f"{final_test['confusion_matrix'][i][j]:>7d}" for j in range(num_classes)
        )
        print(row)

    # Export Full Telemetry and Metrics
    export_payload = {
        "model": "Federated Learning (FedAvg 3-Hospital)",
        "framework": "PyTorch FedAvg",
        "num_clients": 3,
        "num_rounds": NUM_ROUNDS,
        "local_epochs_per_round": LOCAL_EPOCHS,
        "client_distribution": {
            name: {
                "sample_count": c["n_samples"],
                "lesion_count": len(c["lesion_ids"]),
                "class_counts": {
                    cls: c["counts"][idx] for idx, cls in enumerate(class_names)
                },
                "smoothed_class_weights": {
                    cls: round(float(c["class_weights"][idx]), 4)
                    for idx, cls in enumerate(class_names)
                },
            }
            for name, c in clients.items()
        },
        "best_round": best_round,
        "best_val_macro_f1": best_val_macro_f1,
        "round_telemetry": telemetry,
        "final_test_metrics": {
            "accuracy": final_test["accuracy"],
            "macro_f1": final_test["macro_f1"],
            "weighted_f1": final_test["weighted_f1"],
            "mel_recall": final_test["mel_recall"],
            "per_class": {
                cls: {
                    "precision": round(final_test["report"][cls]["precision"], 4),
                    "recall": round(final_test["report"][cls]["recall"], 4),
                    "f1": round(final_test["report"][cls]["f1-score"], 4),
                    "support": int(final_test["report"][cls]["support"]),
                }
                for cls in class_names
            },
            "confusion_matrix": final_test["confusion_matrix"],
        },
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\n  Exported complete federated telemetry & metrics -> {METRICS_OUT}")
    print("=" * 80)
    print("Federated Learning (FedAvg) phase completed successfully!")


if __name__ == "__main__":
    main()
