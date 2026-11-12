"""
scripts/precompute_samples.py
─────────────────────────────
Precomputes exact ResNet-50 + GBDT + MoE + Federated Global inference
for the 7 representative held-out test samples and copies their JPEG images
into public/samples/ for static CDN serving on Vercel.
"""

import json
import shutil
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models as models
import joblib
import numpy as np
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import torch.nn as nn
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PUBLIC_SAMPLES_DIR = ROOT / "public" / "samples"
PUBLIC_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

class MoEGatingNetwork(nn.Module):
    def __init__(self, in_features=88, hidden_dim=32, num_experts=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.softmax(self.fc2(h))

class FederatedClassifier(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=256, num_classes=7, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.fc2(h)

def main():
    print("Loading models for precomputation...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.eval()

    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    pca = joblib.load(MODELS_DIR / "moe_pca.joblib")

    m_xgb = xgb.XGBClassifier()
    m_xgb.load_model(str(MODELS_DIR / "xgboost_resnet50.json"))

    m_cb = cb.CatBoostClassifier()
    m_cb.load_model(str(MODELS_DIR / "catboost_resnet50.cbm"))

    m_lgb = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_resnet50.txt"))

    gate = MoEGatingNetwork()
    gate.load_state_dict(torch.load(MODELS_DIR / "moe_gate.pt", map_location="cpu"))
    gate.eval()

    fed = FederatedClassifier()
    fed.load_state_dict(torch.load(MODELS_DIR / "federated_global.pt", map_location="cpu"))
    fed.eval()

    with open(RESULTS_DIR / "label_mapping.json") as f:
        label_map = json.load(f)
    idx2cls = {v: k for k, v in label_map.items()}
    class_names = [idx2cls[i] for i in range(len(idx2cls))]

    CLINICAL_TAXONOMY = {
        "nv": {"name": "Melanocytic Nevi", "nature": "Benign Mole", "risk": "Benign", "urgency": "Low risk; common benign proliferation of cutaneous melanocytes. Routine dermatological skin self-exams recommended.", "badge_color": "emerald", "malignant": False},
        "mel": {"name": "Melanoma", "nature": "Malignant Cancer", "risk": "Malignant", "urgency": "CRITICAL ONCOLOGICAL RISK: High metastatic potential. Immediate urgent dermoscopy, wide local excision, and histopathology required.", "badge_color": "rose", "malignant": True},
        "bkl": {"name": "Benign Keratosis-like Lesions", "nature": "Benign Keratosis", "risk": "Benign", "urgency": "Non-cancerous epithelial growth (seborrheic keratosis, solar lentigo, or lichen-planus like keratosis). Reassurance or cryo-ablation if symptomatic.", "badge_color": "emerald", "malignant": False},
        "bcc": {"name": "Basal Cell Carcinoma", "nature": "Malignant Cancer", "risk": "Malignant", "urgency": "HIGH RISK: Most prevalent non-melanocytic skin cancer. Locally invasive; prompt Mohs micrographic surgery or surgical excision advised.", "badge_color": "amber", "malignant": True},
        "akiec": {"name": "Actinic Keratoses / Bowen's Disease", "nature": "Pre-cancerous", "risk": "Precancerous", "urgency": "PRE-MALIGNANT WARNING: Keratinocyte atypia induced by ultraviolet radiation. Cryotherapy, 5-fluorouracil, or photodynamic therapy indicated to halt squamous progression.", "badge_color": "amber", "malignant": False},
        "vasc": {"name": "Vascular Lesions", "nature": "Benign Vascular", "risk": "Benign", "urgency": "Harmless endothelial proliferation (cherry angioma, angiokeratoma, pyogenic granuloma). Generally benign unless ulcerating.", "badge_color": "emerald", "malignant": False},
        "df": {"name": "Dermatofibroma", "nature": "Benign Fibroma", "risk": "Benign", "urgency": "Benign indolent dermal histiocytoma exhibiting positive 'dimple sign'. No oncological intervention needed.", "badge_color": "emerald", "malignant": False}
    }

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    splits = pd.read_csv(RESULTS_DIR / "splits.csv")
    test_splits = splits[splits["split"] == "test"]

    precomputed = {}
    samples_list = []

    for cls in class_names:
        rows = test_splits[test_splits["dx"] == cls]
        for _, row in rows.iterrows():
            fp = Path(row["filepath"])
            if fp.exists():
                img_id = row["image_id"]
                dest_fp = PUBLIC_SAMPLES_DIR / f"{img_id}.jpg"
                shutil.copyfile(fp, dest_fp)
                print(f"Copied {img_id}.jpg to public/samples/")

                img = Image.open(fp).convert("RGB")
                tensor_img = transform(img).unsqueeze(0)
                with torch.no_grad():
                    raw_feat = resnet(tensor_img).cpu().numpy().reshape(1, 2048)
                scaled_feat = scaler.transform(raw_feat)
                pca_feat = pca.transform(scaled_feat)

                p_xgb = m_xgb.predict_proba(scaled_feat)[0]
                p_cb = m_cb.predict_proba(scaled_feat)[0]
                p_lgb = m_lgb.predict(scaled_feat)[0]

                expert_probs = np.stack([p_xgb, p_cb, p_lgb], axis=0)
                p_clip = np.clip(expert_probs, 1e-8, 1.0)
                entropies = -np.sum(p_clip * np.log(p_clip), axis=1)
                probs_flat = expert_probs.reshape(1, 21)

                x_gate = np.concatenate([pca_feat, probs_flat, entropies.reshape(1, 3)], axis=1).astype(np.float32)

                with torch.no_grad():
                    gate_weights = gate(torch.tensor(x_gate, dtype=torch.float32)).cpu().numpy()[0]
                    w_xgb, w_cb, w_lgb = float(gate_weights[0]), float(gate_weights[1]), float(gate_weights[2])
                    p_moe = (w_xgb * p_xgb) + (w_cb * p_cb) + (w_lgb * p_lgb)
                    fed_logits = fed(torch.tensor(scaled_feat, dtype=torch.float32))
                    p_fed = torch.softmax(fed_logits, dim=-1).cpu().numpy()[0]

                def build_diag(probs):
                    top_idx = int(np.argmax(probs))
                    code = idx2cls[top_idx]
                    meta = CLINICAL_TAXONOMY.get(code, {})
                    return {
                        "code": code,
                        "name": meta.get("name", code),
                        "nature": meta.get("nature", "Unknown"),
                        "risk": meta.get("risk", "Unknown"),
                        "urgency": meta.get("urgency", ""),
                        "badge_color": meta.get("badge_color", "slate"),
                        "malignant": meta.get("malignant", False),
                        "confidence": round(float(probs[top_idx]) * 100, 2),
                        "probabilities": {idx2cls[i]: float(probs[i]) for i in range(len(class_names))}
                    }

                diag_fed = build_diag(p_fed)
                diag_moe = build_diag(p_moe)

                precomputed[img_id] = {
                    "status": "success",
                    "image_id": img_id,
                    "dx": cls,
                    "lesion_id": row["lesion_id"],
                    "federated_prediction": diag_fed,
                    "moe_prediction": diag_moe,
                    "gate_weights": {
                        "xgboost": round(w_xgb, 4),
                        "catboost": round(w_cb, 4),
                        "lightgbm": round(w_lgb, 4)
                    },
                    "expert_entropies": {
                        "xgboost": round(float(entropies[0]), 4),
                        "catboost": round(float(entropies[1]), 4),
                        "lightgbm": round(float(entropies[2]), 4)
                    },
                    "expert_predictions": {
                        "xgboost": build_diag(p_xgb),
                        "catboost": build_diag(p_cb),
                        "lightgbm": build_diag(p_lgb)
                    },
                    "ground_truth": {
                        "image_id": img_id,
                        "code": cls,
                        "name": CLINICAL_TAXONOMY[cls]["name"],
                        "risk": CLINICAL_TAXONOMY[cls]["risk"],
                        "lesion_id": row["lesion_id"]
                    }
                }

                samples_list.append({
                    "image_id": img_id,
                    "dx": cls,
                    "lesion_id": row["lesion_id"],
                    "name": CLINICAL_TAXONOMY[cls]["name"],
                    "nature": CLINICAL_TAXONOMY[cls]["nature"],
                    "risk": CLINICAL_TAXONOMY[cls]["risk"],
                    "badge_color": CLINICAL_TAXONOMY[cls]["badge_color"],
                    "image_url": f"/samples/{img_id}.jpg"
                })

                print(f"Precomputed {cls:6s} -> {img_id}: Fed={diag_fed['code']}, MoE={diag_moe['code']}")
                break

    with open(MODELS_DIR / "precomputed_sample_predictions.json", "w") as f:
        json.dump(precomputed, f, indent=2)

    with open(MODELS_DIR / "curated_samples.json", "w") as f:
        json.dump(samples_list, f, indent=2)

    print("Successfully generated:")
    print(f"  - {MODELS_DIR / 'precomputed_sample_predictions.json'}")
    print(f"  - {MODELS_DIR / 'curated_samples.json'}")
    print(f"  - {PUBLIC_SAMPLES_DIR} (7 sample images copied)")

if __name__ == "__main__":
    main()
