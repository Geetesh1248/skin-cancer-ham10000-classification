"""
app/main.py
───────────
FastAPI Clinical Inference & Telemetry Backend for HAM10000 Skin Lesion Diagnosis.

Key Endpoints:
  1. POST /api/predict: Live 2048-D ResNet50 extraction -> GBDT probability & entropy estimation
     -> 88-D MoE Gating Router -> Federated Global Classifier prediction.
  2. GET /api/samples: Pre-selected test set samples (1 per class) with metadata.
  3. GET /api/sample-image/{image_id}: Direct static streaming of test sample images.
  4. GET /api/benchmarks: Full test leaderboard, MoE routing breakdown, and 15-round FedAvg telemetry.
  5. GET /: Serves the modern interactive clinical dashboard frontend.
"""

import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import catboost as cb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import xgboost as xgb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = Path(__file__).resolve().parent / "static"

LABEL_MAP_PATH = RESULTS_DIR / "label_mapping.json"
SPLITS_CSV = RESULTS_DIR / "splits.csv"

# Preprocessing transform
IMAGE_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Clinical pathology knowledge base
CLINICAL_TAXONOMY = {
    "nv": {
        "name": "Melanocytic Nevi",
        "nature": "Benign Mole",
        "risk": "Benign",
        "urgency": "Low risk; common benign proliferation of cutaneous melanocytes. Routine dermatological skin self-exams recommended.",
        "badge_color": "emerald",
        "malignant": False,
    },
    "mel": {
        "name": "Melanoma",
        "nature": "Malignant Cancer",
        "risk": "Malignant",
        "urgency": "CRITICAL ONCOLOGICAL RISK: High metastatic potential. Immediate urgent dermoscopy, wide local excision, and histopathology required.",
        "badge_color": "rose",
        "malignant": True,
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesions",
        "nature": "Benign Keratosis",
        "risk": "Benign",
        "urgency": "Non-cancerous epithelial growth (seborrheic keratosis, solar lentigo, or lichen-planus like keratosis). Reassurance or cryo-ablation if symptomatic.",
        "badge_color": "emerald",
        "malignant": False,
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "nature": "Malignant Cancer",
        "risk": "Malignant",
        "urgency": "HIGH RISK: Most prevalent non-melanocytic skin cancer. Locally invasive; prompt Mohs micrographic surgery or surgical excision advised.",
        "badge_color": "amber",
        "malignant": True,
    },
    "akiec": {
        "name": "Actinic Keratoses / Bowen's Disease",
        "nature": "Pre-cancerous",
        "risk": "Precancerous",
        "urgency": "PRE-MALIGNANT WARNING: Keratinocyte atypia induced by ultraviolet radiation. Cryotherapy, 5-fluorouracil, or photodynamic therapy indicated to halt squamous progression.",
        "badge_color": "amber",
        "malignant": False,
    },
    "vasc": {
        "name": "Vascular Lesions",
        "nature": "Benign Vascular",
        "risk": "Benign",
        "urgency": "Harmless endothelial proliferation (cherry angioma, angiokeratoma, pyogenic granuloma). Generally benign unless ulcerating.",
        "badge_color": "emerald",
        "malignant": False,
    },
    "df": {
        "name": "Dermatofibroma",
        "nature": "Benign Fibroma",
        "risk": "Benign",
        "urgency": "Benign indolent dermal histiocytoma exhibiting positive 'dimple sign'. No oncological intervention needed.",
        "badge_color": "emerald",
        "malignant": False,
    },
}


# ── PyTorch Architecture Definitions ──────────────────────────────────────────
class MoEGatingNetwork(nn.Module):
    """
    88-D Multimodal Gating Network:
      Linear(88, 32) -> BatchNorm1d(32) -> ReLU() -> Dropout(0.3) -> Linear(32, 3) -> Softmax(dim=-1)
    """

    def __init__(
        self, in_features: int = 88, hidden_dim: int = 32, num_experts: int = 3
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return self.softmax(logits)


class FederatedClassifier(nn.Module):
    """
    Federated Global Model Classifier:
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


# ── Application State Container ───────────────────────────────────────────────
class ModelContainer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.resnet: Optional[nn.Module] = None
        self.scaler = None
        self.pca = None
        self.m_xgb = None
        self.m_cb = None
        self.m_lgb = None
        self.moe_gate: Optional[nn.Module] = None
        self.fed_global: Optional[nn.Module] = None
        self.label_map: Dict[str, int] = {}
        self.idx2cls: Dict[int, str] = {}
        self.class_names: List[str] = []
        self.num_classes: int = 7
        self.test_samples: Dict[str, Dict] = {}
        self.benchmarks_payload: Dict = {}


models_app = ModelContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: Load all models, scalers, weights, and sample data ─────────────
    print("=" * 70)
    print("Initializing HAM10000 Clinical Inference Backend...")
    print("=" * 70)

    # 1. Labels
    with open(LABEL_MAP_PATH) as f:
        models_app.label_map = json.load(f)
    models_app.idx2cls = {v: k for k, v in models_app.label_map.items()}
    models_app.class_names = [
        models_app.idx2cls[i] for i in range(len(models_app.idx2cls))
    ]
    models_app.num_classes = len(models_app.class_names)
    print(f"  Loaded 7 diagnostic classes: {models_app.class_names}")

    # 2. ResNet50 Feature Extractor
    print("  Loading ResNet-50 visual backbone (IMAGENET1K_V1)...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet.eval()
    models_app.resnet = resnet

    # 3. Scaler and PCA
    print("  Loading StandardScaler and 64-D PCA transformation models...")
    models_app.scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    models_app.pca = joblib.load(MODELS_DIR / "moe_pca.joblib")

    # 4. GBDT Base Classifiers
    print("  Loading frozen GBDT experts (XGBoost, CatBoost, LightGBM)...")
    models_app.m_xgb = xgb.XGBClassifier()
    models_app.m_xgb.load_model(str(MODELS_DIR / "xgboost_resnet50.json"))

    models_app.m_cb = cb.CatBoostClassifier()
    models_app.m_cb.load_model(str(MODELS_DIR / "catboost_resnet50.cbm"))

    models_app.m_lgb = lgb.Booster(
        model_file=str(MODELS_DIR / "lightgbm_resnet50.txt")
    )

    # 5. Neural Gate & Federated Global Model
    print("  Loading 88-D MoE Gating Network & Federated Global Model...")
    gate = MoEGatingNetwork(in_features=88, hidden_dim=32, num_experts=3)
    gate.load_state_dict(
        torch.load(MODELS_DIR / "moe_gate.pt", map_location=models_app.device)
    )
    gate.eval()
    models_app.moe_gate = gate

    fed = FederatedClassifier(
        in_features=2048, hidden_dim=256, num_classes=models_app.num_classes
    )
    fed.load_state_dict(
        torch.load(
            MODELS_DIR / "federated_global.pt", map_location=models_app.device
        )
    )
    fed.eval()
    models_app.fed_global = fed

    # 6. Test Set Samples
    print("  Curating 7 verified held-out test samples (1 per pathology)...")
    splits_df = pd.read_csv(SPLITS_CSV)
    test_df = splits_df[splits_df["split"] == "test"].copy()

    for cls in models_app.class_names:
        class_rows = test_df[test_df["dx"] == cls]
        for _, row in class_rows.iterrows():
            fp = Path(row["filepath"])
            if fp.exists():
                models_app.test_samples[row["image_id"]] = {
                    "image_id": row["image_id"],
                    "dx": row["dx"],
                    "lesion_id": row["lesion_id"],
                    "filepath": str(fp),
                    "clinical_info": CLINICAL_TAXONOMY.get(row["dx"], {}),
                }
                break
    print(
        f"  Successfully curated {len(models_app.test_samples)} representative test samples."
    )

    # 7. Benchmarks Payload
    print("  Aggregating complete multi-model leaderboard & FedAvg telemetry...")
    with open(MODELS_DIR / "xgboost_metrics.json") as f:
        xgb_m = json.load(f)
    with open(MODELS_DIR / "catboost_metrics.json") as f:
        cb_m = json.load(f)
    with open(MODELS_DIR / "lightgbm_metrics.json") as f:
        lgb_m = json.load(f)
    with open(MODELS_DIR / "moe_metrics.json") as f:
        moe_m = json.load(f)
    with open(MODELS_DIR / "federated_metrics.json") as f:
        fed_m = json.load(f)

    models_app.benchmarks_payload = {
        "leaderboard": [
            {
                "model": "Federated Global Model (FedAvg)",
                "paradigm": "Federated Learning (3 Hospitals)",
                "accuracy": 0.7722,
                "macro_f1": 0.5751,
                "weighted_f1": 0.7760,
                "mel_recall": 0.5536,
                "status": "Winning Benchmark",
                "color": "emerald",
            },
            {
                "model": "XGBoost (Hist)",
                "paradigm": "Centralized GBDT",
                "accuracy": 0.7675,
                "macro_f1": 0.5078,
                "weighted_f1": 0.7507,
                "mel_recall": 0.4405,
                "status": "High Benign Specificity",
                "color": "blue",
            },
            {
                "model": "MoE (88-D Context Gate)",
                "paradigm": "Centralized Dynamic Ensemble",
                "accuracy": 0.7649,
                "macro_f1": 0.5204,
                "weighted_f1": 0.7528,
                "mel_recall": 0.4464,
                "status": "Dynamic Routing",
                "color": "indigo",
            },
            {
                "model": "Tuned Soft-Voting",
                "paradigm": "Centralized Static Ensemble",
                "accuracy": 0.7602,
                "macro_f1": 0.5269,
                "weighted_f1": 0.7528,
                "mel_recall": 0.4821,
                "status": "Static Weighting",
                "color": "slate",
            },
            {
                "model": "CatBoost (Oblivious)",
                "paradigm": "Centralized GBDT",
                "accuracy": 0.7595,
                "macro_f1": 0.5337,
                "weighted_f1": 0.7518,
                "mel_recall": 0.4702,
                "status": "Oblivious Regularized",
                "color": "amber",
            },
            {
                "model": "LightGBM (Leaf-wise)",
                "paradigm": "Centralized GBDT",
                "accuracy": 0.7415,
                "macro_f1": 0.4845,
                "weighted_f1": 0.7386,
                "mel_recall": 0.4940,
                "status": "Leaf-wise Split",
                "color": "cyan",
            },
        ],
        "moe_routing_breakdown": {
            "global_weights": {
                "xgboost": 0.5906,
                "lightgbm": 0.2785,
                "catboost": 0.1310,
            },
            "by_class": {
                "nv": {
                    "xgboost": 0.6625,
                    "catboost": 0.0787,
                    "lightgbm": 0.2588,
                    "rationale": (
                        "Delegates strongly to XGBoost for maximum specificity"
                        " on majority benign nevi."
                    ),
                },
                "mel": {
                    "xgboost": 0.5183,
                    "catboost": 0.1353,
                    "lightgbm": 0.3464,
                    "rationale": (
                        "Routes to XGBoost and LightGBM (LightGBM has the"
                        " highest raw melanoma recall)."
                    ),
                },
                "bkl": {
                    "xgboost": 0.5088,
                    "catboost": 0.1980,
                    "lightgbm": 0.2932,
                    "rationale": (
                        "Prefers XGBoost for textured border discrimination."
                    ),
                },
                "bcc": {
                    "xgboost": 0.2683,
                    "catboost": 0.3941,
                    "lightgbm": 0.3376,
                    "rationale": (
                        "Transfers control to CatBoost and LightGBM for basal"
                        " cell epithelial malignancy."
                    ),
                },
                "df": {
                    "xgboost": 0.2926,
                    "catboost": 0.4389,
                    "lightgbm": 0.2685,
                    "rationale": (
                        "Strongly favors CatBoost where dermatofibroma"
                        " precision reaches 100%."
                    ),
                },
                "akiec": {
                    "xgboost": 0.3944,
                    "catboost": 0.3227,
                    "lightgbm": 0.2828,
                    "rationale": (
                        "Harmonious three-way consensus across all base"
                        " estimators."
                    ),
                },
                "vasc": {
                    "xgboost": 0.3379,
                    "catboost": 0.3207,
                    "lightgbm": 0.3414,
                    "rationale": (
                        "Equitable distribution over distinct vascular"
                        " structures."
                    ),
                },
            },
        },
        "federated_telemetry": fed_m.get("round_telemetry", []),
        "federated_clients": fed_m.get("client_distribution", {}),
        "federated_final_test": fed_m.get("final_test_metrics", {}),
    }

    print("=" * 70)
    print("Backend ready to serve clinical inference on http://localhost:8000")
    print("=" * 70)

    yield


# ── FastAPI App Setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="HAM10000 Skin Lesion Diagnosis | Live MoE & FedAvg Dashboard",
    description="Full-stack clinical diagnostic system supporting live 2048-D ResNet50 extraction, 88-D Mixture of Experts routing, and 3-Hospital Federated Learning evaluation.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Internal Inference Pipeline ───────────────────────────────────────────────
def process_and_infer(img: Image.Image, ground_truth: Optional[Dict] = None):
    # 1. Preprocess and feature extraction
    img_rgb = img.convert("RGB")
    tensor_img = IMAGE_TRANSFORM(img_rgb).unsqueeze(0)

    with torch.no_grad():
        raw_feat = models_app.resnet(tensor_img).cpu().numpy().reshape(1, 2048)

    # 2. Standardization & PCA
    scaled_feat = models_app.scaler.transform(raw_feat)
    pca_feat = models_app.pca.transform(scaled_feat)

    # 3. GBDT Base Probabilities
    p_xgb = models_app.m_xgb.predict_proba(scaled_feat)[0]
    p_cb = models_app.m_cb.predict_proba(scaled_feat)[0]
    p_lgb = models_app.m_lgb.predict(scaled_feat)[0]

    # Stacking (1, 3, 7) for entropy computation
    expert_probs = np.stack([p_xgb, p_cb, p_lgb], axis=0)  # (3, 7)
    p_clip = np.clip(expert_probs, 1e-8, 1.0)
    entropies = -np.sum(p_clip * np.log(p_clip), axis=1)  # (3,)
    probs_flat = expert_probs.reshape(1, 21)

    # 4. 88-D Gate Vector
    x_gate = np.concatenate(
        [pca_feat, probs_flat, entropies.reshape(1, 3)], axis=1
    ).astype(np.float32)

    with torch.no_grad():
        gate_weights = (
            models_app.moe_gate(
                torch.tensor(x_gate, dtype=torch.float32, device=models_app.device)
            )
            .cpu()
            .numpy()[0]
        )

        w_xgb = float(gate_weights[0])
        w_cb = float(gate_weights[1])
        w_lgb = float(gate_weights[2])

        p_moe = (w_xgb * p_xgb) + (w_cb * p_cb) + (w_lgb * p_lgb)

        # 5. Federated Global Classifier Prediction
        fed_logits = models_app.fed_global(
            torch.tensor(scaled_feat, dtype=torch.float32, device=models_app.device)
        )
        p_fed = torch.softmax(fed_logits, dim=-1).cpu().numpy()[0]

    # Helper to construct model summary
    def build_diagnosis(probs: np.ndarray):
        top_idx = int(np.argmax(probs))
        top_code = models_app.idx2cls[top_idx]
        conf = float(probs[top_idx])
        meta = CLINICAL_TAXONOMY.get(top_code, {})
        probs_dict = {
            models_app.idx2cls[i]: float(probs[i])
            for i in range(models_app.num_classes)
        }
        return {
            "code": top_code,
            "name": meta.get("name", top_code),
            "nature": meta.get("nature", "Unknown"),
            "risk": meta.get("risk", "Unknown"),
            "urgency": meta.get("urgency", ""),
            "badge_color": meta.get("badge_color", "slate"),
            "malignant": meta.get("malignant", False),
            "confidence": round(conf * 100, 2),
            "probabilities": probs_dict,
        }

    fed_summary = build_diagnosis(p_fed)
    moe_summary = build_diagnosis(p_moe)

    experts_summary = {
        "xgboost": build_diagnosis(p_xgb),
        "catboost": build_diagnosis(p_cb),
        "lightgbm": build_diagnosis(p_lgb),
    }

    return {
        "status": "success",
        "federated_prediction": fed_summary,
        "moe_prediction": moe_summary,
        "gate_weights": {
            "xgboost": round(w_xgb, 4),
            "catboost": round(w_cb, 4),
            "lightgbm": round(w_lgb, 4),
        },
        "expert_entropies": {
            "xgboost": round(float(entropies[0]), 4),
            "catboost": round(float(entropies[1]), 4),
            "lightgbm": round(float(entropies[2]), 4),
        },
        "expert_predictions": experts_summary,
        "ground_truth": ground_truth,
    }


# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "resnet50": models_app.resnet is not None,
            "scaler": models_app.scaler is not None,
            "pca": models_app.pca is not None,
            "xgboost": models_app.m_xgb is not None,
            "catboost": models_app.m_cb is not None,
            "lightgbm": models_app.m_lgb is not None,
            "moe_gate": models_app.moe_gate is not None,
            "federated_global": models_app.fed_global is not None,
        },
        "curated_samples": len(models_app.test_samples),
    }


@app.get("/api/samples")
async def get_test_samples():
    """Returns curated held-out test set samples (1 per class) with metadata."""
    samples_list = []
    for img_id, item in models_app.test_samples.items():
        samples_list.append(
            {
                "image_id": img_id,
                "dx": item["dx"],
                "lesion_id": item["lesion_id"],
                "name": item["clinical_info"].get("name", item["dx"]),
                "nature": item["clinical_info"].get("nature", "Unknown"),
                "risk": item["clinical_info"].get("risk", "Benign"),
                "badge_color": item["clinical_info"].get("badge_color", "slate"),
                "image_url": f"/api/sample-image/{img_id}",
            }
        )
    return {"samples": samples_list}


@app.get("/api/sample-image/{image_id}")
async def get_sample_image(image_id: str):
    """Serves the raw dermoscopic image file for a curated sample."""
    if image_id not in models_app.test_samples:
        raise HTTPException(status_code=404, detail="Sample image not found")
    filepath = models_app.test_samples[image_id]["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404, detail="Image file missing from disk"
        )
    return FileResponse(filepath, media_type="image/jpeg")


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Returns the full 6-model benchmark leaderboard, MoE routing breakdown, and FedAvg telemetry."""
    return models_app.benchmarks_payload


@app.post("/api/predict")
async def predict_lesion(
    file: Optional[UploadFile] = File(None),
    image_id: Optional[str] = Form(None),
):
    """
    Accepts an uploaded image file OR a curated image_id from test set.
    Performs live 2048-D feature extraction, dynamic 88-D MoE routing, and Federated inference.
    """
    ground_truth = None

    if image_id and image_id in models_app.test_samples:
        sample_meta = models_app.test_samples[image_id]
        filepath = sample_meta["filepath"]
        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=404, detail=f"File for {image_id} not found on disk"
            )
        img = Image.open(filepath)
        ground_truth = {
            "image_id": image_id,
            "code": sample_meta["dx"],
            "name": sample_meta["clinical_info"].get(
                "name", sample_meta["dx"]
            ),
            "risk": sample_meta["clinical_info"].get("risk", "Unknown"),
            "lesion_id": sample_meta["lesion_id"],
        }
    elif file:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400, detail="Uploaded file is empty"
            )
        try:
            img = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image format: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either an uploaded image file or a valid sample image_id",
        )

    return process_and_infer(img, ground_truth)


# ── Serve Frontend ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            "<h1>HAM10000 API Online</h1><p>Frontend template initializing...</p>",
            status_code=200,
        )
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# Mount static folder
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
