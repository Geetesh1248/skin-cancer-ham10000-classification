import io
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

def test_full_pipeline():
    print("Beginning automated API verification suite...")
    with TestClient(app) as client:
        # 1. Health
        r_health = client.get("/api/health")
        print(f"1. GET /api/health: {r_health.status_code}")
        assert r_health.status_code == 200, f"Expected 200, got {r_health.status_code}"
        health_data = r_health.json()
        assert health_data["status"] == "healthy"
        print(f"   Health check confirmed. Models loaded: {health_data['models_loaded']}")

        # 2. Benchmarks
        r_bench = client.get("/api/benchmarks")
        print(f"2. GET /api/benchmarks: {r_bench.status_code}")
        assert r_bench.status_code == 200
        bench_data = r_bench.json()
        assert "leaderboard" in bench_data
        assert "moe_routing_breakdown" in bench_data
        assert "federated_telemetry" in bench_data
        print(f"   Leaderboard contains {len(bench_data['leaderboard'])} benchmarked models.")
        print(f"   FedAvg Telemetry contains {len(bench_data['federated_telemetry'])} communication rounds.")

        # 3. Samples
        r_samples = client.get("/api/samples")
        print(f"3. GET /api/samples: {r_samples.status_code}")
        assert r_samples.status_code == 200
        samples = r_samples.json()["samples"]
        assert len(samples) == 7, f"Expected 7 test samples (1 per class), got {len(samples)}"
        print(f"   Found {len(samples)} verified test samples: {[s['dx'] for s in samples]}")

        # 4. Sample image stream
        first_id = samples[0]["image_id"]
        r_img = client.get(f"/api/sample-image/{first_id}")
        print(f"4. GET /api/sample-image/{first_id}: {r_img.status_code}, Content-Type: {r_img.headers.get('content-type')}")
        assert r_img.status_code == 200
        assert len(r_img.content) > 1000

        # 5. Predict with sample image_id
        r_pred1 = client.post("/api/predict", data={"image_id": first_id})
        print(f"5. POST /api/predict (via image_id): {r_pred1.status_code}")
        assert r_pred1.status_code == 200
        p1 = r_pred1.json()
        assert "federated_prediction" in p1
        assert "moe_prediction" in p1
        assert "gate_weights" in p1
        print(f"   Sample ID: {first_id} ({samples[0]['dx']})")
        print(f"   Federated Global Model: {p1['federated_prediction']['name']} ({p1['federated_prediction']['confidence']}%)")
        print(f"   MoE Gating Router: {p1['moe_prediction']['name']} ({p1['moe_prediction']['confidence']}%)")
        print(f"   Gate Weights: {p1['gate_weights']}")

        # 6. Predict with file upload
        file_bytes = r_img.content
        r_pred2 = client.post(
            "/api/predict",
            files={"file": ("test_lesion.jpg", io.BytesIO(file_bytes), "image/jpeg")},
        )
        print(f"6. POST /api/predict (via multipart upload): {r_pred2.status_code}")
        assert r_pred2.status_code == 200
        p2 = r_pred2.json()
        assert p2["status"] == "success"
        print(f"   Uploaded File Federated Pred: {p2['federated_prediction']['name']} ({p2['federated_prediction']['confidence']}%)")

        # 7. Frontend index
        r_index = client.get("/")
        print(f"7. GET /: {r_index.status_code}, HTML Length: {len(r_index.text)} bytes")
        assert r_index.status_code == 200
        assert "HAM10000 Skin Lesion Diagnosis" in r_index.text

    print("\n" + "=" * 60)
    print("ALL 7 AUTOMATED VERIFICATION CHECKS PASSED WITH 200 OK!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_pipeline()
