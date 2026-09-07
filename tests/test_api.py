"""Tests for FastAPI endpoints, request validation, and server robustness."""

import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

from src.api.server import app
from src.monitoring.audit_logger import log_prediction, read_audit_log

@pytest.fixture
def mock_app_state():
    """Configure app.state with mock artifacts and config."""
    X = np.array([[10, 5, 100], [50, 1, 20], [2, 10, 500]], dtype=float)
    X_trans = np.log1p(X)
    scaler = StandardScaler().fit(X_trans)
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.transform(X_trans))
    
    app.state.artifacts = {
        "model": model,
        "scaler": scaler,
        "segment_map": {0: "VIP", 1: "Regular", 2: "Churned"},
        "feature_columns": ["Recency", "Frequency", "Monetary"],
        "preprocessor_config": {"log_transform": True, "clip_thresholds": {}}
    }
    app.state.config = {
        "paths": {"audit_log": "test_reports/audit_log.jsonl"},
        "segment_labels": {
            "descriptions": {
                "VIP": "VIP description",
                "Regular": "Regular description",
                "Churned": "Churned description"
            }
        }
    }


def test_health_endpoint():
    """Test health check returns status 200 and healthy."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_valid(mock_app_state, tmp_path):
    """Test predict endpoint with valid payload."""
    app.state.config["paths"]["audit_log"] = str(tmp_path / "audit.jsonl")
    client = TestClient(app, raise_server_exceptions=False)
    
    payload = {
        "Recency": 10.0,
        "Frequency": 5.0,
        "Monetary": 250.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "cluster" in data
    assert "segment" in data
    assert "description" in data
    assert data["segment"] in ["VIP", "Regular", "Churned"]


def test_predict_endpoint_out_of_bounds_frequency():
    """Test predict endpoint rejects Frequency exceeding upper boundary (10000)."""
    client = TestClient(app, raise_server_exceptions=False)
    payload = {
        "Recency": 10.0,
        "Frequency": 50000.0,  # Exceeds le=10000
        "Monetary": 250.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_out_of_bounds_recency():
    """Test predict endpoint rejects negative Recency."""
    client = TestClient(app, raise_server_exceptions=False)
    payload = {
        "Recency": -5.0,  # Below ge=0
        "Frequency": 5.0,
        "Monetary": 250.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_audit_logger_thread_safe(tmp_path):
    """Test concurrent predictions logging to file without corruption."""
    import concurrent.futures
    log_file = tmp_path / "concurrent_audit.jsonl"
    config = {"paths": {"audit_log": str(log_file)}}
    
    def worker(i):
        log_prediction({"Recency": float(i)}, {"cluster": i, "segment": "VIP"}, config)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(worker, range(20)))
        
    records = read_audit_log(config)
    assert len(records) == 20
