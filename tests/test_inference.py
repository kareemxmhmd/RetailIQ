"""Tests for model inference module."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.models.inference import predict_segment


def _make_mock_artifacts() -> dict:
    """Create mock artifacts for testing."""
    X = np.array([[10, 5, 100], [50, 1, 20], [2, 10, 500]], dtype=float)
    scaler = StandardScaler().fit(X)
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.transform(X))
    segment_map = {0: "VIP", 1: "Regular", 2: "Churned"}
    feature_columns = ["Recency", "Frequency", "Monetary"]
    return {
        "model": model,
        "scaler": scaler,
        "segment_map": segment_map,
        "feature_columns": feature_columns,
    }


def test_predict_segment_returns_valid_keys() -> None:
    """Test that predict_segment returns a dictionary with the correct keys."""
    artifacts = _make_mock_artifacts()
    sample_input = {"Recency": 5.0, "Frequency": 2.0, "Monetary": 150.0}

    result = predict_segment(sample_input, artifacts)

    assert {"cluster", "segment", "input"}.issubset(result.keys())


def test_predict_segment_returns_valid_segment() -> None:
    """Test that the predicted segment is one of the mapped values."""
    artifacts = _make_mock_artifacts()
    sample_input = {"Recency": 5.0, "Frequency": 2.0, "Monetary": 150.0}

    result = predict_segment(sample_input, artifacts)

    assert result["segment"] in artifacts["segment_map"].values()


def test_predict_segment_cluster_type() -> None:
    """Test that the predicted cluster is of integer type."""
    artifacts = _make_mock_artifacts()
    sample_input = {"Recency": 5.0, "Frequency": 2.0, "Monetary": 150.0}

    result = predict_segment(sample_input, artifacts)

    assert isinstance(result["cluster"], (int, np.integer))
