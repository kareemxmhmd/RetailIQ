"""Tests for model training, preprocessing, and cluster evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.models.train_model import (
    scale_features,
    find_optimal_k,
    train_kmeans,
    label_segments,
    evaluate_cluster_stability
)

def test_scale_features_log_transform(sample_rfm_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that scale_features applies log1p and returns standard scaled data."""
    X_scaled, scaler, feature_cols, prep_cfg = scale_features(sample_rfm_df, sample_config)
    
    assert X_scaled.shape == (len(sample_rfm_df), 3)
    assert prep_cfg["log_transform"] is True
    assert isinstance(scaler, StandardScaler)
    assert feature_cols == ["Recency", "Frequency", "Monetary"]


def test_scale_features_clipping(sample_rfm_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that extreme outliers are clipped when configured."""
    df = sample_rfm_df.copy()
    # Add an extreme outlier
    df.loc[99999] = [10.0, 500.0, 999999.0]
    
    config = dict(sample_config)
    config["preprocessing"]["outlier_clip_percentile"] = 0.5  # aggressively clip at median for test
    
    X_scaled, scaler, feature_cols, prep_cfg = scale_features(df, config)
    assert "Monetary" in prep_cfg["clip_thresholds"]
    assert prep_cfg["clip_thresholds"]["Monetary"] < 999999.0


def test_find_optimal_k(sample_rfm_df: pd.DataFrame, sample_config: Dict[str, Any], tmp_path) -> None:
    """Test find_optimal_k returns valid optimal_k and comprehensive metrics dict."""
    config = dict(sample_config)
    config["paths"]["reports_dir"] = str(tmp_path)
    config["model"]["k_range"] = [2, 3]
    
    # Create a synthetic dataset with at least 10 points
    synthetic_data = np.random.RandomState(42).randn(15, 3)
    
    optimal_k, metrics = find_optimal_k(synthetic_data, config)
    assert optimal_k in [2, 3]
    assert "inertias" in metrics
    assert "silhouettes" in metrics
    assert "calinski_scores" in metrics
    assert "davies_scores" in metrics


def test_label_segments_all_8_personas(sample_config: Dict[str, Any]) -> None:
    """Test label_segments assigns all 8 personas uniquely when k=8."""
    # Create synthetic cluster data representing 8 clusters
    np.random.seed(42)
    clusters = []
    # Generate 8 distinct clusters with varied RFM
    means = [
        (10, 30, 20000),  # VIP
        (15, 10, 3500),   # Loyal
        (40, 5, 1200),    # Regular
        (60, 8, 3000),    # At-Risk
        (10, 2, 400),     # New
        (150, 2, 800),    # Dormant
        (50, 1, 250),     # Occasional
        (250, 1, 200)     # Churned
    ]
    dfs = []
    for cid, (r, f, m) in enumerate(means):
        cluster_df = pd.DataFrame({
            "Recency": np.maximum(1, np.random.normal(r, 2, 10)),
            "Frequency": np.maximum(1, np.random.normal(f, 0.5, 10)),
            "Monetary": np.maximum(10, np.random.normal(m, 50, 10)),
            "Cluster": cid
        })
        dfs.append(cluster_df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Mock KMeans
    class MockKMeans:
        n_clusters = 8
        labels_ = full_df["Cluster"].values
        
    mock_model = MockKMeans()
    mock_scaler = StandardScaler()
    
    labeled_df, segment_map = label_segments(
        full_df, mock_model, mock_scaler, ["Recency", "Frequency", "Monetary"], sample_config
    )
    
    # Verify all 8 unique segment names are mapped
    expected_segments = set(sample_config["segment_labels"]["names"])
    assert set(segment_map.values()) == expected_segments
    assert len(set(segment_map.keys())) == 8
    assert set(labeled_df["Segment"].unique()) == expected_segments


def test_evaluate_cluster_stability(sample_config: Dict[str, Any]) -> None:
    """Test that evaluate_cluster_stability returns valid ARI metrics."""
    synthetic_X = np.random.RandomState(42).randn(50, 3)
    config = dict(sample_config)
    config["model"]["stability_bootstrap_samples"] = 2
    
    stability = evaluate_cluster_stability(synthetic_X, k=3, config=config)
    assert "mean_bootstrap_ari" in stability
    assert "mean_seed_ari" in stability
    assert -1.0 <= stability["mean_bootstrap_ari"] <= 1.0
