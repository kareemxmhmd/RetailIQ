"""Tests for feature engineering module."""

import pandas as pd
from typing import Dict, Any
from src.features.build_features import compute_rfm, build_features

def test_compute_rfm_shape(sample_clean_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that compute_rfm returns a DataFrame with expected shape and index."""
    rfm_df = compute_rfm(sample_clean_df, sample_config)
    assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm_df.columns)
    assert len(rfm_df) == sample_clean_df[sample_config["data"]["customer_id_col"]].nunique()
    assert rfm_df.index.name == sample_config["data"]["customer_id_col"]

def test_compute_rfm_recency_positive(sample_clean_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that Recency is non-negative."""
    rfm_df = compute_rfm(sample_clean_df, sample_config)
    assert (rfm_df["Recency"] >= 0).all()

def test_compute_rfm_frequency_positive(sample_clean_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that Frequency is at least 1."""
    rfm_df = compute_rfm(sample_clean_df, sample_config)
    assert (rfm_df["Frequency"] >= 1).all()

def test_compute_rfm_monetary_positive(sample_clean_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that Monetary is strictly positive."""
    rfm_df = compute_rfm(sample_clean_df, sample_config)
    assert (rfm_df["Monetary"] > 0).all()

def test_build_features_extra_columns(sample_clean_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that build_features adds extra columns."""
    features_df = build_features(sample_clean_df, sample_config)
    expected_cols = {"AvgBasketSize", "AvgBasketValue", "Tenure", "UniqueProducts", "Country"}
    assert expected_cols.issubset(features_df.columns)
