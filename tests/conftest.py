"""Shared pytest fixtures for RetailIQ tests."""

import pytest
import pandas as pd
from typing import Dict, Any
from src.data.data_loader import clean_data

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for tests."""
    return {
        "paths": {
            "raw_data": "dummy_path.csv",
            "artifacts_dir": "test_artifacts",
            "reports_dir": "test_reports",
            "audit_log": "test_reports/audit_log.jsonl"
        },
        "data": {
            "encoding": "ISO-8859-1",
            "date_column": "InvoiceDate",
            "customer_id_col": "CustomerID",
            "invoice_col": "InvoiceNo",
            "quantity_col": "Quantity",
            "price_col": "UnitPrice",
            "country_col": "Country",
            "description_col": "Description",
            "stock_code_col": "StockCode"
        },
        "cleaning": {
            "drop_cancelled": True,
            "drop_negative_qty": True,
            "drop_missing_customer": True,
            "deduplicate": True
        },
        "rfm": {
            "reference_date": "auto"
        },
        "preprocessing": {
            "log_transform": True,
            "outlier_clip_percentile": 0.999
        },
        "features": {
            "model_features": ["Recency", "Frequency", "Monetary"],
            "extra_features": ["AvgBasketSize", "AvgBasketValue", "Tenure", "UniqueProducts", "Country"]
        },
        "model": {
            "algorithm": "kmeans",
            "k": 8,
            "k_range": [2, 10],
            "random_state": 42,
            "n_init": 10,
            "stability_bootstrap_samples": 3
        },
        "segment_labels": {
            "names": ["VIP", "Loyal", "Regular", "At-Risk", "Dormant", "New", "Occasional", "Churned"],
            "descriptions": {}
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000
        },
        "ui": {
            "mode": "direct",
            "api_url": "http://localhost:8000"
        },
        "monitoring": {
            "psi_buckets": 10,
            "psi_threshold": 0.2
        }
    }

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Provide a sample raw DataFrame with mixed validity."""
    data = [
        # Valid rows
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "Quantity": 6, "InvoiceDate": pd.Timestamp("2010-12-01 08:26:00"), "UnitPrice": 2.55, "CustomerID": 17850.0, "Country": "United Kingdom"},
        {"InvoiceNo": "536365", "StockCode": "71053", "Description": "WHITE METAL LANTERN", "Quantity": 6, "InvoiceDate": pd.Timestamp("2010-12-01 08:26:00"), "UnitPrice": 3.39, "CustomerID": 17850.0, "Country": "United Kingdom"},
        {"InvoiceNo": "536366", "StockCode": "22633", "Description": "HAND WARMER UNION JACK", "Quantity": 6, "InvoiceDate": pd.Timestamp("2010-12-01 08:28:00"), "UnitPrice": 1.85, "CustomerID": 17850.0, "Country": "United Kingdom"},
        {"InvoiceNo": "536367", "StockCode": "84879", "Description": "ASSORTED COLOUR BIRD ORNAMENT", "Quantity": 32, "InvoiceDate": pd.Timestamp("2010-12-01 08:34:00"), "UnitPrice": 1.69, "CustomerID": 13047.0, "Country": "United Kingdom"},
        {"InvoiceNo": "536370", "StockCode": "22728", "Description": "ALARM CLOCK BAKELITE PINK", "Quantity": 24, "InvoiceDate": pd.Timestamp("2010-12-01 08:45:00"), "UnitPrice": 3.75, "CustomerID": 12583.0, "Country": "France"},
        {"InvoiceNo": "536370", "StockCode": "22727", "Description": "ALARM CLOCK BAKELITE RED ", "Quantity": 24, "InvoiceDate": pd.Timestamp("2010-12-01 08:45:00"), "UnitPrice": 3.75, "CustomerID": 12583.0, "Country": "France"},
        # Cancelled
        {"InvoiceNo": "C536379", "StockCode": "D", "Description": "Discount", "Quantity": -1, "InvoiceDate": pd.Timestamp("2010-12-01 09:41:00"), "UnitPrice": 27.50, "CustomerID": 14527.0, "Country": "United Kingdom"},
        # Negative quantity
        {"InvoiceNo": "536380", "StockCode": "22728", "Description": "ALARM CLOCK BAKELITE PINK", "Quantity": -5, "InvoiceDate": pd.Timestamp("2010-12-01 09:42:00"), "UnitPrice": 3.75, "CustomerID": 13047.0, "Country": "United Kingdom"},
        # Missing CustomerID
        {"InvoiceNo": "536381", "StockCode": "22728", "Description": "ALARM CLOCK BAKELITE PINK", "Quantity": 1, "InvoiceDate": pd.Timestamp("2010-12-01 09:43:00"), "UnitPrice": 3.75, "CustomerID": float('nan'), "Country": "United Kingdom"},
        # Duplicate row
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "Quantity": 6, "InvoiceDate": pd.Timestamp("2010-12-01 08:26:00"), "UnitPrice": 2.55, "CustomerID": 17850.0, "Country": "United Kingdom"},
    ]
    return pd.DataFrame(data)

@pytest.fixture
def sample_clean_df(sample_raw_df: pd.DataFrame, sample_config: Dict[str, Any]) -> pd.DataFrame:
    """Provide a cleaned DataFrame using the sample configuration."""
    return clean_data(sample_raw_df, sample_config)

@pytest.fixture
def sample_rfm_df() -> pd.DataFrame:
    """Provide a mock RFM DataFrame."""
    data = {
        "CustomerID": [17850, 13047, 12583],
        "Recency": [1, 2, 5],
        "Frequency": [2, 1, 1],
        "Monetary": [46.86, 54.08, 180.0]
    }
    return pd.DataFrame(data).set_index("CustomerID")
