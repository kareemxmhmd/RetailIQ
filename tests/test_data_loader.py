"""Tests for data loading and cleaning module."""

import pandas as pd
from typing import Dict, Any
from src.data.data_loader import clean_data

def test_clean_data_removes_cancelled(sample_raw_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that cancelled invoices are removed."""
    cleaned = clean_data(sample_raw_df, sample_config)
    invoice_col = sample_config["data"]["invoice_col"]
    assert not cleaned[invoice_col].astype(str).str.startswith("C").any()

def test_clean_data_removes_negative_qty(sample_raw_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that rows with negative quantity are removed."""
    cleaned = clean_data(sample_raw_df, sample_config)
    qty_col = sample_config["data"]["quantity_col"]
    assert (cleaned[qty_col] > 0).all()

def test_clean_data_removes_missing_customer(sample_raw_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that rows with missing CustomerID are removed."""
    cleaned = clean_data(sample_raw_df, sample_config)
    customer_col = sample_config["data"]["customer_id_col"]
    assert cleaned[customer_col].notna().all()

def test_clean_data_deduplicates(sample_raw_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that duplicate rows are removed."""
    cleaned = clean_data(sample_raw_df, sample_config)
    assert not cleaned.duplicated().any()

def test_clean_data_correct_dtypes(sample_raw_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that output DataFrame has correct data types and TotalPrice column."""
    cleaned = clean_data(sample_raw_df, sample_config)
    customer_col = sample_config["data"]["customer_id_col"]
    date_col = sample_config["data"]["date_column"]
    
    assert pd.api.types.is_integer_dtype(cleaned[customer_col])
    assert pd.api.types.is_datetime64_any_dtype(cleaned[date_col])
    assert "TotalPrice" in cleaned.columns

def test_clean_data_total_price(sample_raw_df: pd.DataFrame, sample_config: Dict[str, Any]) -> None:
    """Test that TotalPrice is computed correctly as Quantity * UnitPrice."""
    cleaned = clean_data(sample_raw_df, sample_config)
    qty_col = sample_config["data"]["quantity_col"]
    price_col = sample_config["data"]["price_col"]
    
    expected = cleaned[qty_col] * cleaned[price_col]
    assert (cleaned["TotalPrice"] == expected).all()
