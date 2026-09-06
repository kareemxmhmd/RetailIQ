import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from datetime import timedelta
import pandas as pd
from src.data.data_loader import load_config, load_raw_data, clean_data

logger = logging.getLogger(__name__)

def compute_rfm(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Computes Recency, Frequency, and Monetary features per customer."""
    date_col = config['data']['date_column']
    customer_col = config['data']['customer_id_col']
    invoice_col = config['data']['invoice_col']
    
    ref_date_str = config['rfm']['reference_date']
    if ref_date_str == 'auto':
        reference_date = df[date_col].max() + timedelta(days=1)
    else:
        reference_date = pd.to_datetime(ref_date_str)
        
    rfm_df = df.groupby(customer_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,
        invoice_col: 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={
        date_col: 'Recency',
        invoice_col: 'Frequency',
        'TotalPrice': 'Monetary'
    })
    
    return rfm_df

def compute_extra_features(df: pd.DataFrame, rfm_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Computes additional customer features and merges them onto the RFM DataFrame."""
    customer_col = config['data']['customer_id_col']
    invoice_col = config['data']['invoice_col']
    qty_col = config['data']['quantity_col']
    date_col = config['data']['date_column']
    stock_col = config['data']['stock_code_col']
    country_col = config['data']['country_col']
    
    # AvgBasketSize and AvgBasketValue
    invoice_stats = df.groupby([customer_col, invoice_col]).agg({
        qty_col: 'mean',
        'TotalPrice': 'sum'
    }).groupby(customer_col).mean().rename(columns={
        qty_col: 'AvgBasketSize',
        'TotalPrice': 'AvgBasketValue'
    })
    
    # Tenure
    tenure = df.groupby(customer_col).agg({
        date_col: lambda x: (x.max() - x.min()).days
    }).rename(columns={date_col: 'Tenure'})
    
    # UniqueProducts
    unique_products = df.groupby(customer_col).agg({
        stock_col: 'nunique'
    }).rename(columns={stock_col: 'UniqueProducts'})
    
    # Country (mode)
    country = df.groupby(customer_col)[country_col].apply(lambda x: x.mode()[0] if not x.empty else None).to_frame(name='Country')
    
    extra_features = pd.concat([invoice_stats, tenure, unique_products, country], axis=1)
    final_df = rfm_df.merge(extra_features, left_index=True, right_index=True, how='left')
    
    return final_df

def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Orchestrator to compute RFM and extra features."""
    logger.info("Computing RFM features...")
    rfm_df = compute_rfm(df, config)
    logger.info("Computing extra features...")
    final_df = compute_extra_features(df, rfm_df, config)
    logger.info(f"Final features shape: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    raw_df = load_raw_data(cfg)
    clean_df = clean_data(raw_df, cfg)
    features_df = build_features(clean_df, cfg)
    print(features_df.head())
