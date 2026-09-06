import os
import logging
import yaml
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def load_config(path: str = 'config/config.yaml') -> dict:
    """Loads the YAML configuration file and resolves paths to absolute paths."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure all paths are absolute relative to project root
    if 'paths' in config:
        for key, rel_path in config['paths'].items():
            if not os.path.isabs(rel_path):
                config['paths'][key] = str((project_root / rel_path).resolve())

    return config

def load_raw_data(config: dict) -> pd.DataFrame:
    """Reads the raw CSV file and parses dates."""
    data_path = Path(config['paths']['raw_data'])
    encoding = config['data']['encoding']
    date_col = config['data']['date_column']
    
    logger.info(f"Loading raw data from {data_path}")
    df = pd.read_csv(data_path, encoding=encoding)
    df[date_col] = pd.to_datetime(df[date_col])
    logger.info(f"Raw data shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cleans the raw DataFrame based on config settings."""
    initial_shape = df.shape
    logger.info(f"Initial shape before cleaning: {initial_shape}")
    
    c_config = config['cleaning']
    d_config = config['data']
    
    if c_config.get('drop_missing_customer', False):
        df = df.dropna(subset=[d_config['customer_id_col']])
        
    if c_config.get('drop_cancelled', False):
        df = df[~df[d_config['invoice_col']].astype(str).str.startswith('C')]
        
    if c_config.get('drop_negative_qty', False):
        df = df[(df[d_config['quantity_col']] > 0) & (df[d_config['price_col']] > 0)]
        
    if c_config.get('deduplicate', False):
        df = df.drop_duplicates()
        
    df[d_config['customer_id_col']] = df[d_config['customer_id_col']].astype(int)
    df[d_config['date_column']] = pd.to_datetime(df[d_config['date_column']])
    
    df['TotalPrice'] = df[d_config['quantity_col']] * df[d_config['price_col']]
    
    final_shape = df.shape
    removed_pct = 100 * (initial_shape[0] - final_shape[0]) / initial_shape[0] if initial_shape[0] > 0 else 0
    logger.info(f"Final shape: {final_shape}. Removed {removed_pct:.2f}% of rows.")
    
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    raw_df = load_raw_data(cfg)
    clean_df = clean_data(raw_df, cfg)
    print(clean_df.head())
