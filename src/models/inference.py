import os
import json
import joblib
import logging
import numpy as np
from typing import Dict, Any

import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_artifacts(config: dict) -> Dict[str, Any]:
    """
    Loads model, scaler, segment mapping, and feature columns from artifacts directory.
    """
    logger.info("Loading artifacts...")
    artifacts_dir = config['paths']['artifacts_dir']
    
    model_path = os.path.join(artifacts_dir, 'model.pkl')
    scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
    segment_map_path = os.path.join(artifacts_dir, 'segment_map.json')
    feature_cols_path = os.path.join(artifacts_dir, 'feature_columns.json')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(segment_map_path, 'r') as f:
        segment_map_str_keys = json.load(f)
    # Convert string keys back to int
    segment_map = {int(k): v for k, v in segment_map_str_keys.items()}
    
    with open(feature_cols_path, 'r') as f:
        feature_columns = json.load(f)
        
    return {
        'model': model,
        'scaler': scaler,
        'segment_map': segment_map,
        'feature_columns': feature_columns
    }

def predict_segment(rfm_input: Dict[str, float], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts the segment for a given RFM input.
    """
    feature_cols = artifacts['feature_columns']
    try:
        features = [rfm_input[col] for col in feature_cols]
    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}")
        
    import pandas as pd
    X = pd.DataFrame([features], columns=feature_cols)
    X_scaled = artifacts['scaler'].transform(X)
    
    cluster_id = int(artifacts['model'].predict(X_scaled)[0])
    segment_name = artifacts['segment_map'].get(cluster_id, f"Unknown_{cluster_id}")
    
    return {
        'cluster': cluster_id,
        'segment': segment_name,
        'input': rfm_input
    }

if __name__ == '__main__':
    try:
        config = load_config()
        artifacts = load_artifacts(config)
        
        sample_input = {
            'Recency': 10.0,
            'Frequency': 50.0,
            'Monetary': 1500.0
        }
        
        prediction = predict_segment(sample_input, artifacts)
        logger.info(f"Sample prediction result: {prediction}")
    except Exception as e:
        logger.error(f"Error in inference: {e}")
