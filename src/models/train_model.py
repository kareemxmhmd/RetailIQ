import pandas as pd
import numpy as np
import logging
import os
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, Any, List

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data_loader import load_config
from src.data.data_loader import load_raw_data, clean_data
from src.features.build_features import build_features

import warnings

# Suppress loky cpu count warning on Windows and pre-cache cores
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
try:
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx.physical_cores_cache = os.cpu_count() or 4
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scale_features(features_df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Extracts RFM columns, fits a StandardScaler, and transforms the data.
    """
    logger.info("Scaling features...")
    feature_cols = ['Recency', 'Frequency', 'Monetary']
    X = features_df[feature_cols].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, feature_cols

def find_optimal_k(X_scaled: np.ndarray, config: dict) -> Tuple[int, Dict[str, List[float]]]:
    """
    Iterates over a range of k to find the optimal number of clusters based on silhouette score.
    Saves elbow and silhouette plots.
    """
    logger.info("Finding optimal k...")
    k_range = config['model']['k_range']
    random_state = config['model']['random_state']
    reports_dir = config['paths']['reports_dir']
    os.makedirs(reports_dir, exist_ok=True)
    
    k_values = list(range(k_range[0], k_range[1] + 1))
    inertias = []
    silhouettes = []
    
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouettes.append(float(silhouette_score(X_scaled, labels)))
        
    optimal_k = k_values[np.argmax(silhouettes)]
    logger.info(f"Optimal k found: {optimal_k}")
    
    # Plotting Elbow
    plt.figure()
    plt.plot(k_values, inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(reports_dir, 'elbow_plot.png'))
    plt.close()
    
    # Plotting Silhouette
    plt.figure()
    plt.plot(k_values, silhouettes, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join(reports_dir, 'silhouette_plot.png'))
    plt.close()
    
    metrics = {
        'inertias': inertias,
        'silhouettes': silhouettes,
        'k_values': k_values
    }
    
    return optimal_k, metrics

def train_kmeans(X_scaled: np.ndarray, k: int, config: dict) -> KMeans:
    """
    Trains the KMeans model with the optimal k.
    """
    logger.info(f"Training KMeans with k={k}...")
    random_state = config['model']['random_state']
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model.fit(X_scaled)
    return model

def label_segments(features_df: pd.DataFrame, model: KMeans, scaler: StandardScaler, feature_cols: List[str], config: dict) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Assigns cluster labels to the dataframe, sorts them by average monetary value,
    and maps them to segment names from the config.
    """
    logger.info("Labeling segments...")
    features_df = features_df.copy()
    features_df['Cluster'] = model.labels_
    
    # Compute mean Monetary per cluster
    cluster_monetary = features_df.groupby('Cluster')['Monetary'].mean().reset_index()
    # Sort clusters by mean Monetary descending
    cluster_monetary = cluster_monetary.sort_values(by='Monetary', ascending=False)
    sorted_clusters = cluster_monetary['Cluster'].tolist()
    
    # Map rank index to segment names
    segment_names = config['segment_labels']['names']
    
    segment_map = {}
    for rank, cluster_id in enumerate(sorted_clusters):
        if rank < len(segment_names):
            segment_map[cluster_id] = segment_names[rank]
        else:
            segment_map[cluster_id] = f"Segment_{rank}"
            
    features_df['Segment'] = features_df['Cluster'].map(segment_map)
    
    return features_df, segment_map

def save_artifacts(model: KMeans, scaler: StandardScaler, feature_cols: List[str], segment_map: Dict[int, str], config: dict) -> None:
    """
    Saves model, scaler, segment mapping, and feature columns to the artifacts directory.
    """
    logger.info("Saving artifacts...")
    artifacts_dir = config['paths']['artifacts_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_path = os.path.join(artifacts_dir, 'model.pkl')
    joblib.dump(model, model_path)
    
    scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    segment_map_path = os.path.join(artifacts_dir, 'segment_map.json')
    with open(segment_map_path, 'w') as f:
        json.dump(segment_map, f, indent=4)
        
    feature_cols_path = os.path.join(artifacts_dir, 'feature_columns.json')
    with open(feature_cols_path, 'w') as f:
        json.dump(feature_cols, f, indent=4)
        
    logger.info(f"Artifacts saved to {artifacts_dir}")

def generate_evaluation_report(features_df: pd.DataFrame, X_scaled: np.ndarray, model: KMeans, config: dict) -> None:
    """
    Generates an evaluation report with silhouette score, cluster sizes, and cluster center stats.
    """
    logger.info("Generating evaluation report...")
    reports_dir = config['paths']['reports_dir']
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'evaluation_report.txt')
    
    sil_score = silhouette_score(X_scaled, model.labels_)
    cluster_sizes = features_df['Segment'].value_counts()
    cluster_stats = features_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    
    with open(report_path, 'w') as f:
        f.write("Evaluation Report\n")
        f.write("=================\n\n")
        f.write(f"Silhouette Score: {sil_score:.4f}\n\n")
        f.write("Cluster Sizes:\n")
        f.write(f"{cluster_sizes.to_string()}\n\n")
        f.write("Cluster Stats (Means):\n")
        f.write(f"{cluster_stats.to_string()}\n")
        
    logger.info(f"Evaluation report saved to {report_path}")

def run_training(features_df: pd.DataFrame, config: dict) -> None:
    """
    Orchestrates the entire training pipeline.
    """
    logger.info("Starting training pipeline...")
    X_scaled, scaler, feature_cols = scale_features(features_df, config)
    optimal_k, metrics = find_optimal_k(X_scaled, config)
    model = train_kmeans(X_scaled, optimal_k, config)
    features_df_labeled, segment_map = label_segments(features_df, model, scaler, feature_cols, config)
    save_artifacts(model, scaler, feature_cols, segment_map, config)
    generate_evaluation_report(features_df_labeled, X_scaled, model, config)
    logger.info("Training pipeline completed successfully.")

if __name__ == '__main__':
    try:
        config = load_config()
        df = load_raw_data(config)
        df_clean = clean_data(df, config)
        features_df = build_features(df_clean, config)
        run_training(features_df, config)
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
