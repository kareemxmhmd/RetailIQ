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
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)
from scipy.optimize import linear_sum_assignment
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

def scale_features(features_df: pd.DataFrame, config: dict) -> Tuple[np.ndarray, StandardScaler, List[str], Dict[str, Any]]:
    """
    Extracts RFM columns, applies outlier clipping and log transformation if configured,
    then fits a StandardScaler.
    """
    logger.info("Preprocessing and scaling features...")
    feature_cols = config.get('features', {}).get('model_features', ['Recency', 'Frequency', 'Monetary'])
    X = features_df[feature_cols].copy()
    
    preprocessing_cfg = config.get('preprocessing', {})
    clip_pct = preprocessing_cfg.get('outlier_clip_percentile', None)
    clip_thresholds: Dict[str, float] = {}
    
    if clip_pct is not None and 0 < clip_pct < 1.0:
        for col in ['Frequency', 'Monetary']:
            if col in X.columns:
                thresh = float(X[col].quantile(clip_pct))
                clip_thresholds[col] = thresh
                X[col] = X[col].clip(upper=thresh)
                logger.info(f"Clipped {col} at {clip_pct*100:.1f}th percentile: {thresh:.2f}")
                
    log_transform = preprocessing_cfg.get('log_transform', True)
    if log_transform:
        logger.info("Applying np.log1p transformation to RFM features.")
        X_trans = np.log1p(X)
    else:
        X_trans = X
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trans)
    
    preprocessor_config = {
        'log_transform': log_transform,
        'clip_thresholds': clip_thresholds,
        'feature_columns': feature_cols
    }
    
    return X_scaled, scaler, feature_cols, preprocessor_config

def find_optimal_k(X_scaled: np.ndarray, config: dict) -> Tuple[int, Dict[str, Any]]:
    """
    Evaluates a range of k using Silhouette score, Calinski-Harabasz score,
    Davies-Bouldin index, and Inertia. Generates diagnostic plots.
    """
    logger.info("Evaluating cluster quality across k values...")
    k_range = config['model']['k_range']
    random_state = config['model']['random_state']
    n_init = config['model'].get('n_init', 20)
    reports_dir = config['paths']['reports_dir']
    os.makedirs(reports_dir, exist_ok=True)
    
    k_values = list(range(k_range[0], k_range[1] + 1))
    inertias = []
    silhouettes = []
    calinski_scores = []
    davies_scores = []
    
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_scaled)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, labels)))
        calinski_scores.append(float(calinski_harabasz_score(X_scaled, labels)))
        davies_scores.append(float(davies_bouldin_score(X_scaled, labels)))
        
    optimal_k = k_values[np.argmax(silhouettes)]
    logger.info(f"Highest Silhouette k found: {optimal_k} (score={max(silhouettes):.4f})")
    
    # Save individual plots for backward compatibility
    plt.figure()
    plt.plot(k_values, inertias, marker='o')
    plt.title('Elbow Method (Inertia)')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.savefig(os.path.join(reports_dir, 'elbow_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.plot(k_values, silhouettes, marker='o', color='darkorange')
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join(reports_dir, 'silhouette_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Multi-panel diagnostic plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(k_values, inertias, marker='o', color='royalblue')
    axs[0, 0].set_title('Elbow Method (Inertia)')
    axs[0, 0].set_xlabel('k')
    axs[0, 0].set_ylabel('Inertia')
    
    axs[0, 1].plot(k_values, silhouettes, marker='s', color='darkorange')
    axs[0, 1].set_title('Silhouette Score (Higher is better)')
    axs[0, 1].set_xlabel('k')
    axs[0, 1].set_ylabel('Silhouette')
    
    axs[1, 0].plot(k_values, calinski_scores, marker='^', color='forestgreen')
    axs[1, 0].set_title('Calinski-Harabasz Score (Higher is better)')
    axs[1, 0].set_xlabel('k')
    axs[1, 0].set_ylabel('Score')
    
    axs[1, 1].plot(k_values, davies_scores, marker='d', color='crimson')
    axs[1, 1].set_title('Davies-Bouldin Index (Lower is better)')
    axs[1, 1].set_xlabel('k')
    axs[1, 1].set_ylabel('Index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'cluster_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    metrics = {
        'k_values': k_values,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'calinski_scores': calinski_scores,
        'davies_scores': davies_scores,
        'optimal_k_silhouette': optimal_k
    }
    
    return optimal_k, metrics

def evaluate_cluster_stability(X_scaled: np.ndarray, k: int, config: dict) -> Dict[str, float]:
    """
    Evaluates cluster stability using bootstrap resampling and random seed variations.
    Returns mean Adjusted Rand Index (ARI).
    """
    logger.info(f"Evaluating clustering stability for k={k}...")
    random_state = config['model']['random_state']
    n_init = config['model'].get('n_init', 20)
    n_samples = len(X_scaled)
    
    base_model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X_scaled)
    base_labels = base_model.labels_
    
    # Bootstrap stability (sub-sampling 80% without replacement)
    n_bootstrap = config['model'].get('stability_bootstrap_samples', 10)
    bootstrap_aris = []
    for i in range(n_bootstrap):
        np.random.seed(random_state + i)
        idx = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
        sub_model = KMeans(n_clusters=k, random_state=random_state + i, n_init=n_init).fit(X_scaled[idx])
        pred_full = sub_model.predict(X_scaled)
        ari = float(adjusted_rand_score(base_labels, pred_full))
        bootstrap_aris.append(ari)
        
    # Seed stability across 5 random seeds
    seed_aris = []
    for seed in [1, 7, 21, 42, 99]:
        if seed != random_state:
            seed_model = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X_scaled)
            seed_aris.append(float(adjusted_rand_score(base_labels, seed_model.labels_)))
            
    return {
        'mean_bootstrap_ari': float(np.mean(bootstrap_aris)) if bootstrap_aris else 0.0,
        'min_bootstrap_ari': float(np.min(bootstrap_aris)) if bootstrap_aris else 0.0,
        'mean_seed_ari': float(np.mean(seed_aris)) if seed_aris else 1.0
    }

def train_kmeans(X_scaled: np.ndarray, k: int, config: dict) -> KMeans:
    """
    Trains the KMeans model with the specified k.
    """
    logger.info(f"Training KMeans with k={k}...")
    random_state = config['model']['random_state']
    n_init = config['model'].get('n_init', 20)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    model.fit(X_scaled)
    return model

def label_segments(features_df: pd.DataFrame, model: KMeans, scaler: StandardScaler, feature_cols: List[str], config: dict) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Assigns cluster labels to the dataframe and maps them to segment personas.
    When k equals the 8 business personas, uses multi-dimensional RFM archetype matching.
    Otherwise falls back to rank ordering by Monetary.
    """
    logger.info("Labeling segments...")
    features_df = features_df.copy()
    features_df['Cluster'] = model.labels_
    
    segment_names = config['segment_labels']['names']
    k = model.n_clusters
    
    segment_map: Dict[int, str] = {}
    
    if k == len(segment_names) and k == 8:
        # Archetype matching based on normalized R, F, M centroids
        cluster_means = features_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).reset_index()
        
        r_min, r_max = cluster_means['Recency'].min(), cluster_means['Recency'].max()
        f_min, f_max = cluster_means['Frequency'].min(), cluster_means['Frequency'].max()
        m_min, m_max = cluster_means['Monetary'].min(), cluster_means['Monetary'].max()
        
        cluster_means['R_norm'] = (cluster_means['Recency'] - r_min) / (r_max - r_min + 1e-9)
        cluster_means['F_norm'] = (cluster_means['Frequency'] - f_min) / (f_max - f_min + 1e-9)
        cluster_means['M_norm'] = (cluster_means['Monetary'] - m_min) / (m_max - m_min + 1e-9)
        
        # Archetype targets reflecting business persona definitions
        archetypes = {
            'VIP':        {'R': 0.0,  'F': 1.0,  'M': 1.0},
            'Loyal':      {'R': 0.0,  'F': 0.3,  'M': 0.2},
            'At-Risk':    {'R': 0.2,  'F': 0.3,  'M': 0.25},
            'Regular':    {'R': 0.1,  'F': 0.12, 'M': 0.06},
            'New':        {'R': 0.0,  'F': 0.04, 'M': 0.02},
            'Dormant':    {'R': 0.55, 'F': 0.08, 'M': 0.05},
            'Occasional': {'R': 0.2,  'F': 0.0,  'M': 0.0},
            'Churned':    {'R': 1.0,  'F': 0.0,  'M': 0.0}
        }
        
        ordered_archetypes = [name for name in segment_names if name in archetypes]
        if len(ordered_archetypes) == k:
            cost_matrix = np.zeros((k, k))
            for i, row in cluster_means.iterrows():
                for j, name in enumerate(ordered_archetypes):
                    target = archetypes[name]
                    dist = np.sqrt(
                        (row['R_norm'] - target['R'])**2 +
                        (row['F_norm'] - target['F'])**2 +
                        (row['M_norm'] - target['M'])**2
                    )
                    cost_matrix[i, j] = dist
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                cid = int(cluster_means.loc[r, 'Cluster'])
                segment_map[cid] = ordered_archetypes[c]
        else:
            # Fallback
            cluster_monetary = features_df.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False).index.tolist()
            for rank, cid in enumerate(cluster_monetary):
                segment_map[cid] = segment_names[rank]
    else:
        # General rank ordering by Monetary
        cluster_monetary = features_df.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False).index.tolist()
        for rank, cid in enumerate(cluster_monetary):
            if rank < len(segment_names):
                segment_map[cid] = segment_names[rank]
            else:
                segment_map[cid] = f"Segment_{rank}"
                
    features_df['Segment'] = features_df['Cluster'].map(segment_map)
    return features_df, segment_map

def save_artifacts(model: KMeans, scaler: StandardScaler, feature_cols: List[str], segment_map: Dict[int, str], preprocessor_config: Dict[str, Any], config: dict) -> None:
    """
    Saves model, scaler, segment mapping, preprocessor config, and feature columns to artifacts.
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
        
    preprocessor_path = os.path.join(artifacts_dir, 'preprocessor_config.json')
    with open(preprocessor_path, 'w') as f:
        json.dump(preprocessor_config, f, indent=4)
        
    logger.info(f"Artifacts saved to {artifacts_dir}")

def generate_evaluation_report(
    features_df: pd.DataFrame,
    X_scaled: np.ndarray,
    model: KMeans,
    stability_metrics: Dict[str, float],
    config: dict
) -> None:
    """
    Generates a comprehensive evaluation report with multi-metric scores,
    stability analysis, cluster distributions, and methodological documentation.
    """
    logger.info("Generating evaluation report...")
    reports_dir = config['paths']['reports_dir']
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'evaluation_report.txt')
    
    sil_score = float(silhouette_score(X_scaled, model.labels_))
    ch_score = float(calinski_harabasz_score(X_scaled, model.labels_))
    db_score = float(davies_bouldin_score(X_scaled, model.labels_))
    
    cluster_sizes = features_df['Segment'].value_counts()
    
    rfm_means = features_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    rfm_medians = features_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].median()
    
    # Extra features profiling if present
    extra_cols = [c for c in ['AvgBasketSize', 'AvgBasketValue', 'Tenure', 'UniqueProducts'] if c in features_df.columns]
    extra_means = features_df.groupby('Segment')[extra_cols].mean() if extra_cols else None
    
    with open(report_path, 'w') as f:
        f.write("RetailIQ Customer Segmentation — Evaluation Report\n")
        f.write("==================================================\n\n")
        f.write("1. Clustering Performance Metrics:\n")
        f.write("----------------------------------\n")
        f.write(f"Number of Clusters (k):   {model.n_clusters}\n")
        f.write(f"Silhouette Score:         {sil_score:.4f}  (Range [-1, 1], higher is better)\n")
        f.write(f"Calinski-Harabasz Index:  {ch_score:.2f} (Higher indicates denser, better-separated clusters)\n")
        f.write(f"Davies-Bouldin Index:     {db_score:.4f}  (Lower indicates better clustering separation)\n\n")
        
        f.write("2. Cluster Stability Assessment:\n")
        f.write("--------------------------------\n")
        f.write(f"Mean Bootstrap ARI:       {stability_metrics.get('mean_bootstrap_ari', 0.0):.4f} (80% sub-sample resampling)\n")
        f.write(f"Min Bootstrap ARI:        {stability_metrics.get('min_bootstrap_ari', 0.0):.4f}\n")
        f.write(f"Mean Seed Stability ARI:  {stability_metrics.get('mean_seed_ari', 0.0):.4f} (across 5 random initializations)\n\n")
        
        f.write("3. Cluster Segment Sizes:\n")
        f.write("-------------------------\n")
        f.write(f"{cluster_sizes.to_string()}\n\n")
        
        f.write("4. Cluster Centroid Stats (Means):\n")
        f.write("----------------------------------\n")
        f.write(f"{rfm_means.round(2).to_string()}\n\n")
        
        f.write("5. Cluster Centroid Stats (Medians):\n")
        f.write("------------------------------------\n")
        f.write(f"{rfm_medians.round(2).to_string()}\n\n")
        
        if extra_means is not None:
            f.write("6. Behavioral & Transactional Profiles (Means):\n")
            f.write("-----------------------------------------------\n")
            f.write(f"{extra_means.round(2).to_string()}\n\n")
            
        f.write("7. Methodological Notes & Limitations:\n")
        f.write("---------------------------------------\n")
        f.write("- Preprocessing: Monetary and Frequency are log-transformed (np.log1p) and scaled with StandardScaler,\n")
        f.write("  preventing wholesale outliers from collapsing the space into 2 clusters.\n")
        f.write("- Outlier Handling: Extreme values above the 99.9th percentile are clipped prior to transformation.\n")
        f.write("- Taxonomy Alignment: Segments are mapped to business personas via multi-dimensional RFM archetype matching.\n")
        f.write("- Model Limitation: KMeans assumes spherical clusters in Euclidean space. Highly correlated features or\n")
        f.write("  non-convex geometries should be monitored using the Population Stability Index (PSI) drift monitor.\n")
        
    logger.info(f"Evaluation report saved to {report_path}")

def run_training(features_df: pd.DataFrame, config: dict) -> Tuple[KMeans, pd.DataFrame, Dict[str, Any]]:
    """
    Orchestrates the entire training pipeline.
    """
    logger.info("Starting training pipeline...")
    X_scaled, scaler, feature_cols, preprocessor_config = scale_features(features_df, config)
    
    # Run k diagnostics
    optimal_k, metrics = find_optimal_k(X_scaled, config)
    
    # Select target k
    target_k_cfg = config.get('model', {}).get('k', 8)
    if target_k_cfg == 'auto' or target_k_cfg is None:
        target_k = optimal_k
    else:
        target_k = int(target_k_cfg)
        
    model = train_kmeans(X_scaled, target_k, config)
    features_df_labeled, segment_map = label_segments(features_df, model, scaler, feature_cols, config)
    
    # Stability evaluation
    stability_metrics = evaluate_cluster_stability(X_scaled, target_k, config)
    
    # Save artifacts and report
    save_artifacts(model, scaler, feature_cols, segment_map, preprocessor_config, config)
    generate_evaluation_report(features_df_labeled, X_scaled, model, stability_metrics, config)
    
    logger.info("Training pipeline completed successfully.")
    return model, features_df_labeled, metrics

if __name__ == '__main__':
    try:
        config = load_config()
        df = load_raw_data(config)
        df_clean = clean_data(df, config)
        features_df = build_features(df_clean, config)
        run_training(features_df, config)
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise
