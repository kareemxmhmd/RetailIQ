"""
Drift monitoring functions for RetailIQ.
Computes Population Stability Index (PSI) and generates reports.
"""
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) for two arrays.

    Args:
        expected (np.ndarray): The reference data array.
        actual (np.ndarray): The current/actual data array.
        buckets (int): Number of bins to use for the distribution.

    Returns:
        float: The PSI value.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    expected_min = np.min(expected)
    expected_max = np.max(expected)
    
    if expected_min == expected_max:
        # Avoid division by zero in binning if all values are identical
        bins = np.linspace(expected_min - 1e-6, expected_max + 1e-6, buckets + 1)
    else:
        bins = np.linspace(expected_min, expected_max, buckets + 1)
        
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    expected_prop = expected_counts / len(expected)
    actual_prop = actual_counts / len(actual)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    expected_prop = np.maximum(expected_prop, epsilon)
    actual_prop = np.maximum(actual_prop, epsilon)
    
    psi_values = (actual_prop - expected_prop) * np.log(actual_prop / expected_prop)
    return float(np.sum(psi_values))

def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str], config: dict) -> dict:
    """
    Check for feature drift between reference and current dataframes.

    Args:
        reference_df (pd.DataFrame): The reference (training) dataframe.
        current_df (pd.DataFrame): The current (production/new) dataframe.
        features (list[str]): List of features to check for drift.
        config (dict): Configuration dictionary containing monitoring settings.

    Returns:
        dict: A dictionary containing drift results per feature and overall flag.
    """
    psi_buckets = config.get('monitoring', {}).get('psi_buckets', 10)
    psi_threshold = config.get('monitoring', {}).get('psi_threshold', 0.2)
    
    results = {'features': {}, 'any_drift': False}
    
    for feat in features:
        if feat in reference_df.columns and feat in current_df.columns:
            expected = reference_df[feat].dropna().values
            actual = current_df[feat].dropna().values
            
            psi_val = compute_psi(expected, actual, buckets=psi_buckets)
            drifted = psi_val > psi_threshold
            
            results['features'][feat] = {
                'psi': psi_val,
                'drifted': drifted
            }
            if drifted:
                results['any_drift'] = True
        else:
            logger.warning(f"Feature '{feat}' missing in one or both dataframes.")
            
    return results

def generate_drift_report(drift_results: dict, config: dict) -> None:
    """
    Generate a human-readable drift report and save it to disk.

    Args:
        drift_results (dict): The dictionary containing drift results.
        config (dict): Configuration dictionary containing paths.
    """
    reports_dir = config['paths']['reports_dir']
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'drift_report.txt')
    
    lines = []
    lines.append("=== Data Drift Report ===")
    lines.append(f"Overall Drift Detected: {'Yes' if drift_results['any_drift'] else 'No'}")
    lines.append("\nFeature-level PSI:")
    
    for feat, stats in drift_results.get('features', {}).items():
        drift_str = "DRIFTED" if stats['drifted'] else "STABLE"
        lines.append(f"  - {feat}: PSI = {stats['psi']:.4f} ({drift_str})")
        
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"Drift report generated and saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to write drift report: {e}")
        raise
