"""
Orchestration script for running the complete RetailIQ pipeline.
"""
import logging
import sys
import os
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

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    
    try:
        from src.data.data_loader import load_config, load_raw_data, clean_data
        from src.features.build_features import build_features
        from src.models.train_model import run_training
        
        config = load_config()
        
        logging.info("Step 1/5: Loading raw data...")
        raw_df = load_raw_data(config)
        
        logging.info("Step 2/5: Cleaning data...")
        clean_df = clean_data(raw_df, config)
        
        logging.info("Step 3/5: Building features...")
        rfm_df = build_features(clean_df, config)
        
        logging.info("Step 4/5: Training model...")
        model = run_training(rfm_df, config)
        
        artifacts_dir = config['paths']['artifacts_dir']
        logging.info(f"Pipeline complete! Artifacts saved to {artifacts_dir}")
        
    except Exception as e:
        logging.exception("Pipeline execution failed")
        sys.exit(1)
