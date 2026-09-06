import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from src.data.data_loader import load_config
from src.models.inference import load_artifacts, predict_segment
from src.monitoring.audit_logger import log_prediction
from src.api.schemas import PredictRequest, PredictResponse, HealthResponse

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for loading config and ML artifacts on startup."""
    logger.info("Loading config and ML artifacts...")
    try:
        config = load_config()
        app.state.config = config
        app.state.artifacts = load_artifacts(config)
        logger.info("Startup complete. Artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load artifacts during startup: {e}")
        raise
    yield
    logger.info("Shutting down API...")

app = FastAPI(title="RetailIQ API", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check API health."""
    return HealthResponse(status="healthy")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict customer segment from RFM features."""
    try:
        rfm_input = {
            'Recency': req.Recency,
            'Frequency': req.Frequency,
            'Monetary': req.Monetary
        }
        
        # Make prediction
        result = predict_segment(rfm_input, app.state.artifacts)
        cluster = result['cluster']
        segment = result['segment']
        
        # Get description
        descriptions = app.state.config.get("segment_labels", {}).get("descriptions", {})
        description = descriptions.get(segment, "Unknown segment")
        
        # Log prediction
        log_prediction(rfm_input, {"cluster": cluster, "segment": segment}, app.state.config)
        
        return PredictResponse(
            cluster=cluster,
            segment=segment,
            description=description
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
