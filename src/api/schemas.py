from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    """Request schema for segment prediction."""
    Recency: float = Field(..., ge=0, description="Days since last purchase")
    Frequency: float = Field(..., ge=1, description="Number of unique invoices")
    Monetary: float = Field(..., gt=0, description="Total spend amount")

class PredictResponse(BaseModel):
    """Response schema for segment prediction."""
    cluster: int = Field(..., description="Numeric cluster ID")
    segment: str = Field(..., description="Business segment name")
    description: str = Field(..., description="Segment description")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
