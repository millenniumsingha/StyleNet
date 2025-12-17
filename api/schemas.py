"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import List, Dict


class PredictionResult(BaseModel):
    """Single prediction result."""
    class_name: str = Field(..., description="Predicted class name")
    class_index: int = Field(..., description="Predicted class index (0-9)")
    confidence: float = Field(..., description="Confidence score (0-1)")


class PredictionResponse(BaseModel):
    """Full prediction response."""
    success: bool = Field(default=True)
    predicted_class: str = Field(..., description="Top predicted class")
    predicted_index: int = Field(..., description="Top predicted class index")
    confidence: float = Field(..., description="Confidence of top prediction")
    top_predictions: List[PredictionResult] = Field(..., description="Top 3 predictions")
    all_probabilities: Dict[str, float] = Field(..., description="All class probabilities")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    model_loaded: bool
    version: str


class ClassNamesResponse(BaseModel):
    """Class names response."""
    class_names: List[str]
    num_classes: int
