"""
Cancer Detection REST API.

FastAPI-based REST API for the cancer detection system.
Provides endpoints for predictions, model info, and health checks.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import logging
import time

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cancer import CancerPredictor, CancerDataLoader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="Cancer Detection API",
    description="""
    REST API for breast cancer detection using machine learning.
    
    This API provides endpoints for:
    - Making predictions on patient features
    - Getting model information
    - Health checks
    
    The model classifies breast tumors as **Malignant** or **Benign** based on
    30 morphological features extracted from cell nuclei images.
    """,
    version="1.0.0",
    contact={
        "name": "Hubert Domagała",
        "url": "https://github.com/hubertdomagalaa",
    },
    license_info={
        "name": "MIT",
    }
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global predictor (loaded on startup)
predictor: Optional[CancerPredictor] = None


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class PatientFeatures(BaseModel):
    """Request model for patient features."""
    
    mean_radius: float = Field(..., description="Mean radius of nuclei", example=17.99)
    mean_texture: float = Field(..., description="Mean texture", example=10.38)
    mean_perimeter: float = Field(..., description="Mean perimeter", example=122.8)
    mean_area: float = Field(..., description="Mean area", example=1001.0)
    mean_smoothness: float = Field(..., description="Mean smoothness", example=0.1184)
    mean_compactness: float = Field(..., description="Mean compactness", example=0.2776)
    mean_concavity: float = Field(..., description="Mean concavity", example=0.3001)
    mean_concave_points: float = Field(..., description="Mean concave points", example=0.1471)
    mean_symmetry: float = Field(..., description="Mean symmetry", example=0.2419)
    mean_fractal_dimension: float = Field(..., description="Mean fractal dimension", example=0.07871)
    
    radius_error: float = Field(..., description="Radius standard error", example=1.095)
    texture_error: float = Field(..., description="Texture standard error", example=0.9053)
    perimeter_error: float = Field(..., description="Perimeter standard error", example=8.589)
    area_error: float = Field(..., description="Area standard error", example=153.4)
    smoothness_error: float = Field(..., description="Smoothness standard error", example=0.006399)
    compactness_error: float = Field(..., description="Compactness standard error", example=0.04904)
    concavity_error: float = Field(..., description="Concavity standard error", example=0.05373)
    concave_points_error: float = Field(..., description="Concave points standard error", example=0.01587)
    symmetry_error: float = Field(..., description="Symmetry standard error", example=0.03003)
    fractal_dimension_error: float = Field(..., description="Fractal dimension standard error", example=0.006193)
    
    worst_radius: float = Field(..., description="Worst radius", example=25.38)
    worst_texture: float = Field(..., description="Worst texture", example=17.33)
    worst_perimeter: float = Field(..., description="Worst perimeter", example=184.6)
    worst_area: float = Field(..., description="Worst area", example=2019.0)
    worst_smoothness: float = Field(..., description="Worst smoothness", example=0.1622)
    worst_compactness: float = Field(..., description="Worst compactness", example=0.6656)
    worst_concavity: float = Field(..., description="Worst concavity", example=0.7119)
    worst_concave_points: float = Field(..., description="Worst concave points", example=0.2654)
    worst_symmetry: float = Field(..., description="Worst symmetry", example=0.4601)
    worst_fractal_dimension: float = Field(..., description="Worst fractal dimension", example=0.1189)
    
    def to_feature_dict(self) -> Dict[str, float]:
        """Convert to feature dictionary with correct names."""
        return {
            "mean radius": self.mean_radius,
            "mean texture": self.mean_texture,
            "mean perimeter": self.mean_perimeter,
            "mean area": self.mean_area,
            "mean smoothness": self.mean_smoothness,
            "mean compactness": self.mean_compactness,
            "mean concavity": self.mean_concavity,
            "mean concave points": self.mean_concave_points,
            "mean symmetry": self.mean_symmetry,
            "mean fractal dimension": self.mean_fractal_dimension,
            "radius error": self.radius_error,
            "texture error": self.texture_error,
            "perimeter error": self.perimeter_error,
            "area error": self.area_error,
            "smoothness error": self.smoothness_error,
            "compactness error": self.compactness_error,
            "concavity error": self.concavity_error,
            "concave points error": self.concave_points_error,
            "symmetry error": self.symmetry_error,
            "fractal dimension error": self.fractal_dimension_error,
            "worst radius": self.worst_radius,
            "worst texture": self.worst_texture,
            "worst perimeter": self.worst_perimeter,
            "worst area": self.worst_area,
            "worst smoothness": self.worst_smoothness,
            "worst compactness": self.worst_compactness,
            "worst concavity": self.worst_concavity,
            "worst concave points": self.worst_concave_points,
            "worst symmetry": self.worst_symmetry,
            "worst fractal dimension": self.worst_fractal_dimension,
        }


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    
    diagnosis: str = Field(..., description="Diagnosis result: 'Malignant' or 'Benign'")
    prediction: int = Field(..., description="Numeric prediction (0=Malignant, 1=Benign)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Class probabilities",
        example={"malignant": 0.95, "benign": 0.05}
    )
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    samples: List[PatientFeatures] = Field(
        ..., 
        description="List of patient feature sets",
        min_length=1,
        max_length=100
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse]
    total_samples: int
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_type: str
    is_fitted: bool
    training_date: Optional[str]
    version: str
    n_features: int
    feature_names: List[str]


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    detail: str
    error_type: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global predictor
    
    try:
        predictor = CancerPredictor.from_pretrained()
        logger.info("Model loaded successfully on startup")
    except FileNotFoundError:
        logger.warning(
            "No pretrained model found. Training new model..."
        )
        predictor = CancerPredictor.train_new(model_type="random_forest", save=True)
        logger.info("New model trained and saved")


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with welcome message."""
    return {
        "message": "Cancer Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check() -> HealthResponse:
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    responses={503: {"model": ErrorResponse}}
)
async def get_model_info() -> ModelInfoResponse:
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    info = predictor.get_model_info()
    feature_names = predictor.get_feature_names()
    
    return ModelInfoResponse(
        model_type=info["model_type"],
        is_fitted=info["is_fitted"],
        training_date=info.get("training_date"),
        version="1.0.0",
        n_features=len(feature_names),
        feature_names=feature_names
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={
        503: {"model": ErrorResponse},
        422: {"model": ErrorResponse}
    }
)
async def predict_single(features: PatientFeatures) -> PredictionResponse:
    """
    Make a cancer prediction for a single patient.
    
    Takes 30 morphological features from cell nuclei and returns
    a diagnosis (Malignant/Benign) with confidence score.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        
        # Convert to feature dict and predict
        feature_dict = features.to_feature_dict()
        result = predictor.predict_single(feature_dict)
        
        inference_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            diagnosis=result["diagnosis"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            inference_time_ms=round(inference_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    responses={503: {"model": ErrorResponse}}
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Make predictions for multiple patients.
    
    Maximum 100 samples per request.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        predictions = []
        
        for sample in request.samples:
            sample_start = time.time()
            feature_dict = sample.to_feature_dict()
            result = predictor.predict_single(feature_dict)
            sample_time = (time.time() - sample_start) * 1000
            
            predictions.append(PredictionResponse(
                diagnosis=result["diagnosis"],
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                inference_time_ms=round(sample_time, 2)
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions),
            total_inference_time_ms=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/features",
    tags=["Model"]
)
async def get_feature_names() -> Dict[str, List[str]]:
    """Get list of required feature names."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "feature_names": predictor.get_feature_names(),
        "total_features": len(predictor.get_feature_names())
    }


# Run with: uvicorn api.cancer_api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
