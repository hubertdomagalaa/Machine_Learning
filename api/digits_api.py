"""
Digit Recognition REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import logging
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.digits import DigitsPredictor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Digit Recognition API",
    description="Handwritten digit recognition (0-9) using machine learning.",
    version="1.0.0",
)


predictor: Optional[DigitsPredictor] = None


class DigitFeatures(BaseModel):
    """64 pixel values from 8x8 image."""
    
    pixels: List[float] = Field(
        ...,
        min_length=64,
        max_length=64,
        description="64 pixel values (0-16) from flattened 8x8 image"
    )


class DigitImage(BaseModel):
    """8x8 image matrix."""
    
    image: List[List[float]] = Field(
        ...,
        description="8x8 pixel matrix"
    )


class PredictionResponse(BaseModel):
    """Prediction response."""
    
    digit: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float


@app.on_event("startup")
async def load_model():
    """Load model."""
    global predictor
    
    try:
        predictor = DigitsPredictor.from_pretrained()
        logger.info("Digits model loaded")
    except FileNotFoundError:
        logger.warning("Training new model...")
        predictor = DigitsPredictor.train_new(model_type="svm", save=True)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {"message": "Digit Recognition API", "docs": "/docs"}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_digit(features: DigitFeatures) -> PredictionResponse:
    """Recognize digit from pixel values."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    X = np.array(features.pixels).reshape(1, -1)
    result = predictor.predict(X)
    
    inference_time = (time.time() - start) * 1000
    
    return PredictionResponse(
        digit=result["digits"][0],
        confidence=result["confidences"][0],
        probabilities={str(i): float(result["probabilities"][0][i]) for i in range(10)},
        inference_time_ms=round(inference_time, 2)
    )


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_from_image(data: DigitImage) -> PredictionResponse:
    """Recognize digit from 8x8 image matrix."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    image = np.array(data.image)
    if image.shape != (8, 8):
        raise HTTPException(status_code=400, detail="Image must be 8x8")
    
    start = time.time()
    result = predictor.predict_image(image)
    inference_time = (time.time() - start) * 1000
    
    return PredictionResponse(
        digit=result["digits"][0],
        confidence=result["confidences"][0],
        probabilities={str(i): float(result["probabilities"][0][i]) for i in range(10)},
        inference_time_ms=round(inference_time, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
