"""
Fraud Detection REST API.

FastAPI-based API for real-time fraud detection on financial transactions.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import logging
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fraud_detection import FraudPredictor, FraudDataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Fraud Detection API",
    description="""
    Real-time fraud detection API for financial transactions.
    
    Detects fraudulent transactions using XGBoost with:
    - Class imbalance handling (SMOTE)
    - Optimized decision threshold
    - Risk level classification
    """,
    version="1.0.0",
)


predictor: Optional[FraudPredictor] = None


class TransactionFeatures(BaseModel):
    """Request model for transaction features."""
    
    step: int = Field(..., description="Time step (hour)", example=1)
    amount: float = Field(..., description="Transaction amount", example=9839.64)
    oldbalanceOrg: float = Field(..., description="Origin balance before", example=170136.0)
    newbalanceOrig: float = Field(..., description="Origin balance after", example=160296.36)
    oldbalanceDest: float = Field(..., description="Dest balance before", example=0.0)
    newbalanceDest: float = Field(..., description="Dest balance after", example=0.0)
    isFlaggedFraud: int = Field(0, description="System flag", example=0)
    
    # Engineered features (optional - will be calculated if not provided)
    isPayment: Optional[int] = Field(None, description="Is payment type")
    isMovement: Optional[int] = Field(None, description="Is cash movement")
    isCashOut: Optional[int] = Field(None, description="Is cash out")
    isTransfer: Optional[int] = Field(None, description="Is transfer")
    transactionRatio: Optional[float] = Field(None, description="Amount/balance ratio")
    accountDiff: Optional[float] = Field(None, description="Balance difference")
    balanceChange: Optional[float] = Field(None, description="Origin balance change")
    destBalanceChange: Optional[float] = Field(None, description="Dest balance change")
    emptyBalance: Optional[int] = Field(None, description="Balance emptied")
    largeTransaction: Optional[int] = Field(None, description="Large transaction flag")
    balanceError: Optional[float] = Field(None, description="Balance calculation error")
    
    def to_feature_array(self) -> np.ndarray:
        """Convert to feature array with engineered features."""
        # Calculate missing features
        features = {
            "step": self.step,
            "amount": self.amount,
            "oldbalanceOrg": self.oldbalanceOrg,
            "newbalanceOrig": self.newbalanceOrig,
            "oldbalanceDest": self.oldbalanceDest,
            "newbalanceDest": self.newbalanceDest,
            "isFlaggedFraud": self.isFlaggedFraud,
            "isPayment": self.isPayment or 0,
            "isMovement": self.isMovement or 1,
            "isCashOut": self.isCashOut or 0,
            "isTransfer": self.isTransfer or 1,
            "transactionRatio": self.transactionRatio or (self.amount / (self.oldbalanceOrg + 1)),
            "accountDiff": self.accountDiff or abs(self.oldbalanceOrg - self.oldbalanceDest),
            "balanceChange": self.balanceChange or (self.oldbalanceOrg - self.newbalanceOrig),
            "destBalanceChange": self.destBalanceChange or (self.newbalanceDest - self.oldbalanceDest),
            "emptyBalance": self.emptyBalance or int(self.newbalanceOrig == 0),
            "largeTransaction": self.largeTransaction or int(self.amount > self.oldbalanceOrg),
            "balanceError": self.balanceError or abs(self.newbalanceOrig - max(0, self.oldbalanceOrg - self.amount)),
        }
        return np.array(list(features.values()))


class FraudPredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    
    is_fraud: bool = Field(..., description="Whether transaction is fraudulent")
    fraud_probability: float = Field(..., description="Fraud probability (0-1)")
    risk_level: str = Field(..., description="Risk level: MINIMAL, LOW, MEDIUM, HIGH, CRITICAL")
    threshold: float = Field(..., description="Decision threshold used")
    inference_time_ms: float = Field(..., description="Inference time in ms")


class BatchTransactionRequest(BaseModel):
    """Request model for batch predictions."""
    
    transactions: List[TransactionFeatures] = Field(
        ..., min_length=1, max_length=1000
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[FraudPredictionResponse]
    total_transactions: int
    fraud_count: int
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global predictor
    
    try:
        predictor = FraudPredictor.from_pretrained()
        logger.info("Fraud model loaded")
    except FileNotFoundError:
        logger.warning("No pretrained model, training new one...")
        predictor = FraudPredictor.train_new(
            model_type="gradient_boosting",  # Fallback if XGBoost not available
            sample_size=10000,
            save=True
        )
        logger.info("New model trained")


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=FraudPredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionFeatures) -> FraudPredictionResponse:
    """Detect fraud in a single transaction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        X = transaction.to_feature_array()
        result = predictor.predict(X)
        
        inference_time = (time.time() - start_time) * 1000
        
        return FraudPredictionResponse(
            is_fraud=result["is_fraud"][0],
            fraud_probability=result["fraud_probabilities"][0],
            risk_level=result["risk_levels"][0],
            threshold=result["threshold"],
            inference_time_ms=round(inference_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchTransactionRequest) -> BatchPredictionResponse:
    """Detect fraud in multiple transactions."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        predictions = []
        
        for txn in request.transactions:
            txn_start = time.time()
            X = txn.to_feature_array()
            result = predictor.predict(X)
            txn_time = (time.time() - txn_start) * 1000
            
            predictions.append(FraudPredictionResponse(
                is_fraud=result["is_fraud"][0],
                fraud_probability=result["fraud_probabilities"][0],
                risk_level=result["risk_levels"][0],
                threshold=result["threshold"],
                inference_time_ms=round(txn_time, 2)
            ))
        
        total_time = (time.time() - start_time) * 1000
        fraud_count = sum(1 for p in predictions if p.is_fraud)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(predictions),
            fraud_count=fraud_count,
            total_inference_time_ms=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
async def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return predictor.get_model_info()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
