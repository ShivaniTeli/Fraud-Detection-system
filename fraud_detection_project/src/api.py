from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from .config import BASE_FEATURES, MODEL_VERSION, REVIEW_THRESHOLD, BLOCK_THRESHOLD, ENABLE_SHAP
from .predictor import FraudPredictor
from .risk import assess
from .database import initialize, save, feedback

predictor = None

@asynccontextmanager
async def lifespan(app):
    global predictor
    initialize()
    try:
        predictor = FraudPredictor()
    except FileNotFoundError:
        predictor = None
    yield

app = FastAPI(title="Financial Fraud Detection API", version="1.0.0", lifespan=lifespan)

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: str = Field(min_length=1, max_length=100)
    features: Dict[str, float]

class FeedbackRequest(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=100)
    actual_label: int = Field(ge=0, le=1)

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": predictor is not None}

@app.get("/model-info")
def model_info():
    if predictor is None:
        raise HTTPException(503, "Train the model first.")
    return {"model_version": predictor.metadata.get("model_version", MODEL_VERSION),
            "raw_features": BASE_FEATURES,
            "model_features": predictor.metadata.get("model_features", []),
            "risk_thresholds": {"review": REVIEW_THRESHOLD, "block": BLOCK_THRESHOLD}}

@app.post("/predict")
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model not ready. Run python -m src.train.")
    missing = [f for f in BASE_FEATURES if f not in request.features]
    extra = [f for f in request.features if f not in BASE_FEATURES]
    if missing or extra:
        raise HTTPException(422, {"missing_features": missing, "unexpected_features": extra})
    try:
        pred = predictor.predict_one(request.features, ENABLE_SHAP)
        risk = assess(pred["fraud_probability"])
        result = {
            "transaction_id": request.transaction_id,
            "fraud_probability": pred["fraud_probability"],
            "risk_score": risk.score, "decision": risk.decision,
            "model_version": predictor.metadata.get("model_version", MODEL_VERSION),
            "explanation": pred["explanation"]
        }
        save(request.transaction_id, result, result["model_version"])
        return result
    except ValueError as exc:
        raise HTTPException(422, str(exc))

@app.post("/feedback")
def add_feedback(request: FeedbackRequest):
    count = feedback(request.transaction_id, request.actual_label)
    if count == 0:
        raise HTTPException(404, "No matching prediction found.")
    return {"status": "feedback_saved"}
