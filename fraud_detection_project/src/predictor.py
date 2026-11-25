import json
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from .config import MODEL_PATH, METADATA_PATH
from .features import engineer_features
from .explain import explain

class FraudPredictor:
    def __init__(self):
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError("Train first: python -m src.train")
        self.model = XGBClassifier()
        self.model.load_model(str(MODEL_PATH))
        self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    def predict_one(self, raw_features, with_explanation=True):
        row = engineer_features(pd.DataFrame([raw_features]))
        probability = float(self.model.predict_proba(row)[0, 1])
        return {
            "fraud_probability": round(probability, 6),
            "explanation": explain(self.model, row) if with_explanation else []
        }
