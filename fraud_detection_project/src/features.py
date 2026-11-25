import numpy as np
import pandas as pd
from .config import BASE_FEATURES, MODEL_FEATURES

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BASE_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    x = frame[BASE_FEATURES].copy()
    for c in BASE_FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    if x.isna().any().any():
        raise ValueError("Features contain missing or non-numeric values.")
    if (x["Amount"] < 0).any():
        raise ValueError("Amount cannot be negative.")
    x["log_amount"] = np.log1p(x["Amount"])
    # Time is elapsed seconds in this dataset, not a real timestamp.
    x["transaction_hour"] = (x["Time"] / 3600.0) % 24.0
    return x[MODEL_FEATURES].astype(float)
