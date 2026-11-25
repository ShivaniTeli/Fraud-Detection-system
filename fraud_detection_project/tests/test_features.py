import pandas as pd
import pytest
from src.config import BASE_FEATURES, MODEL_FEATURES
from src.features import engineer_features

def test_feature_engineering():
    row = {c: 0.0 for c in BASE_FEATURES}
    row["Amount"] = 99.0
    result = engineer_features(pd.DataFrame([row]))
    assert list(result.columns) == MODEL_FEATURES
    assert result.loc[0, "log_amount"] > 0

def test_missing_feature():
    with pytest.raises(ValueError):
        engineer_features(pd.DataFrame([{"Amount": 5}]))
