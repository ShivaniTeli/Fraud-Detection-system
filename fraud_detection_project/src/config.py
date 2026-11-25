from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data/creditcard.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts/xgboost_fraud_model.json"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", ROOT / "artifacts/metadata.json"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "artifacts/fraud_predictions.db"))
REPORTS = ROOT / "reports"
BASE_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
MODEL_FEATURES = BASE_FEATURES + ["log_amount", "transaction_hour"]
MODEL_VERSION = "xgboost-fraud-v1"
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "30"))
BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", "70"))
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "true").lower() == "true"
