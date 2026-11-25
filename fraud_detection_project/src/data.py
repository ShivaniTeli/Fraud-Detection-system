from pathlib import Path
import pandas as pd
from .config import BASE_FEATURES

def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and save as data/creditcard.csv")
    df = pd.read_csv(path)
    required = BASE_FEATURES + ["Class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = df[required].copy()
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; inspect before training.")
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain binary labels 0 and 1.")
    df["Class"] = df["Class"].astype(int)
    return df
