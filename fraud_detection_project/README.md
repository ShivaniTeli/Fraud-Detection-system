# Real-Time Financial Fraud Detection & Explainable Risk Scoring

An end-to-end educational ML project using XGBoost, FastAPI, SHAP, SQLite and Streamlit.

## Business objective
Estimate the probability that a card transaction is fraudulent, convert it into a configurable 0–100 risk score, assign an illustrative action (`APPROVE`, `REVIEW`, `BLOCK`), and provide feature-level explanations.

## Dataset
Download the Kaggle Credit Card Fraud Detection dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` in `data/`. Expected columns: `Time`, `Amount`, `V1`–`V28`, `Class`. `Class=1` means fraud. The dataset is anonymized; do not interpret V1–V28 as real-world customer attributes.

## Architecture
```text
CSV -> validation -> feature engineering -> train/validation/test split
    -> XGBoost + class weighting -> validation threshold selection
    -> test metrics -> model artifact + metadata

Client -> FastAPI -> validation -> shared feature engineering -> XGBoost
       -> fraud probability -> risk policy -> optional SHAP
       -> SQLite audit record -> JSON response -> Streamlit dashboard
```

## Setup
Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Put the CSV in `data/`, then train:
```bash
python -m src.train --data data/creditcard.csv
```

Run API:
```bash
uvicorn src.api:app --reload
```
Docs: http://127.0.0.1:8000/docs

Run dashboard in another terminal:
```bash
streamlit run app/streamlit_app.py
```

Run tests:
```bash
pytest -q
```

## API
- `GET /health` — readiness.
- `GET /model-info` — model and feature metadata.
- `POST /predict` — probability, score, decision and SHAP attributions.
- `POST /feedback` — attach a delayed binary label to the latest matching transaction ID.

Prediction request must contain `transaction_id` and all 30 raw inputs (`Time`, `Amount`, `V1`–`V28`). See `docs/API_EXAMPLES.md`.

## Model approach
XGBoost is used for tabular classification. `scale_pos_weight` is calculated from the training split to account for class imbalance. Threshold selection uses validation data only; the test split is reserved for final evaluation. Report precision, recall, F1, PR-AUC, ROC-AUC and confusion matrix—not accuracy alone.

Features added: `log_amount` and an approximate `transaction_hour=(Time/3600)%24`. The latter is only an elapsed-time transformation, not a true wall-clock hour.

## Risk policy
The demo defaults are `<30 = APPROVE`, `30–<70 = REVIEW`, `>=70 = BLOCK`. These are illustrative thresholds, not banking policy. Calibrate using validation data, fraud-loss costs, false-positive costs and investigation capacity.

## Important limitations
This is not a production banking system. The public dataset is anonymized and may not represent current fraud patterns. Model probabilities are not necessarily calibrated. SHAP is an attribution method, not causal proof. SQLite is for local demonstration. Do not use real payment-card or sensitive customer data.

## Production extensions
PostgreSQL, Kafka, Redis, point-in-time feature store, MLflow, authentication/RBAC, encryption, Docker, CI/CD, drift monitoring and cloud deployment are documented as future work—not implemented in this starter package.
