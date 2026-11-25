# Project Documentation and Explanation

## Executive summary
This project is an educational end-to-end card fraud detection prototype. XGBoost estimates the probability of fraud. A separate risk policy maps that probability to an illustrative 0–100 score and an action band. SHAP provides local feature attributions. FastAPI exposes predictions, SQLite stores audit records, and Streamlit provides a simple interface.

## Business problem
Fraud is a rare event compared with legitimate transactions. A model can achieve high accuracy by predicting almost everything as legitimate while missing fraud. The goal is to balance fraud detection, alert quality, customer friction and analyst capacity.

## Data
The Kaggle dataset contains `Time`, `Amount`, anonymized PCA features `V1`–`V28`, and target `Class`. `Class=1` means fraud. V-features do not have publicly interpretable business semantics. Do not claim V14 means location or V4 means spending behavior.

## Feature engineering
The pipeline retains raw features and adds `log_amount=log1p(Amount)` and `transaction_hour=(Time/3600)%24`. Time is elapsed seconds from the first transaction in this dataset, so the derived hour is only an approximation, not an actual local clock hour.

## Why XGBoost?
XGBoost is gradient-boosted decision trees. Trees are added sequentially to reduce errors from earlier trees. It performs well on tabular data, learns nonlinear interactions and supports class weighting and SHAP.
Key parameters: `n_estimators` (number of trees), `max_depth` (tree complexity), `learning_rate` (contribution per tree), `subsample` (row sampling), `colsample_bytree` (feature sampling), `min_child_weight` (minimum child weight), `reg_lambda` (L2 regularization), `scale_pos_weight` (positive class weighting).

## Imbalance and metrics
`scale_pos_weight` is calculated from the training partition only as negative count divided by positive count. It is a baseline strategy, not guaranteed to be optimal. Evaluate precision, recall, F1, PR-AUC, ROC-AUC and the confusion matrix. Accuracy alone is inadequate for rare fraud.

Precision answers: of the transactions flagged, how many were fraud? Recall answers: of all fraud, how many were caught? PR-AUC summarizes precision/recall across thresholds and is useful for imbalanced data.

## Threshold and risk policy
The model outputs a probability. A threshold is selected on validation data to meet a requested precision target where possible, otherwise best validation F1. Test data is held out for final evaluation. The API's default decision bands (below 30 approve, 30–69.99 review, 70+ block) are illustrative policy thresholds and are not the same as the ML classification threshold. Real thresholds require business cost analysis and calibration.

## SHAP
SHAP assigns contributions to features relative to a baseline model output. Positive contributions push the output toward fraud; negative contributions push it away. SHAP is not causal proof. With anonymized V-features, the explanation is limited to feature IDs.

## Architecture
Offline training: CSV -> validation -> feature engineering -> stratified train/validation/test -> XGBoost -> validation threshold selection -> held-out test metrics -> saved model and metadata.
Online scoring: client -> FastAPI -> schema validation -> shared feature transformation -> model probability -> risk policy -> optional SHAP -> SQLite audit log -> JSON response.

## API
- `GET /health`: service and model readiness.
- `GET /model-info`: model and feature metadata.
- `POST /predict`: fraud probability, score, decision and explanation.
- `POST /feedback`: attaches a delayed binary label to the latest prediction for a transaction ID.

## Monitoring and production
Monitor request latency, errors, feature distributions, missing inputs, prediction rates, and precision/recall when delayed labels arrive. Drift is a signal to investigate, not automatic proof that retraining is required. Production improvements include PostgreSQL, Kafka, Redis, a point-in-time feature store, MLflow, auth/RBAC, encryption, CI/CD and cloud deployment. These are extensions, not included implementations.

## Limitations and responsible use
This is not a production banking system. Public anonymized data may not generalize to real fraud. Probability calibration is not implemented. SQLite is local-only. No real customer or card data should be uploaded. SHAP explains model behavior but does not establish causality. Human governance and review are required for consequential decisions.

## 30-second interview pitch
"I built a fraud detection prototype using XGBoost. I validated and transformed an anonymized transaction dataset, handled class imbalance using positive-class weighting, and evaluated the model with precision, recall, F1 and PR-AUC rather than relying only on accuracy. I selected a classification threshold using validation data, exposed inference through FastAPI, converted probability into a configurable risk score, and used SHAP to explain individual predictions. I added an audit log and a Streamlit interface to demonstrate the complete workflow."
