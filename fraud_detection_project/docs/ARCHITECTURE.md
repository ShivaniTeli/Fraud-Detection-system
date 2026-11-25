# Architecture Guide

## Offline training
1. Load and validate CSV columns and labels.
2. Apply shared feature engineering.
3. Split train/validation/test with stratification.
4. Train XGBoost using training-only class weighting.
5. Select threshold on validation data.
6. Evaluate on held-out test data.
7. Save model and metadata.

## Online inference
1. Client sends transaction ID and raw feature dictionary.
2. FastAPI validates required feature names.
3. Shared feature function builds the model row.
4. XGBoost returns fraud probability.
5. Risk policy creates score and action band.
6. SHAP optionally generates local feature attributions.
7. SQLite records an audit row.
8. API returns structured JSON.

## Design choices
- Shared feature module prevents training/serving transformations from diverging.
- Probability and business decision policy are separate.
- SHAP failure does not need to take down the prediction service.
- Test data is not used to select thresholds.
- SQLite keeps local setup simple; use PostgreSQL for a multi-instance production service.

## Production topology extension
```text
Banking channel -> API gateway/auth -> fraud scoring replicas
                                      |-> online feature service
                                      |-> versioned model registry
                                      |-> risk policy
                                      -> immediate decision
                                      -> event broker (async)
                                           |-> audit/data lake
                                           |-> monitoring
                                           |-> case management
                                           |-> delayed labels
```
