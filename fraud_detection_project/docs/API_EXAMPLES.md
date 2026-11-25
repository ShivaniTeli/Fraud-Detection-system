# API Examples

Start the service with `uvicorn src.api:app --reload`.

## Health
`GET http://127.0.0.1:8000/health`

## Prediction example
```python
import requests
raw = {"Time": 1200.0, "Amount": 149.5}
raw.update({f"V{i}": 0.0 for i in range(1, 29)})
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"transaction_id": "DEMO-001", "features": raw},
    timeout=60
)
print(response.status_code, response.json())
```
All 30 raw features are required. Values are illustrative only.

## Feedback
```python
requests.post(
    "http://127.0.0.1:8000/feedback",
    json={"transaction_id": "DEMO-001", "actual_label": 0}
)
```
In production, feedback requires authenticated/authorized access and a unique prediction identifier.
