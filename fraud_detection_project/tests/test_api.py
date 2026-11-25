from fastapi.testclient import TestClient
from src.api import app

def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert "model_ready" in response.json()
