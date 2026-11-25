import pytest
from src.risk import assess

def test_risk_bands():
    assert assess(0.1).decision == "APPROVE"
    assert assess(0.5).decision == "REVIEW"
    assert assess(0.9).decision == "BLOCK"

def test_invalid_probability():
    with pytest.raises(ValueError):
        assess(1.5)
