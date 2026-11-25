from dataclasses import dataclass
from .config import REVIEW_THRESHOLD, BLOCK_THRESHOLD

@dataclass
class Risk:
    score: float
    decision: str

def assess(probability, review=REVIEW_THRESHOLD, block=BLOCK_THRESHOLD):
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1].")
    if not 0 <= review < block <= 100:
        raise ValueError("Require 0 <= review < block <= 100.")
    score = round(probability * 100, 2)
    decision = "BLOCK" if score >= block else "REVIEW" if score >= review else "APPROVE"
    return Risk(score, decision)
