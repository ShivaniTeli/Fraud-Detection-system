import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import json
from .config import DATABASE_PATH

def connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize():
    with connect() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            risk_score REAL NOT NULL,
            decision TEXT NOT NULL,
            model_version TEXT NOT NULL,
            explanation_json TEXT,
            actual_label INTEGER
        )''')

def save(transaction_id, result, version):
    with connect() as db:
        db.execute('''INSERT INTO predictions
        (transaction_id, created_at, fraud_probability, risk_score, decision, model_version, explanation_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)''', (
            transaction_id, datetime.now(timezone.utc).isoformat(),
            result["fraud_probability"], result["risk_score"], result["decision"],
            version, json.dumps(result.get("explanation", []))
        ))

def feedback(transaction_id, label):
    with connect() as db:
        cur = db.execute('''UPDATE predictions SET actual_label = ?
        WHERE id = (SELECT id FROM predictions WHERE transaction_id = ? ORDER BY id DESC LIMIT 1)''',
        (label, transaction_id))
        return cur.rowcount
