import argparse, json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score,
    f1_score, accuracy_score, average_precision_score, roc_auc_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from .config import DATA_PATH, MODEL_PATH, METADATA_PATH, REPORTS, MODEL_FEATURES, MODEL_VERSION
from .data import load_dataset
from .features import engineer_features

def select_threshold(y, p, target_precision):
    precision, recall, thresholds = precision_recall_curve(y, p)
    candidates = []
    for i, threshold in enumerate(thresholds):
        if precision[i] >= target_precision:
            candidates.append((f1_score(y, p >= threshold, zero_division=0), threshold))
    if candidates:
        return float(max(candidates)[1]), "precision_target"
    scores = [f1_score(y, p >= t, zero_division=0) for t in thresholds]
    return float(thresholds[max(range(len(scores)), key=scores.__getitem__)]), "best_f1_fallback"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 < args.test_size < 0.5 or not 0 < args.target_precision <= 1:
        raise ValueError("Invalid test size or target precision.")
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    X, y = engineer_features(df), df["Class"]
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, random_state=42, stratify=y_tv)

    scale = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        reg_lambda=1.0, objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=scale, tree_method="hist", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    threshold, method = select_threshold(
        y_val, model.predict_proba(X_val)[:, 1], args.target_precision)
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)

    metrics = {
        "model_version": MODEL_VERSION, "threshold": threshold,
        "threshold_selection": method, "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y_test, probs)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0)
    }
    model.save_model(str(MODEL_PATH))
    metadata = {
        "model_version": MODEL_VERSION, "model_type": "XGBClassifier",
        "model_features": MODEL_FEATURES, "threshold": threshold,
        "threshold_selection": method, "scale_pos_weight": scale,
        "training_rows": len(X_train), "validation_rows": len(X_val), "test_rows": len(X_test)
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title("Fraud Detection - Test Confusion Matrix")
    plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix.png", dpi=150); plt.close()
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure(); plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.tight_layout()
    plt.savefig(REPORTS / "precision_recall_curve.png", dpi=150); plt.close()
    importance = pd.Series(model.feature_importances_, index=MODEL_FEATURES).nlargest(20).sort_values()
    plt.figure(figsize=(9, 7)); importance.plot(kind="barh")
    plt.title("Top 20 XGBoost Features"); plt.tight_layout()
    plt.savefig(REPORTS / "feature_importance.png", dpi=150); plt.close()
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
