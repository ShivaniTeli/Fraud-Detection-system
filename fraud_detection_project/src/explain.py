def explain(model, row, top_n=5):
    try:
        import shap
        values = shap.TreeExplainer(model).shap_values(row)
        if isinstance(values, list):
            values = values[-1]
        vals = values[0]
        items = []
        for name, feature_value, contribution in zip(row.columns, row.iloc[0], vals):
            items.append({
                "feature": str(name),
                "feature_value": float(feature_value),
                "shap_value": float(contribution),
                "impact": "increases_fraud_score" if contribution > 0 else "decreases_fraud_score"
            })
        return sorted(items, key=lambda z: abs(z["shap_value"]), reverse=True)[:top_n]
    except Exception:
        # Explanations are optional; prediction remains available if SHAP fails.
        return []
