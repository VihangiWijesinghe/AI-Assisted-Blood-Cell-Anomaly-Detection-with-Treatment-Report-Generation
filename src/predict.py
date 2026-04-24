import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = r"d:\AML project\models"

anomaly_model = joblib.load(os.path.join(BASE_DIR, "anomaly_model.pkl"))
cell_model = joblib.load(os.path.join(BASE_DIR, "celltype_model.pkl"))
cell_label_encoder = joblib.load(os.path.join(BASE_DIR, "celltype_label_encoder.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

def predict(features: dict):
    # Convert input dict to DataFrame
    X = pd.DataFrame([features])

    # Apply same encoding as training
    X = pd.get_dummies(X)

    # Add missing columns from training
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # Remove unexpected extra columns
    X = X[feature_columns]

    anomaly_prob = anomaly_model.predict_proba(X)[0][1]
    anomaly_pred = int(anomaly_prob > 0.5)

    cell_pred_encoded = cell_model.predict(X)[0]
    cell_type = cell_label_encoder.inverse_transform([int(cell_pred_encoded)])[0]

    confidence = max(anomaly_prob, 1 - anomaly_prob)

    result = {
        "predicted_cell_type": cell_type,
        "predicted_anomaly_label": anomaly_pred,
        "confidence": float(confidence),
        "anomaly_score": float(anomaly_prob),
        "manual_review_required": anomaly_prob > 0.7 or confidence < 0.65
    }

    return result