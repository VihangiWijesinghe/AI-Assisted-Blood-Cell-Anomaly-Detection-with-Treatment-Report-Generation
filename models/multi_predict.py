import os
import joblib
import pandas as pd


MODEL_DIR = r"D:\AML project\models"

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "multi_feature_columns.pkl")
)

models = {
    "cell_type": joblib.load(os.path.join(MODEL_DIR, "cell_type_model.pkl")),
    "anomaly_label": joblib.load(os.path.join(MODEL_DIR, "anomaly_label_model.pkl")),
    "disease_category": joblib.load(os.path.join(MODEL_DIR, "disease_category_model.pkl")),
}

encoders = {
    "cell_type": joblib.load(os.path.join(MODEL_DIR, "cell_type_label_encoder.pkl")),
    "anomaly_label": joblib.load(os.path.join(MODEL_DIR, "anomaly_label_label_encoder.pkl")),
    "disease_category": joblib.load(os.path.join(MODEL_DIR, "disease_category_label_encoder.pkl")),
}


def prepare_input(features: dict):
    X = pd.DataFrame([features])

    X = pd.get_dummies(X, drop_first=False)

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    return X


def predict_all(features: dict) -> dict:
    X = prepare_input(features)

    results = {}

    for target_name, model in models.items():
        encoder = encoders[target_name]

        pred_encoded = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        predicted_label = encoder.inverse_transform([pred_encoded])[0]
        confidence = float(probabilities[pred_encoded])

        class_probabilities = {}

        for i, class_name in enumerate(encoder.classes_):
            class_probabilities[str(class_name)] = round(float(probabilities[i]), 4)

        results[target_name] = {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "class_probabilities": class_probabilities
        }

    anomaly_prediction = results["anomaly_label"]["prediction"]
    disease_prediction = results["disease_category"]["prediction"]

    manual_review_required = (
        str(anomaly_prediction) == "1"
        or results["disease_category"]["confidence"] < 0.70
        or disease_prediction != "Normal"
)

    return {
        "predicted_cell_type": results["cell_type"]["prediction"],
        "cell_type_confidence": results["cell_type"]["confidence_percentage"],

        "predicted_anomaly_label": results["anomaly_label"]["prediction"],
        "anomaly_confidence": results["anomaly_label"]["confidence_percentage"],

        "predicted_disease": results["disease_category"]["prediction"],
        "disease_confidence": results["disease_category"]["confidence_percentage"],

        "manual_review_required": manual_review_required,

        "details": results
    }