import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


DATASET_PATH = r"D:\AML project\blood_cell_anomaly_detection.csv"
MODEL_DIR = r"D:\AML project\models"

os.makedirs(MODEL_DIR, exist_ok=True)


def clean_disease_label(value):
    value = str(value).strip()
    value_lower = value.lower()

    normal_labels = [
        "normal_platelet",
        "normal_wbc",
        "normal_rbc",
        "normal",
    ]

    if value_lower in normal_labels:
        return "Normal"

    if value_lower in ["artefact", "artifact"]:
        return "Invalid Sample"

    if "leukemia" in value_lower or "leukaemia" in value_lower:
        return "Leukemia"

    if "anemia" in value_lower or "anaemia" in value_lower:
        return "Anemia"

    if "infection" in value_lower:
        return "Infection"

    return value


print("=" * 70)
print("MODEL COMPARISON EXPERIMENT: RANDOM FOREST BASELINE VS XGBOOST")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

df["clean_disease"] = df["disease_category"].apply(clean_disease_label)
df = df[df["clean_disease"] != "Invalid Sample"].copy()

target_columns = [
    "cell_type",
    "anomaly_label",
    "clean_disease",
]

drop_columns = [
    "cell_id",
    "cell_type",
    "anomaly_label",
    "disease_category",
    "clean_disease",
    "cytodiffusion_anomaly_score",
    "cytodiffusion_classification_confidence",
    "labeller_confidence_score",
    "dataset_source",
]

X = df.drop(columns=[col for col in drop_columns if col in df.columns])
X = pd.get_dummies(X, drop_first=False)

results = []

for target in target_columns:
    print("\n" + "=" * 70)
    print(f"Target: {target}")
    print("=" * 70)

    y = df[target]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2
    stratify_value = y_encoded if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value,
    )

    models = {
        "Random Forest Baseline": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost Final Model": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss",
        ),
    }

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        results.append(
            {
                "Target": target,
                "Model": model_name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-score": round(f1, 4),
            }
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")


results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)
print(results_df)

output_path = os.path.join(MODEL_DIR, "model_comparison_results.csv")
results_df.to_csv(output_path, index=False)

print("\nSaved comparison results to:")
print(output_path)