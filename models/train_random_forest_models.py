import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATASET_PATH = r"D:\AML project\blood_cell_anomaly_detection.csv"
MODEL_DIR = r"D:\AML project\models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("Random Forest Baseline Training")
print("=" * 70)
print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# Clean disease labels
# ---------------------------------------------------------

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


df["clean_disease"] = df["disease_category"].apply(clean_disease_label)

print("\nOriginal disease labels:")
print(df["disease_category"].value_counts())

print("\nClean disease labels before removing invalid samples:")
print(df["clean_disease"].value_counts())


# ---------------------------------------------------------
# Remove invalid / artefact samples
# ---------------------------------------------------------

df = df[df["clean_disease"] != "Invalid Sample"].copy()

print("\nClean disease labels after removing invalid samples:")
print(df["clean_disease"].value_counts())

print("\nDataset shape after cleaning:", df.shape)


# ---------------------------------------------------------
# Target columns
# ---------------------------------------------------------

target_columns = [
    "cell_type",
    "anomaly_label",
    "clean_disease",
]


# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------

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

# Convert categorical columns into numeric columns
X = pd.get_dummies(X, drop_first=False)

feature_columns = list(X.columns)

# Save Random Forest feature columns separately
joblib.dump(
    feature_columns,
    os.path.join(MODEL_DIR, "rf_feature_columns.pkl")
)

print("\nRandom Forest feature columns saved:")
print(os.path.join(MODEL_DIR, "rf_feature_columns.pkl"))
print("Number of input features:", len(feature_columns))


# ---------------------------------------------------------
# Store experiment results
# ---------------------------------------------------------

experiment_results = []


# ---------------------------------------------------------
# Train Random Forest models
# ---------------------------------------------------------

for target in target_columns:
    print("\n" + "=" * 70)
    print(f"Training Random Forest model for: {target}")
    print("=" * 70)

    y = df[target]

    print("\nClass distribution:")
    print(y.value_counts())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2

    if can_stratify:
        stratify_value = y_encoded
    else:
        stratify_value = None
        print(
            "\nWarning: At least one class has fewer than 2 samples. "
            "Training without stratification."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value,
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nRandom Forest Accuracy for {target}: {accuracy:.4f}")

    target_names = [str(class_name) for class_name in label_encoder.classes_]

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0,
        )
    )

    weighted_precision = report["weighted avg"]["precision"]
    weighted_recall = report["weighted avg"]["recall"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    experiment_results.append(
        {
            "Target": target,
            "Model": "Random Forest",
            "Accuracy": round(accuracy, 4),
            "Weighted Precision": round(weighted_precision, 4),
            "Weighted Recall": round(weighted_recall, 4),
            "Weighted F1-score": round(weighted_f1, 4),
        }
    )

    # Save RF model files separately, so Streamlit XGBoost files are not affected
    if target == "clean_disease":
        model_filename = "rf_disease_category_model.pkl"
        encoder_filename = "rf_disease_category_label_encoder.pkl"
    else:
        model_filename = f"rf_{target}_model.pkl"
        encoder_filename = f"rf_{target}_label_encoder.pkl"

    model_path = os.path.join(MODEL_DIR, model_filename)
    encoder_path = os.path.join(MODEL_DIR, encoder_filename)

    joblib.dump(rf_model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print("\nSaved Random Forest model:")
    print(model_path)

    print("Saved Random Forest label encoder:")
    print(encoder_path)


# ---------------------------------------------------------
# Save experiment summary
# ---------------------------------------------------------

results_df = pd.DataFrame(experiment_results)

results_path = os.path.join(MODEL_DIR, "random_forest_experiment_results.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 70)
print("Random Forest Experiment Summary")
print("=" * 70)
print(results_df)

print("\nSaved experiment results to:")
print(results_path)

print("\n" + "=" * 70)
print("All Random Forest baseline models trained successfully.")
print("Note: Streamlit app still uses XGBoost models only.")
print("=" * 70)