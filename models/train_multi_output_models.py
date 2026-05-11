import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


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

print("=" * 60)
print("Dataset loaded successfully")
print("Dataset shape:", df.shape)
print("=" * 60)


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
# Remove invalid / artefact samples from disease training
# ---------------------------------------------------------

df = df[df["clean_disease"] != "Invalid Sample"].copy()

print("\nClean disease labels after removing invalid samples:")
print(df["clean_disease"].value_counts())

print("\nDataset shape after cleaning:", df.shape)
print("=" * 60)


# ---------------------------------------------------------
# Targets
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

# Convert categorical input features to numeric
X = pd.get_dummies(X, drop_first=False)

feature_columns = list(X.columns)

joblib.dump(feature_columns, os.path.join(MODEL_DIR, "multi_feature_columns.pkl"))

print("\nFeature columns saved:")
print(os.path.join(MODEL_DIR, "multi_feature_columns.pkl"))
print("Number of input features:", len(feature_columns))


# ---------------------------------------------------------
# Train models
# ---------------------------------------------------------

for target in target_columns:
    print("\n" + "=" * 60)
    print(f"Training model for: {target}")
    print("=" * 60)

    y = df[target]

    print("\nClass distribution:")
    print(y.value_counts())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # If a class has fewer than 2 samples, stratify will fail.
    # This fallback prevents training from crashing on very small classes.
    class_counts = y.value_counts()
    can_stratify = class_counts.min() >= 2

    if can_stratify:
        stratify_value = y_encoded
    else:
        stratify_value = None
        print(
            "\nWarning: At least one class has fewer than 2 samples. "
            "Training without stratify for this target."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value,
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{target} Accuracy: {accuracy:.4f}")

    target_names = [str(class_name) for class_name in label_encoder.classes_]

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0,
        )
    )

    # Important:
    # Save clean_disease model using disease_category filenames
    # because multi_predict.py already loads disease_category_model.pkl
    if target == "clean_disease":
        model_filename = "disease_category_model.pkl"
        encoder_filename = "disease_category_label_encoder.pkl"
    else:
        model_filename = f"{target}_model.pkl"
        encoder_filename = f"{target}_label_encoder.pkl"

    model_path = os.path.join(MODEL_DIR, model_filename)
    encoder_path = os.path.join(MODEL_DIR, encoder_filename)

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print("\nSaved model:")
    print(model_path)

    print("Saved label encoder:")
    print(encoder_path)


print("\n" + "=" * 60)
print("All multi-output models trained and saved successfully.")
print("=" * 60)