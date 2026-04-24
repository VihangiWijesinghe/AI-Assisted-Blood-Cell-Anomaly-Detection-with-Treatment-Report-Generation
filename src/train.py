import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

# Load data
df = pd.read_csv(r"d:\AML project\blood_cell_anomaly_detection.csv")

# Drop non-feature columns
X = df.drop(columns=["cell_id", "anomaly_label", "cell_type"])
y_anomaly = df["anomaly_label"]
y_cell = df["cell_type"]

# Convert categorical input columns to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=False)

# Encode target labels if needed
cell_le = LabelEncoder()
y_cell_encoded = cell_le.fit_transform(y_cell)

anomaly_le = LabelEncoder()
y_anomaly_encoded = anomaly_le.fit_transform(y_anomaly)

# Split once so everything stays aligned
X_train, X_test, y_train_a, y_test_a, y_train_c, y_test_c = train_test_split(
    X,
    y_anomaly_encoded,
    y_cell_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_anomaly_encoded
)

# Train models
anomaly_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

cell_model = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss"
)

anomaly_model.fit(X_train, y_train_a)
cell_model.fit(X_train, y_train_c)

# Create output folder if it doesn't exist
os.makedirs(r"d:\AML project\models", exist_ok=True)

# Save models
joblib.dump(anomaly_model, r"d:\AML project\models\anomaly_model.pkl")
joblib.dump(cell_model, r"d:\AML project\models\celltype_model.pkl")
joblib.dump(cell_le, r"d:\AML project\models\celltype_label_encoder.pkl")
joblib.dump(anomaly_le, r"d:\AML project\models\anomaly_label_encoder.pkl")
joblib.dump(list(X.columns), r"d:\AML project\models\feature_columns.pkl")

print("Models trained and saved.")