
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle
import joblib

# Load data
diabetes = pd.read_csv('diabetes_unclean.csv')

# --- Data Cleaning ---
# Drop duplicates
diabetes = diabetes.drop_duplicates()

# Handle missing values for numerical columns
num_cols = ["AGE","Urea", "Cr", "HbA1c","Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]
for col in num_cols:
    if col in diabetes.columns and diabetes[col].isnull().sum() > 0:
        diabetes[col].fillna(diabetes[col].median(), inplace=True)

# Handle missing values for categorical columns
cat_cols = ["Gender"]
for col in cat_cols:
    if col in diabetes.columns and diabetes[col].isnull().sum() > 0:
        diabetes[col].fillna(diabetes[col].mode()[0], inplace=True)

# Encode Gender if present
if "Gender" in diabetes.columns:
    diabetes["Gender"] = diabetes["Gender"].str.title().str[0]
    diabetes["Gender"] = diabetes["Gender"].map({"M": 1, "F": 0})

# Target variable
y = diabetes["CLASS"]
X = diabetes.drop(["CLASS"], axis=1)

# Encode target
labEn = LabelEncoder()
y = labEn.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=True, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a single model (Random Forest performs well on tabular data)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save artifacts for API
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(labEn, "label_encoder.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\nArtifacts saved: diabetes_model.pkl, scaler.pkl, label_encoder.pkl, feature_columns.pkl")
