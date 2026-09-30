import os
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/youtube_category_assignable_model.pkl"


print("[INFO] Starting reproducibility validation...")


# -----------------------------
# Check files
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# FIXED: Updated columns for your schema
df = df.dropna(
    subset=["title", "id"]
)

# FIXED: Mapped text features to 'title' and target to 'id'
X = df["title"].astype(str)
y = df["id"].astype(str)


# -----------------------------
# Same split as training
# -----------------------------
# FIXED: Removed stratify=y rule for unique class safety
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Load model
# -----------------------------
model = joblib.load(
    MODEL_PATH
)


# -----------------------------
# Predict
# -----------------------------
predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("-----------------------------------------")
print("Reproducibility Validation")
print("-----------------------------------------")

print(f"Dataset rows       : {len(df)}")
print(f"Training samples   : {len(X_train)}")
print(f"Testing samples    : {len(X_test)}")
print(f"Accuracy           : {accuracy:.4f}")

print("-----------------------------------------")
print("[SUCCESS] Reproducibility validation completed!")
