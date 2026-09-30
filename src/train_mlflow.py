import os
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/youtube_category_assignable_model.pkl"

os.makedirs("models", exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
print("[INFO] Loading YouTube dataset...")

df = pd.read_csv(DATA_PATH)

print(f"[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}")


# -----------------------------
# Check required columns (UPDATED FOR YOUR DATASET)
# -----------------------------
if "title" not in df.columns:
    raise ValueError("Column 'title' not found in dataset.")


# -----------------------------
# Remove missing values
# -----------------------------
df = df.dropna(
    subset=["title", "id"]
)


# -----------------------------
# Input and target (UPDATED FOR YOUR DATASET)
# -----------------------------
# We map 'title' as text input features and 'assignable' or 'id' as a dummy category status
X = df["title"].astype(str)
y = df["id"].astype(str)


# -----------------------------
# Train/Test split (FIXED: REMOVED STRATIFY=Y)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_experiment(
    "YouTube_Category_Prediction"
)


# -----------------------------
# Start MLflow run
# -----------------------------
with mlflow.start_run():

    print("[INFO] Starting MLflow experiment...")


    # -----------------------------
    # Model parameters
    # -----------------------------
    n_neighbors = 3
    test_size = 0.2
    random_state = 42


    # -----------------------------
    # Create model
    # -----------------------------
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2)
            )
        ),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=n_neighbors
            )
        )
    ])


    # -----------------------------
    # Train model
    # -----------------------------
    print("[INFO] Training model...")

    model.fit(
        X_train,
        y_train
    )


    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict(
        X_test
    )


    # -----------------------------
    # Calculate accuracy
    # -----------------------------
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"[INFO] Accuracy: {accuracy:.4f}"
    )


    # -----------------------------
    # Log parameters to MLflow
    # -----------------------------
    mlflow.log_param(
        "model",
        "KNeighborsClassifier"
    )

    mlflow.log_param(
        "n_neighbors",
        n_neighbors
    )

    mlflow.log_param(
        "test_size",
        test_size
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "vectorizer",
        "TfidfVectorizer"
    )

    mlflow.log_param(
        "ngram_range",
        "(1,2)"
    )


    # -----------------------------
    # Log metric
    # -----------------------------
    mlflow.log_metric(
        "accuracy",
        accuracy
    )


    # -----------------------------
    # Save model locally
    # -----------------------------
    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"[INFO] Model saved to: {MODEL_PATH}"
    )


    # -----------------------------
    # Log model to MLflow (FIXED ARTIFACT PARAMETER & FORMAT)
    # -----------------------------
    print("[INFO] Logging model to MLflow...")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="youtube_category_model",
        serialization_format="pickle"
    )


    # -----------------------------
    # Success
    # -----------------------------
    print(
        "[SUCCESS] MLflow model logged successfully."
    )

    print(
        "[SUCCESS] MLflow experiment completed."
    )
