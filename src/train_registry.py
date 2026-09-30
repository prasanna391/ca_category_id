from pathlib import Path
import json
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "CA_category_id.csv"

REGISTRY_DIR = ROOT / "model_registry"
REGISTRY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def train_model():

    print("[INFO] Loading YouTube category dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"[INFO] Dataset rows: {len(df)}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    required_columns = [
        "id",
        "channelId",
        "title",
        "assignable"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    df = df.dropna(
        subset=["title", "assignable"]
    )

    df["title"] = (
        df["title"]
        .astype(str)
        .str.strip()
    )

    X = df["title"]

    y = df["assignable"].astype(str)

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain at least two classes."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                max_features=5000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    print("[INFO] Training model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"[INFO] Model accuracy: {accuracy:.4f}")

    version = "v1"

    model_path = (
        REGISTRY_DIR /
        f"youtube_category_model_{version}.pkl"
    )

    metadata_path = (
        REGISTRY_DIR /
        f"metadata_{version}.json"
    )

    joblib.dump(
        model,
        model_path
    )

    metadata = {
        "model_name": "youtube_category_model",
        "version": version,
        "dataset": "CA_category_id.csv",
        "target": "assignable",
        "feature": "title",
        "accuracy": accuracy,
        "rows": len(df)
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    print(f"[INFO] Model saved: {model_path}")
    print(f"[INFO] Metadata saved: {metadata_path}")

    print(
        "[SUCCESS] Model training and registration completed!"
    )


if __name__ == "__main__":
    train_model()