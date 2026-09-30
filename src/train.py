import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "CA_category_id.csv"
MODEL_PATH = BASE_DIR / "models" / "youtube_category_assignable_model.pkl"


def train_model():

    # Check CSV
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {DATA_PATH}"
        )

    # Read CSV
    df = pd.read_csv(DATA_PATH)

    print("CSV loaded successfully!")
    print("Columns:", df.columns.tolist())
    print("Number of rows:", len(df))

    # Remove rows with missing values
    df = df.dropna(subset=["title", "assignable"])

    # Input and target
    X = df["title"].astype(str)
    y = df["assignable"].astype(str)

    # Convert True/False to 1/0
    y = y.map({
        "True": 1,
        "False": 0
    })

    # Create ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(ngram_range=(1, 2))
        ),
        (
            "knn",
            KNeighborsClassifier(n_neighbors=3)
        )
    ])

    # Train
    model.fit(X, y)

    # Create models folder
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model
    joblib.dump(model, MODEL_PATH)

    print()
    print("Model trained successfully!")
    print("Model saved at:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_model()