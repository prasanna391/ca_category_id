from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

# CHANGE THIS LINE TO MATCH YOUR ACTUAL CSV FILE NAME
RAW_DATA = BASE_DIR / "data" / "raw" / "CA_category_id.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "processed_data.csv"


def preprocess_data():
    """Read raw YouTube data and create processed data."""

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA}"
        )

    df = pd.read_csv(RAW_DATA)

    print("Original dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA, index=False)

    print("\nProcessed dataset shape:", df.shape)
    print(f"\nProcessed data saved to: {PROCESSED_DATA}")


if __name__ == "__main__":
    preprocess_data()
