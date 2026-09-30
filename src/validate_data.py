from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "CA_category_id.csv"


def validate_data():
    print("[INFO] Validating raw YouTube category data...")

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA}"
        )

    df = pd.read_csv(RAW_DATA)

    print(f"[INFO] Dataset found: {RAW_DATA}")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    required_columns = [
        "id",
        "channelId",
        "title",
        "assignable"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["id"].isnull().any():
        raise ValueError("Missing values found in id.")

    if df["title"].isnull().any():
        raise ValueError("Missing values found in title.")

    print("[SUCCESS] Raw data validation passed!")


if __name__ == "__main__":
    validate_data()