from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = ROOT / "data" / "processed" / "processed_data.csv"


def validate_outputs():
    print("[INFO] Validating processed output...")

    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA}"
        )

    df = pd.read_csv(PROCESSED_DATA)

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
            f"Missing output columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            "Processed dataset is empty."
        )

    if df["title"].isnull().any():
        raise ValueError(
            "Processed output contains empty titles."
        )

    print(f"[INFO] Output rows: {len(df)}")
    print(f"[INFO] Output columns: {df.columns.tolist()}")
    print("[SUCCESS] Output validation passed!")


if __name__ == "__main__":
    validate_outputs()