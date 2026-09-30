from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw" / "CA_category_id.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DATA = PROCESSED_DIR / "processed_data.csv"


def preprocess_data():
    print("[INFO] Starting production preprocessing...")

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(RAW_DATA)

    # Keep only the columns needed for the production dataset
    df = df[
        [
            "id",
            "channelId",
            "title",
            "assignable"
        ]
    ].copy()

    # Remove rows without a category title
    df = df.dropna(subset=["title"])

    # Clean text
    df["title"] = (
        df["title"]
        .astype(str)
        .str.strip()
    )

    # Remove duplicate category records
    df = df.drop_duplicates()

    # Save processed data
    df.to_csv(
        PROCESSED_DATA,
        index=False
    )

    print(f"[INFO] Processed rows: {len(df)}")
    print(f"[INFO] Saved to: {PROCESSED_DATA}")
    print("[SUCCESS] Preprocessing completed!")


if __name__ == "__main__":
    preprocess_data()