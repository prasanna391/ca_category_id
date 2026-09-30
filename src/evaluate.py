import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "youtube_category_assignable_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "processed_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

def evaluate_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    text_column = "title"
    target_column = "id"

    df = df.dropna(subset=[text_column, target_column])

    X = df[text_column].astype(str)
    y = df[target_column].astype(str)

    predictions = model.predict(X).astype(str)
    accuracy = accuracy_score(y, predictions)

    # Prepare the evaluation text report
    report_lines = [
        "Model Evaluation",
        "=" * 40,
        f"Accuracy: {accuracy:.4f}\n",
        "Classification Report:",
        classification_report(y, predictions),
        "\nConfusion Matrix:",
        str(confusion_matrix(y, predictions))
    ]
    report_text = "\n".join(report_lines)

    # 1. Print to the terminal
    print(report_text)

    # 2. Save inside the outputs directory
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
        
    report_save_path = OUTPUT_DIR / "evaluation_report.txt"
    with open(report_save_path, "w") as f:
        f.write(report_text)
        
    print(f"\n[SUCCESS] Report successfully saved to: {report_save_path}")

if __name__ == "__main__":
    evaluate_model()
