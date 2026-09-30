from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "youtube_category_assignable_model.pkl"


def load_model():
    """Load the trained YouTube category prediction model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_category(text):
    """Predict the category for YouTube text."""
    model = load_model()

    prediction = model.predict([text])[0]

    return prediction


def main():
    print("YouTube Category Prediction")
    print("-" * 40)

    text = input("Enter YouTube title/description: ").strip()

    if not text:
        print("Please enter some text.")
        return

    prediction = predict_category(text)

    print(f"\nPredicted category: {prediction}")


if __name__ == "__main__":
    main()