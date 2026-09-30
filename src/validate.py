from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "youtube_category_assignable_model.pkl"


def validate_model():

    print("Checking model...")

    if not MODEL_PATH.exists():
        print("ERROR: Model file does not exist.")
        return False

    try:
        model = joblib.load(MODEL_PATH)

        print("Model loaded successfully.")
        print("\nModel type:")
        print(type(model))

        print("\nPipeline:")
        print(model)

        if hasattr(model, "named_steps"):
            print("\nPipeline steps:")
            print(model.named_steps)

        return True

    except Exception as error:
        print("ERROR while loading model:")
        print(error)

        return False


if __name__ == "__main__":
    validate_model()