from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

from src.preprocess import preprocess_data


def run_preprocessing():
    print("Starting preprocessing...")

    preprocess_data()

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    run_preprocessing()