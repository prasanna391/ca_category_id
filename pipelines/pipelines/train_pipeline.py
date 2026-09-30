from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

from src.train import train_model


def run_training():
    print("Starting model training...")

    train_model()

    print("\nTraining completed.")


if __name__ == "__main__":
    run_training()