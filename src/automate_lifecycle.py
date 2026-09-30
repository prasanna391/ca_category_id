from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_DIR = ROOT / "model_registry"

CHAMPION_FILE = (
    REGISTRY_DIR /
    "champion.json"
)


def promote_champion():

    print("[INFO] Checking model registry...")

    metadata_files = list(
        REGISTRY_DIR.glob("metadata_*.json")
    )

    if not metadata_files:
        raise FileNotFoundError(
            "No registered model metadata found."
        )

    models = []

    for file in metadata_files:

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

            models.append(metadata)

    print(
        f"[INFO] Found {len(models)} registered model version(s)."
    )

    champion = max(
        models,
        key=lambda x: x["accuracy"]
    )

    champion_data = {
        "model_name": champion["model_name"],
        "version": champion["version"],
        "accuracy": champion["accuracy"],
        "status": "Champion"
    }

    with open(
        CHAMPION_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            champion_data,
            file,
            indent=4
        )

    print(
        f"[INFO] Champion model: {champion['version']}"
    )

    print(
        f"[INFO] Champion accuracy: "
        f"{champion['accuracy']:.4f}"
    )

    print(
        "[SUCCESS] Model lifecycle completed!"
    )


if __name__ == "__main__":
    promote_champion()