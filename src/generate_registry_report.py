from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_DIR = ROOT / "model_registry"

CHAMPION_FILE = (
    REGISTRY_DIR /
    "champion.json"
)

REPORT_FILE = (
    REGISTRY_DIR /
    "registry_report.json"
)


def generate_report():

    print("[INFO] Generating model registry report...")

    if not CHAMPION_FILE.exists():
        raise FileNotFoundError(
            "Champion model information not found."
        )

    with open(
        CHAMPION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        champion = json.load(file)

    report = {
        "project": "YouTube Category Prediction",
        "dataset": "CA_category_id.csv",
        "model_name": champion["model_name"],
        "champion_version": champion["version"],
        "accuracy": champion["accuracy"],
        "status": champion["status"],
        "deployment_ready": True
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"[INFO] Report saved to: {REPORT_FILE}"
    )

    print(
        "[SUCCESS] Registry report generated!"
    )


if __name__ == "__main__":
    generate_report()