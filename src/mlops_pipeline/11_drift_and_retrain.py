import pandas as pd
import subprocess
from scipy.stats import chi2_contingency

def detect_categorical_drift(baseline_path, production_path, threshold=0.05):
    print(f" Loading baseline data: {baseline_path}")
    df_base = pd.read_csv(baseline_path)
    print(f" Loading corrupted streaming data: {production_path}")
    df_prod = pd.read_csv(production_path)
    drift_detected = False
    if "assignable" in df_base.columns and "assignable" in df_prod.columns:
        print("\n Testing categorical distribution drift for assignable attribute...")
        base_counts = df_base["assignable"].astype(str).value_counts()
        prod_counts = df_prod["assignable"].astype(str).value_counts()
        combined = pd.DataFrame({"baseline": base_counts, "production": prod_counts}).fillna(0)
        stat, p_value, dof, expected = chi2_contingency(combined.values)
        print(f" - Chi-Square p-value: {p_value:.5f}")
        if p_value < threshold:
            print(" DRIFT DETECTED: The categorical distribution has radically broken baseline parameters!")
            drift_detected = True
    return drift_detected

def trigger_retraining():
    print("\n Drift validation failed. Executing automated training loop pipeline...")
    try:
        result = subprocess.run(["python", "-m", "pipelines.pipelines.train_pipeline"], check=True)
        if result.returncode == 0:
            print("🎉 Retraining loop pipeline finished completely without errors!")
    except Exception as e:
        print(f"❌ Automation failed to trigger pipeline script: {e}")

if __name__ == "__main__":
    try:
        if detect_categorical_drift("data/CA_category_id.csv", "data/corrupted_categories.csv"):
            trigger_retraining()
        else:
            print("\n System Stable: Distribution matches baseline expectations.")
    except Exception as e:
        print(f" Error running script: {e}")