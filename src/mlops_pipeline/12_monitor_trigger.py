import importlib

# Correctly import modules that start with numbers using importlib
try:
    data_drift_mod = importlib.import_module("12_data_drift")
    retrain_model_mod = importlib.import_module("12_retrain_model")
    
    check_data_drift = data_drift_mod.check_data_drift
    retrain_pipeline = retrain_model_mod.retrain_pipeline
except ModuleNotFoundError:
    # Fallback if executing directly outside the folder structure
    data_drift_mod = importlib.import_module("mlops_pipeline.12_data_drift")
    retrain_model_mod = importlib.import_module("mlops_pipeline.12_retrain_model")
    
    check_data_drift = data_drift_mod.check_data_drift
    retrain_pipeline = retrain_model_mod.retrain_pipeline

def run_orchestrator():
    print("Starting system orchestration monitor script...")
    
    # 1. Run Data Drift Check
    drift_detected = check_data_drift()
    
    # 2. Trigger retraining if drift thresholds fail
    if drift_detected:
        print("\n[Trigger Alert] Condition Met -> Activating fallback loops.")
        retrain_pipeline()
    else:
        print("\nSystem running normally. No automated retraining required.")

if __name__ == "__main__":
    run_orchestrator()
