import os
import subprocess

def run_integration_test():
    print(" Starting Lab 9 Integration Tests...")
    
    # Run the quality gate check automatically
    result = subprocess.run(["python", "src/mlops_pipeline/07_quality_gate.py", "--input", "data/CA_category_id.csv"])
    
    if result.returncode == 0:
        print(" Lab 9: Complete Pipeline Integrity Test PASSED successfully!")
    else:
        print(" Lab 9: System integration test failed.")

if __name__ == "__main__":
    run_integration_test()
