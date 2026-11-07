import pandas as pd
from scipy.stats import chi2_contingency

def check_data_drift():
    try:
        df_base = pd.read_csv("baseline_dataset.csv")
        df_prod = pd.read_csv("production_logs.csv")
    except FileNotFoundError:
        print("Error: Missing data files. Run 12_create_evolved_dataset.py first.")
        return False

    # Get categorical frequencies for 'assignable'
    base_counts = df_base['assignable'].astype(str).value_counts()
    prod_counts = df_prod['assignable'].astype(str).value_counts()
    
    contingency_table = pd.DataFrame({'Baseline': base_counts, 'Production': prod_counts}).fillna(0)
    
    # Statistical test
    stat, p_value, dof, expected = chi2_contingency(contingency_table.T)
    
    print("\n=== DATA DRIFT REPORT ===")
    print(f"P-Value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(" ALERT: Data drift detected! Distribution has changed drastically.")
        return True
    else:
        print("Success: No data drift detected.")
        return False

if __name__ == "__main__":
    check_data_drift()
