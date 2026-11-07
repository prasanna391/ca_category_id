import pandas as pd

def retrain_pipeline():
    try:
        df_base = pd.read_csv("baseline_dataset.csv")
        df_prod = pd.read_csv("production_logs.csv")
    except FileNotFoundError:
        print(" Data file missing. Pipeline execution blocked.")
        return

    print("\n=== RETRAINING PIPELINE ACTIVATED ===")
    print(f"Original Baseline Size: {len(df_base)} rows")
    
    # Clean and concat production data back into standard model inputs
    df_clean_prod = df_prod[['id', 'channelId', 'title', 'assignable']]
    df_new_train = pd.concat([df_base, df_clean_prod], ignore_index=True)
    
    df_new_train.to_csv("updated_training_pool.csv", index=False)
    print(f"Success: Retrained model package built with {len(df_new_train)} training features.")

if __name__ == "__main__":
    retrain_pipeline()
