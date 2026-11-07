import pandas as pd

def monitor_api_performance():
    try:
        df_prod = pd.read_csv("production_logs.csv")
    except FileNotFoundError:
        print("Error: production_logs.csv not found.")
        return

    avg_latency = df_prod['api_latency_ms'].mean()
    errors = df_prod[df_prod['http_status'] != 200]
    error_rate = (len(errors) / len(df_prod)) * 100

    print("\n=== API PERFORMANCE REPORT ===")
    print(f"Total Traffic Processed : {len(df_prod)} requests")
    print(f"Average Server Latency  : {avg_latency:.2f} ms")
    print(f"HTTP Failure Rate       : {error_rate:.2f}% ({len(errors)} errors)")

if __name__ == "__main__":
    monitor_api_performance()
