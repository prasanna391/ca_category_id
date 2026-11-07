import pandas as pd
import numpy as np
import argparse
import os

def inject_failures(input_path, output_path, failure_type):
    print(f" Loading clean YouTube data: {input_path}")
    df = pd.read_csv(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if failure_type == "missing_values":
        print(" Injecting Failure: Missing titles...")
        mask = np.random.rand(len(df)) < 0.3
        df.loc[mask, "title"] = np.nan
    elif failure_type == "invalid_types":
        print(" Injecting Failure: Corrupting assignable...")
        mask = np.random.rand(len(df)) < 0.2
        df["assignable"] = ["MALFORMED_DATA" if m else str(val) for m, val in zip(mask, df["assignable"])]
    elif failure_type == "corrupt_ids":
        print(" Injecting Failure: Corrupting ids...")
        mask = np.random.rand(len(df)) < 0.2
        df["id"] = ["INVALID_ID_FORMAT" if m else str(val) for m, val in zip(mask, df["id"])]
    df.to_csv(output_path, index=False)
    print(f" Saved corrupted dataset to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--type", required=True)
    args = parser.parse_args()
    inject_failures(args.input, args.output, args.type)