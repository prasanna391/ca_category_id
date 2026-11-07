import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to input dataset")
    args = parser.parse_args()

    print(f" Validating data quality for: {args.input}")
    try:
        df = pd.read_csv(args.input)
        required_columns = ['id', 'title', 'assignable']
        
        # Check columns
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Quality Gate Failed: Missing required column '{col}'")
                sys.exit(1)
                
        # Check empty rows
        if df.isnull().values.any():
            print(" Warning: Data contains missing values, but continuing.")
            
        print(" Quality Gate Passed successfully! All required headers match.")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
