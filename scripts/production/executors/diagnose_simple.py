import pandas as pd
import re

INPUT_DATA_PATH = r"[REDACTED_BY_SCRIPT]"

def sanitize_column_names(df):
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df

def diagnose():
    try:
        df = pd.read_csv(INPUT_DATA_PATH, nrows=100)
        df = sanitize_column_names(df)
        
        print("--- DIAGNOSTIC START ---")
        
        # Check DNO
        if 'dno' in df.columns:
            print(f"[REDACTED_BY_SCRIPT]'dno'].value_counts()}")
        else:
            print("[REDACTED_BY_SCRIPT]")
            
        # Check specific columns
        cols_to_check = [
            '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]'
        ]
        
        for col in cols_to_check:
            if col in df.columns:
                print(f"Found '{col}'[REDACTED_BY_SCRIPT]")
            else:
                print(f"MISSING '{col}'")
                
        print("--- DIAGNOSTIC END ---")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose()
