import pandas as pd
import re

INPUT_DATA_PATH = r"[REDACTED_BY_SCRIPT]"

def sanitize_column_names(df):
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df

def diagnose_full():
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        # Load only relevant columns to save memory/time
        # We need to sanitize headers first, but read_csv reads headers as is.
        # So we read headers first.
        headers = pd.read_csv(INPUT_DATA_PATH, nrows=0).columns.tolist()
        sanitized_headers = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in headers]
        
        # Identify the actual column names in the file that correspond to what we want
        # We want '[REDACTED_BY_SCRIPT]' and '[REDACTED_BY_SCRIPT]'
        # But we need to find their *original* names in the file.
        
        # Helper to find original name given sanitized target
        def find_orig(target):
            for orig, san in zip(headers, sanitized_headers):
                if san == target:
                    return orig
            return None
            
        ukpn_col = find_orig('[REDACTED_BY_SCRIPT]')
        ng_col = find_orig('[REDACTED_BY_SCRIPT]')
        hex_col = find_orig('hex_id')
        
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        
        usecols = [c for c in [hex_col, ukpn_col, ng_col] if c is not None]
        
        df = pd.read_csv(INPUT_DATA_PATH, usecols=usecols)
        # Sanitize now
        df = sanitize_column_names(df)
        
        print(f"[REDACTED_BY_SCRIPT]")
        
        if '[REDACTED_BY_SCRIPT]' in df.columns:
            ukpn_count = df['[REDACTED_BY_SCRIPT]'].count()
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            
        if '[REDACTED_BY_SCRIPT]' in df.columns:
            ng_count = df['[REDACTED_BY_SCRIPT]'].count()
            print(f"[REDACTED_BY_SCRIPT]")
            
        # Check for overlap or exclusivity
        if '[REDACTED_BY_SCRIPT]' in df.columns and '[REDACTED_BY_SCRIPT]' in df.columns:
            both = df[df['[REDACTED_BY_SCRIPT]'].notna() & df['[REDACTED_BY_SCRIPT]'].notna()]
            print(f"[REDACTED_BY_SCRIPT]")
            
            ukpn_only = df[df['[REDACTED_BY_SCRIPT]'].notna() & df['[REDACTED_BY_SCRIPT]'].isna()]
            print(f"[REDACTED_BY_SCRIPT]")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    diagnose_full()
