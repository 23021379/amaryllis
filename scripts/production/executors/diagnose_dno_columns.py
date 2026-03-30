import pandas as pd
import re
import sys

# Path from exec_99_decision_dossier.py
INPUT_DATA_PATH = r"[REDACTED_BY_SCRIPT]"

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df

def diagnose_input():
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        # Load just headers and a few rows
        df = pd.read_csv(INPUT_DATA_PATH, nrows=100)
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    df = sanitize_column_names(df)
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 1. Check for DNO column
    dno_cols = [c for c in df.columns if 'dno' in c.lower()]
    print(f"[REDACTED_BY_SCRIPT]")
    
    if dno_cols:
        for col in dno_cols:
            print(f"Value counts for '{col}':")
            print(df[col].value_counts())
            
    # 2. Check for UKPN columns
    ukpn_cols = [c for c in df.columns if 'ukpn' in c.lower()]
    print(f"[REDACTED_BY_SCRIPT]")
    for c in ukpn_cols[:10]: # Print first 10
        print(f"  - {c}")
        
    # 3. Check for NG columns
    ng_cols = [c for c in df.columns if 'ng_' in c.lower() or 'nged' in c.lower()]
    print(f"[REDACTED_BY_SCRIPT]")
    for c in ng_cols[:10]:
        print(f"  - {c}")

    # 4. Test Coalesce Logic
    print("[REDACTED_BY_SCRIPT]")
    COALESCE_MAP = {
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
    }
    
    for target, sources in COALESCE_MAP.items():
        existing = [c for c in sources if c in df.columns]
        print(f"  Target: {target}")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        
        if not existing:
            print("    ❌ NO SOURCES FOUND!")
        else:
            # Check if we have data for these sources
            for src in existing:
                non_null = df[src].count()
                print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    diagnose_input()
