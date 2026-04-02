import pandas as pd
import numpy as np

# Load the CSV file
try:
    df = pd.read_csv('[REDACTED_BY_SCRIPT]')
except FileNotFoundError:
    print("[REDACTED_BY_SCRIPT]")
    exit()
except Exception as e:
    print(f"[REDACTED_BY_SCRIPT]")
    exit()

# List of columns to check
columns_to_check = [
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]",
    "dhv_corridor_intersects_site",
    "[REDACTED_BY_SCRIPT]",
    "[REDACTED_BY_SCRIPT]"
]

# Check for missing columns
available_columns = [col for col in columns_to_check if col in df.columns]
missing_columns = list(set(columns_to_check) - set(available_columns))

if missing_columns:
    print(f"[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]")

if not available_columns:
    print("[REDACTED_BY_SCRIPT]")
else:
    print("[REDACTED_BY_SCRIPT]")
    for col in available_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            zero_count = (df[col] == 0).sum()
            print(f"- '{col}': {zero_count}")
        else:
            print(f"- '{col}'[REDACTED_BY_SCRIPT]")
