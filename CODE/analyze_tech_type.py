import pandas as pd
import os

files_to_check = [
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]',
    r'[REDACTED_BY_SCRIPT]'
]

for file_name in files_to_check:
    if not os.path.exists(file_name):
        continue
        
    try:
        df = pd.read_csv(file_name)
        if 'technology_type' in df.columns:
            print(f"[REDACTED_BY_SCRIPT]")
            total_samples = len(df)
            counts = df['technology_type'].value_counts()
            percentages = (counts / total_samples) * 100
            
            summary = pd.DataFrame({
                'Count': counts,
                'Percentage': percentages
            })
            print(summary)
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]'technology_type' column not found.")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
