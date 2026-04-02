import pandas as pd
import os

# Define file paths
file1 = '[REDACTED_BY_SCRIPT]'
file2 = '[REDACTED_BY_SCRIPT]'
file3 = '[REDACTED_BY_SCRIPT]'
output_file = '[REDACTED_BY_SCRIPT]'

# Read the CSV files into pandas DataFrames
try:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Concatenate the DataFrames horizontally
    merged_df = pd.concat([df1, df2, df3], axis=1)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)

    print(f"[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]", merged_df.columns.tolist())
    print("[REDACTED_BY_SCRIPT]", len(merged_df))

except FileNotFoundError as e:
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"An error occurred: {e}")
