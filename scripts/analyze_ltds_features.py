import pandas as pd

def analyze_ltds_features(file_path, columns_to_check):
    """
    Analyzes specified columns in a CSV file to check if they contain only '0.0' values.

    Args:
        file_path (str): The path to the CSV file.
        columns_to_check (list): A list of column names to analyze.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"[REDACTED_BY_SCRIPT]")

        results = {}
        for col in columns_to_check:
            if col not in df.columns:
                results[col] = "[REDACTED_BY_SCRIPT]"
                continue

            # Convert column to numeric, coercing errors to NaN
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            
            # Check if all non-NaN values are equal to 0
            if (numeric_col.dropna() == 0).all():
                results[col] = "[REDACTED_BY_SCRIPT]"
            else:
                non_zero_values = numeric_col[numeric_col != 0].dropna()
                results[col] = f"[REDACTED_BY_SCRIPT]"

        print("[REDACTED_BY_SCRIPT]")
        for col, result in results.items():
            print(f"- {col}: {result}")

    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # The user is currently viewing REPD_Amaryllis_L12.csv, so we'll analyze that.
    # If they want to analyze another file, they can change this path.
    csv_file = '[REDACTED_BY_SCRIPT]'
    
    features_to_analyze = [
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "ltds_count_in_10km",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]"
    ]
    
    analyze_ltds_features(csv_file, features_to_analyze)
