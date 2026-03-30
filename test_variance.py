import pandas as pd
import os

def compare_feature_statistics(file1_path, file2_path):
    """
    Loads two CSV files, calculates descriptive statistics for each,
    saves them to new CSV files, and shows the difference to identify
    features with differing metrics.

    Args:
        file1_path (str): The file path for the first CSV.
        file2_path (str): The file path for the second CSV.
    """
    try:
        # Load the datasets
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        # Find common columns to compare
        common_columns = df1.columns.intersection(df2.columns)
        if len(common_columns) == 0:
            print("[REDACTED_BY_SCRIPT]")
            return
            
        print(f"[REDACTED_BY_SCRIPT]")

        # Filter dataframes to only common columns
        df1_common = df1[common_columns]
        df2_common = df2[common_columns]

        # Calculate descriptive statistics on common columns
        desc1 = df1_common.describe()
        desc2 = df2_common.describe()

        desc1.drop(index=['count'], inplace=True)
        desc2.drop(index=['count'], inplace=True)

        # Create output file paths for the statistics
        file1_base, file1_ext = os.path.splitext(file1_path)
        stats1_path = f"[REDACTED_BY_SCRIPT]"
        
        file2_base, file2_ext = os.path.splitext(file2_path)
        stats2_path = f"[REDACTED_BY_SCRIPT]"

        # Save the descriptive statistics to CSV files
        desc1.to_csv(stats1_path)
        desc2.to_csv(stats2_path)

        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")

        # Calculate the absolute difference between the statistics
        # The .reindex_like() ensures both dataframes have the same columns and rows for subtraction
        diff = (desc1 - desc2).abs()

        # Save the difference to its own CSV file
        diff_path = r"[REDACTED_BY_SCRIPT]"
        diff.to_csv(diff_path)
        print(f"[REDACTED_BY_SCRIPT]")

        print("[REDACTED_BY_SCRIPT]")
        print(desc1)
        print("[REDACTED_BY_SCRIPT]")
        print(desc2)
        print("[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")
        
        # --- START: Identify and print significant changes ---
        print("[REDACTED_BY_SCRIPT]")
        # Create a boolean mask for differences greater than 1
        significant_changes_mask = diff > 1
        
        # Get the columns (features) that have at least one significant change
        features_with_changes = diff.columns[significant_changes_mask.any()]

        if features_with_changes.empty:
            print("[REDACTED_BY_SCRIPT]")
        else:
            # Iterate through features that have significant changes
            for feature in features_with_changes:
                # Select the specific metrics for that feature that are > 1
                changed_metrics = diff.loc[significant_changes_mask[feature], feature]
                print(f"\nFeature: '{feature}'")
                print(changed_metrics.to_string())
        print("\n" + "="*50 + "\n")
        # --- END: Identify and print significant changes ---
        features_with_changes = diff[features_with_changes]
        features_with_changes.to_csv(r"[REDACTED_BY_SCRIPT]")
        # Set display options to show all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        #print(diff)

    except FileNotFoundError as e:
        print(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Replace with the correct file paths
    file1 = r"[REDACTED_BY_SCRIPT]"
    # !!! IMPORTANT: Please replace this with the path to your second file.
    file2 = r"[REDACTED_BY_SCRIPT]"

    compare_feature_statistics(file1, file2)