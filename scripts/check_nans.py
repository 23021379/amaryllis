import pandas as pd
import os

Null_sentinels = [ -1.0, -9999, 'Unknown', 'N/A', 'NA', None ]

def analyze_nan_percentage(csv_file_paths, features_to_check):
    """
    Analyzes a list of CSV files to determine the percentage of NaN values 
    for a specific list of features (columns) and prints a formatted report.

    Args:
        csv_file_paths (list): A list of strings, where each string is a 
                               path to a CSV file.
        features_to_check (list): A list of column names to analyze.
    """
    if not csv_file_paths:
        print("[REDACTED_BY_SCRIPT]")
        return

    for file_path in csv_file_paths:
        # Use os.path.basename to get just the filename for the report header
        file_name = os.path.basename(file_path)
        header = f"[REDACTED_BY_SCRIPT]"
        print(header)

        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path)

            nan_proxy = df.isin(Null_sentinels)
            df = df.mask(nan_proxy, other=pd.NA)
            if df.empty:
                print("[REDACTED_BY_SCRIPT]")
            else:
                # Calculate the percentage of NaN values for all columns
                nan_percentage = (df.isnull().sum() / len(df)) * 100

                # Print the overall percentage of NaN values for all columns
                print("[REDACTED_BY_SCRIPT]")
                for column in df.columns:
                    if nan_percentage[column] > 90:
                        percentage = nan_percentage[column]
                        print(f"  - Feature '{column}'[REDACTED_BY_SCRIPT]")

                # # Display the results only for the specified columns
                # for column in features_to_check:
                #     if column in df.columns:
                #         percentage = nan_percentage[column]
                #         print(f"  - Feature '{column}'[REDACTED_BY_SCRIPT]")
                #     else:
                #         print(f"  - Feature '{column}'[REDACTED_BY_SCRIPT]")

        except FileNotFoundError:
            print(f"  ERROR: The file '{file_path}' was not found.")
        except pd.errors.EmptyDataError:
            print("[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
        
        # Print a footer for separation and add a newline for readability
        print("-" * len(header))
        print() # Adds a blank line between reports

# --- Main execution block ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Add the file paths of your CSV files to this list.
    # Example for Windows: ["[REDACTED_BY_SCRIPT]", "data\\data2.csv"]
    # Example for macOS/Linux: ["[REDACTED_BY_SCRIPT]", "data/data2.csv"]
    
    feature_to_check = [
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'avg_total_kva_5nn',
        '[REDACTED_BY_SCRIPT]',
        'idno_is_within',
        '[REDACTED_BY_SCRIPT]',
        'idno_count_in_5km',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'lpa_idno_area_as_percent_of_total_area',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'ltds_count_in_10km',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'lct_total_connections_in_5km',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'lct_ev_connections_in_5km',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'lct_reconciliation_delta_connections',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'dnoa_avg_dist_knn_m',
        'dnoa_avg_deferred_kva_knn',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'pq_idw_thd_knn5',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'pq_max_thd_in_knn5',
        'pq_std_thd_in_knn5',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'sitevoltage',
        'powertransformercount',
        'datecommissioned',
        'assessmentdate',
        'transratingwinter',
        'transratingsummer',
        'maxdemandsummer',
        'maxdemandwinter',
        'resistance_ohm',
        'reversepower_encoded',
        'substation_age_at_submission',
        'time_since_last_assessment_days',
        'is_hot_site',
        '[REDACTED_BY_SCRIPT]',
        'has_demand_data',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'kva_per_transformer',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'ohl_local_structure_count',
        'ohl_local_tower_count',
        'ohl_local_tower_ratio',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'ohl_nearest_voltage',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'constraint_season_Winter',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'solar_farm_id',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'sec_sub_gov_area_sqm',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
    ]


    files_to_analyze = [
        r"[REDACTED_BY_SCRIPT]"
    ]


    if not files_to_analyze:
        print("[REDACTED_BY_SCRIPT]'files_to_analyze'[REDACTED_BY_SCRIPT]")
    else:
        analyze_nan_percentage(files_to_analyze, feature_to_check)
