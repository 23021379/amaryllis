import pandas as pd
import numpy as np

# File path for the dataset
file_path = r"[REDACTED_BY_SCRIPT]"

try:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Define the feature to analyze
    feature = '[REDACTED_BY_SCRIPT]'

    if feature in df.columns and 'solar_site_area_sqm' in df.columns:
        # Define the bins for stratification
        bins = [0, 5, 10, 15, 20, 25, 30, 40, 49, 50, 60, np.inf]
        
        # Create labels for the bins
        labels = [f'[REDACTED_BY_SCRIPT]' for i in range(len(bins)-2)]
        labels.append(f'60+')

        # Separate into solar and non-solar sites
        solar_df = df[df['solar_site_area_sqm'].notna()].copy()
        non_solar_df = df[df['solar_site_area_sqm'].isna()].copy()

        # --- Analysis for Solar Sites ---
        solar_df['capacity_strata'] = pd.cut(solar_df[feature], bins=bins, labels=labels, right=False)
        solar_strata_distribution = solar_df['capacity_strata'].value_counts().sort_index()
        
        print("[REDACTED_BY_SCRIPT]")
        print(solar_strata_distribution)
        print("\n" + "="*30 + "\n")

        # --- Analysis for Non-Solar Sites ---
        non_solar_df['capacity_strata'] = pd.cut(non_solar_df[feature], bins=bins, labels=labels, right=False)
        non_solar_strata_distribution = non_solar_df['capacity_strata'].value_counts().sort_index()

        print("[REDACTED_BY_SCRIPT]")
        print(non_solar_strata_distribution)

    else:
        if feature not in df.columns:
            print(f"Error: Feature '{feature}' not found in the file.")
        if 'solar_site_area_sqm' not in df.columns:
            print(f"Error: Feature 'solar_site_area_sqm' not found in the file.")

except FileNotFoundError:
    print(f"[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"An error occurred: {e}")
