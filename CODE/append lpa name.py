import pandas as pd
import os

def append_lpa_column():
    # Define file paths based on your workspace structure
    repd_path = r'[REDACTED_BY_SCRIPT]'
    model_ready_path = r'[REDACTED_BY_SCRIPT]'
    output_path = r'[REDACTED_BY_SCRIPT]'

    print("Loading datasets...")
    try:
        repd_df = pd.read_csv(repd_path)
        model_ready_df = pd.read_csv(model_ready_path)
    except FileNotFoundError as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    # Prepare the lookup dataframe from REPD
    # We only need the coordinates and the target column
    repd_lookup = repd_df[['easting', 'northing', 'planning_authority']].copy()

    # Important: Drop duplicates based on coordinates in the source file
    # This prevents the merge from creating duplicate rows if REPD has multiple entries for one site
    initial_len = len(repd_lookup)
    repd_lookup.drop_duplicates(subset=['easting', 'northing'], keep='first', inplace=True)
    print(f"[REDACTED_BY_SCRIPT]")

    print("Merging data...")
    # Perform a left merge to keep all rows from the ModelReady file
    merged_df = pd.merge(
        model_ready_df,
        repd_lookup,
        left_on=['easting_x', 'northing_x'],
        right_on=['easting', 'northing'],
        how='left'
    )

    # Drop the redundant coordinate columns from the REPD file (easting/northing)
    # keeping easting_x/northing_x
    merged_df.drop(columns=['easting', 'northing'], inplace=True)

    # Check for missing matches
    missing_count = merged_df['planning_authority'].isna().sum()
    if missing_count > 0:
        print(f"[REDACTED_BY_SCRIPT]")

    print(f"[REDACTED_BY_SCRIPT]")
    merged_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    append_lpa_column()