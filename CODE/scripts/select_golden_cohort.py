import pandas as pd
import numpy as np

# Configuration
SOURCE_CSV_PATH = r'[REDACTED_BY_SCRIPT]'
OUTPUT_COHORT_CSV_PATH = r'[REDACTED_BY_SCRIPT]'
# Use the original column names as seen in the raw CSV file
COLUMN_MAPPING = {
    'Planning Authority': 'planning_authority',
    'Technology Type': 'technology_type',
    'Storage Type': 'storage_type',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'CHP Enabled': 'chp_enabled',
    '[REDACTED_BY_SCRIPT]': 'ro_banding_roc_mwh',
    'FiT Tariff (p/kWh)': 'fit_tariff_p_kwh',
    'CfD Capacity (MW)': 'cfd_capacity_mw',
    'Turbine Capacity': 'turbine_capacity',
    'No. of Turbines': 'no_of_turbines',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'X-coordinate': 'easting',
    'Y-coordinate': 'northing',
    'Secretary of State Reference': 'secretary_of_state_reference',
    '[REDACTED_BY_SCRIPT]': 'type_of_sos_intervention',
    'Judicial Review': 'judicial_review',
    'Offshore Wind Round': 'offshore_wind_round',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'Appeal Lodged': 'appeal_lodged',
    'Appeal Withdrawn': 'appeal_withdrawn',
    'Appeal Refused': 'appeal_refused',
    'Appeal Granted': 'appeal_granted',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': 'sos_intervened',
    '[REDACTED_BY_SCRIPT]': 'sos_refusal',
    '[REDACTED_BY_SCRIPT]': 'sos_granted',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'Under Construction': 'under_construction',
    'Operational': 'operational',
    'Heat Network Ref': 'heat_network_ref',
    '[REDACTED_BY_SCRIPT]': 'solar_site_area_sqm'
}
# Invert the mapping to go from snake_case back to original, which is needed for selection
REVERSE_COLUMN_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}

def select_golden_cohort(source_path):
    """
    Selects 5 diverse solar applications to form the Golden Cohort for pipeline auditing.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        df = pd.read_csv(source_path, encoding='latin-1', low_memory=False)
        print("[REDACTED_BY_SCRIPT]")
    except FileNotFoundError:
        print(f"[REDACTED_BY_SCRIPT]'{source_path}'.")
        return

    # Use original column names for filtering
    tech_col = 'Technology Type'
    capacity_col = '[REDACTED_BY_SCRIPT]'
    status_col = '[REDACTED_BY_SCRIPT]'
    submitted_col = '[REDACTED_BY_SCRIPT]'
    granted_col = '[REDACTED_BY_SCRIPT]'
    refused_col = '[REDACTED_BY_SCRIPT]'
    ref_col = 'Ref ID' # Assuming 'Ref' is the unique ID column

    # 1. Filter for Solar PV only
    df_solar = df[df[tech_col] == 'Solar Photovoltaics'].copy()
    print(f"[REDACTED_BY_SCRIPT]")

    # 2. Convert relevant columns to numeric/datetime, coercing errors
    df_solar[capacity_col] = pd.to_numeric(df_solar[capacity_col], errors='coerce')
    df_solar[submitted_col] = pd.to_datetime(df_solar[submitted_col], errors='coerce', dayfirst=True)
    df_solar[granted_col] = pd.to_datetime(df_solar[granted_col], errors='coerce', dayfirst=True)
    df_solar[refused_col] = pd.to_datetime(df_solar[refused_col], errors='coerce', dayfirst=True)

    # 3. Calculate durations
    df_solar['approval_duration'] = (df_solar[granted_col] - df_solar[submitted_col]).dt.days
    df_solar['refusal_duration'] = (df_solar[refused_col] - df_solar[submitted_col]).dt.days

    # --- SELECTION CRITERIA ---
    cohort_indices = []

    # Case 1: Rural, sub-5MW, approved quickly.
    case1_df = df_solar[
        (df_solar[capacity_col] < 5) &
        (df_solar[status_col].isin(['Operational', 'Under Construction'])) &
        (df_solar['approval_duration'].notna())
    ].sort_values('approval_duration', ascending=True)
    if not case1_df.empty:
        cohort_indices.append(case1_df.index[0])
        print(f"[REDACTED_BY_SCRIPT]")


    # Case 2: Medium-scale, refused.
    case2_df = df_solar[
        (df_solar[capacity_col] >= 5) & (df_solar[capacity_col] <= 20) &
        (df_solar[status_col] == 'Refused') &
        (df_solar['refusal_duration'].notna())
    ].sort_values('refusal_duration', ascending=False)
    if not case2_df.empty:
        cohort_indices.append(case2_df.index[0])
        print(f"[REDACTED_BY_SCRIPT]")


    # Case 3: Large-scale, approved after long delay.
    case3_df = df_solar[
        (df_solar[capacity_col] > 20) &
        (df_solar[status_col].isin(['Operational', 'Under Construction'])) &
        (df_solar['approval_duration'].notna())
    ].sort_values('approval_duration', ascending=False)
    if not case3_df.empty:
        cohort_indices.append(case3_df.index[0])
        print(f"[REDACTED_BY_SCRIPT]")

    # Case 4 & 5: AONB / BMV proxies. We select based on location.
    # We need to pick candidates that are geographically distinct to increase the chance
    # of them falling into different special land designations.
    # Let's pick a refused one in a rural-sounding authority and an approved one in another.
    
    # Case 4: Potential AONB/Landscape issue (select a refused case in a rural county)
    case4_df = df_solar[
        (df_solar['Planning Authority'].str.contains('Cornwall', na=False)) &
        (df_solar[status_col] == 'Refused')
    ]
    if not case4_df.empty:
        cohort_indices.append(case4_df.index[0])
        print(f"[REDACTED_BY_SCRIPT]")


    # Case 5: Potential BMV/Farmland issue (select an approved case in a known agricultural county)
    case5_df = df_solar[
        (df_solar['Planning Authority'].str.contains('Lincolnshire', na=False)) &
        (df_solar[status_col].isin(['Operational', 'Under Construction']))
    ]
    if not case5_df.empty:
        cohort_indices.append(case5_df.index[0])
        print(f"[REDACTED_BY_SCRIPT]")

    # 4. Create and save the Golden Cohort DataFrame
    if len(cohort_indices) > 0:
        golden_cohort_df = df.loc[list(set(cohort_indices))] # Use set to remove duplicates
        print(f"[REDACTED_BY_SCRIPT]")
        golden_cohort_df.to_csv(OUTPUT_COHORT_CSV_PATH, index=False)
        print(f"[REDACTED_BY_SCRIPT]")
        # Display the selected rows for review
        print("[REDACTED_BY_SCRIPT]")
        print(golden_cohort_df[[ref_col, 'Planning Authority', tech_col, capacity_col, status_col, submitted_col, granted_col, refused_col]])
        print("---------------------------------")
    else:
        print("[REDACTED_BY_SCRIPT]")


if __name__ == "__main__":
    select_golden_cohort(SOURCE_CSV_PATH)
