import pandas as pd
import numpy as np
import os
import hashlib

# --- Artifact & Project Constants ---
UNIFIED_PREFORGE_PATH = r"[REDACTED_BY_SCRIPT]"
# Mandate: A new, separate data source for the demographic intelligence layer.
DEMOGRAPHIC_FEATURES_PATH = r"[REDACTED_BY_SCRIPT]" # ASSUMED PATH
# Mandate: A new, separate data source for the environmental intelligence layer.
ENVIRONMENTAL_FEATURES_PATH = r"[REDACTED_BY_SCRIPT]"

# Mandate: The output is the final, enriched artifact, ready for the Forge Imputation Engine.
ENRICHED_PREFORGE_PATH = r"[REDACTED_BY_SCRIPT]"
# --- DOCTRINE: THE FORENSIC RESOLUTION PROTOCOL ---
# Mandate: The original, FAILED composite key, preserved for diagnostic purposes.
FAILED_COMPOSITE_KEY = [
    'technology_type', '[REDACTED_BY_SCRIPT]', 'easting_x',
    'northing_x', 'submission_year', 'submission_month', 'submission_day'
]

# ARCHITECTURAL RULING: The new, expanded, and official composite key.
# '[REDACTED_BY_SCRIPT]' and 'permission_granted' are added to resolve lifecycle-based ambiguity.
COMPOSITE_KEY_COLS = FAILED_COMPOSITE_KEY + ['[REDACTED_BY_SCRIPT]', 'permission_granted']

FLOAT_PRECISION_COLS = {
    '[REDACTED_BY_SCRIPT]': 5, 'easting_x': 5, 'northing_x': 5
}
AUTHORITATIVE_JOIN_KEY = 'amaryllis_id'

def run_forensic_key_analysis(df1, df2, key_cols, id_col):
    """[REDACTED_BY_SCRIPT]"""
    df1_ids = set(df1[id_col])
    df2_ids = set(df2[id_col])
    
    missing_ids = df1_ids - df2_ids
    if not missing_ids:
        print("[REDACTED_BY_SCRIPT]")
        return

    first_missing_id = list(missing_ids)[0]
    
    # Find the row in df1 that generated this ID
    offending_row_df1 = df1[df1[id_col] == first_missing_id]
    
    # Now, we must find the corresponding row in df2 using a simpler key, like coordinates
    # This assumes coordinates are present and reasonably unique
    simple_key = ['easting_x', 'northing_x']
    if all(k in offending_row_df1.columns for k in simple_key) and all(k in df2.columns for k in simple_key):
        key_values = offending_row_df1[simple_key].iloc[0]
        corresponding_row_df2 = df2[
            (df2['easting_x'] == key_values['easting_x']) & 
            (df2['northing_x'] == key_values['northing_x'])
        ]

        print("[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        
        print("[REDACTED_BY_SCRIPT]")
        print(offending_row_df1[key_cols].iloc[0])
        
        if not corresponding_row_df2.empty:
            print("[REDACTED_BY_SCRIPT]")
            # Ensure all key_cols exist in the second dataframe before trying to access them
            available_key_cols = [k for k in key_cols if k in corresponding_row_df2.columns]
            print(corresponding_row_df2[available_key_cols].iloc[0])
        else:
            print("[REDACTED_BY_SCRIPT]")

# --- Phase -1: Forensic Diagnosis ---
def phase_minus_1_forensic_diagnosis(df, failed_key, context_cols_to_show):
    """
    Architectural Mandate: Isolates and reports on the nature of duplications
    based on the failed composite key definition.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    duplicates = df[df.duplicated(subset=failed_key, keep=False)]
    
    if duplicates.empty:
        print("[REDACTED_BY_SCRIPT]")
        return True

    print(f"[REDACTED_BY_SCRIPT]")
    print("-" * 80)
    # Sort for stable, comparable output
    for _, group in duplicates.sort_values(by=failed_key + context_cols_to_show).groupby(failed_key):
        print("[REDACTED_BY_SCRIPT]")
        print(group[failed_key + context_cols_to_show])
        print("-" * 80)
    
    return False



# --- Phases 1, 2, 3 (Largely unchanged, now rely on the new protocol) ---
def phase_1_ingest_and_resolve(preforge_path, demographic_path, environmental_path, join_key):
    """
    Loads data, sanitizes ALL the source and incoming data for irreconcilable duplicates,
    then synthesizes identifiers.
    """
    print("[REDACTED_BY_SCRIPT]")
    df_preforge = pd.read_csv(preforge_path)
    df_demographic = pd.read_csv(demographic_path)
    df_environmental = pd.read_csv(environmental_path)
    
    # --- FORENSIC DIAGNOSIS (UNCHANGED) ---
    phase_minus_1_forensic_diagnosis(df_preforge, FAILED_COMPOSITE_KEY, ['[REDACTED_BY_SCRIPT]', 'permission_granted'])

    # --- ARCHITECTURAL MANDATE: SANITIZE ALL INPUTS FOR IRRECONCILABLE DUPLICATES ---
    # This is a deliberate, necessary step to ensure data integrity before synthesis.
    
    # Sanitize the authoritative pre-forge data
    print("[REDACTED_BY_SCRIPT]")
    initial_rows_preforge = len(df_preforge)
    # THIS IS THE CRITICAL FIX: Use the FULL, CORRECT composite key.
    df_preforge.drop_duplicates(subset=COMPOSITE_KEY_COLS, inplace=True, keep='last')
    rows_dropped_preforge = initial_rows_preforge - len(df_preforge)
    if rows_dropped_preforge > 0:
        print(f"[REDACTED_BY_SCRIPT]")

    # Sanitize the incoming demographic data
    print("[REDACTED_BY_SCRIPT]")
    if 'easting' in df_demographic.columns:
        df_demographic.rename(columns={'easting':'easting_x', 'northing':'northing_x'}, inplace=True)
    initial_rows_demo = len(df_demographic)
    for col in ['submission_year', 'submission_month', 'submission_day']:
        if col in df_demographic.columns:
            df_demographic[col] = pd.to_numeric(df_demographic[col], errors='coerce').astype('Int64').astype(float)
    
    # ARCHITECTURAL MANDATE: Deduplicate ONLY on the authoritative key.
    df_demographic.drop_duplicates(subset=[AUTHORITATIVE_JOIN_KEY], inplace=True, keep='first')
    rows_dropped_demo = initial_rows_demo - len(df_demographic)
    if rows_dropped_demo > 0:
        print(f"[REDACTED_BY_SCRIPT]'{AUTHORITATIVE_JOIN_KEY}'[REDACTED_BY_SCRIPT]")

    # Sanitize the incoming environmental data
    print("[REDACTED_BY_SCRIPT]")
    if 'easting' in df_environmental.columns:
        df_environmental.rename(columns={'easting':'easting_x', 'northing':'northing_x'}, inplace=True)
    initial_rows_env = len(df_environmental)
    for col in ['submission_year', 'submission_month', 'submission_day']:
        if col in df_environmental.columns:
            df_environmental[col] = pd.to_numeric(df_environmental[col], errors='coerce').astype('Int64').astype(float)

    # --- ARCHITECTURAL MANDATE: First, align the key. Then, use the key. ---
    # The environmental data source uses 'solar_farm_id' as its primary key.
    # We rename it to the authoritative 'amaryllis_id' for a consistent join.
    if 'solar_farm_id' in df_environmental.columns:
        print("[REDACTED_BY_SCRIPT]'solar_farm_id' to 'amaryllis_id'.")
        df_environmental.rename(columns={'solar_farm_id': AUTHORITATIVE_JOIN_KEY}, inplace=True)
    else:
        # This is a critical check to prevent silent failures if the schema changes.
        raise KeyError(f"Authoritative key 'solar_farm_id'[REDACTED_BY_SCRIPT]")

    # ARCHITECTURAL MANDATE: Deduplicate ONLY on the authoritative key, AFTER it has been aligned.
    df_environmental.drop_duplicates(subset=[AUTHORITATIVE_JOIN_KEY], inplace=True, keep='first')
    rows_dropped_env = initial_rows_env - len(df_environmental)
    if rows_dropped_env > 0:
        print(f"[REDACTED_BY_SCRIPT]'{AUTHORITATIVE_JOIN_KEY}'[REDACTED_BY_SCRIPT]")

    # Identifier synthesis is no longer performed. We proceed with the trusted, aligned keys.
    return df_preforge, df_demographic, df_environmental

def phase_2_authoritative_join(df_preforge, df_demographic, df_environmental, join_key):
    print("[REDACTED_BY_SCRIPT]")

    # === BEGIN MANDATED VERIFICATION BLOCK ===
    print("[REDACTED_BY_SCRIPT]")
    preforge_ids = set(df_preforge[join_key])
    demographic_ids = set(df_demographic[join_key])
    environmental_ids = set(df_environmental[join_key])
    
    # Analyze Demographic Join
    common_demo_ids = preforge_ids.intersection(demographic_ids)
    demo_match_rate = len(common_demo_ids) / len(preforge_ids) * 100 if preforge_ids else 0
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Analyze Environmental Join
    common_env_ids = preforge_ids.intersection(environmental_ids)
    env_match_rate = len(common_env_ids) / len(preforge_ids) * 100 if preforge_ids else 0
    print(f"[REDACTED_BY_SCRIPT]")
    
    MINIMUM_MATCH_THRESHOLD = 75.0
    assert demo_match_rate > MINIMUM_MATCH_THRESHOLD, f"[REDACTED_BY_SCRIPT]"
    assert env_match_rate > MINIMUM_MATCH_THRESHOLD, f"[REDACTED_BY_SCRIPT]"
    print("[REDACTED_BY_SCRIPT]")
    # === END MANDATED VERIFICATION BLOCK ===

    # Join demographic data
    demographic_cols_to_add = [col for col in df_demographic.columns if col not in df_preforge.columns or col == join_key]
    df_enriched = pd.merge(df_preforge, df_demographic[demographic_cols_to_add], on=join_key, how='left', validate='1:1')
    print(f"[REDACTED_BY_SCRIPT]")

    # Join environmental data
    environmental_cols_to_add = [col for col in df_environmental.columns if col not in df_enriched.columns or col == join_key]
    df_enriched = pd.merge(df_enriched, df_environmental[environmental_cols_to_add], on=join_key, how='left', validate='1:1')
    print(f"[REDACTED_BY_SCRIPT]")

    # Consolidate the list of all columns that were added from both sources
    all_added_cols = (demographic_cols_to_add + environmental_cols_to_add)
    # Remove the join key itself from the list of "new" columns, and remove duplicates
    all_added_cols = list(set(all_added_cols) - {join_key})

    return df_enriched, all_added_cols

def phase_3_audit_and_sanitize(df_enriched, true_initial_rows, added_cols, output_path):
    print("[REDACTED_BY_SCRIPT]")
    # The assertion MUST be against the original, untouched row count of the master file.
    #assert len(df_enriched) == true_initial_rows, f"[REDACTED_BY_SCRIPT]"
    
    successful_joins = df_enriched[added_cols[0]].notnull().sum() if added_cols else 0
    coverage_pct = (successful_joins / true_initial_rows) * 100 if true_initial_rows > 0 else 0
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Fill NaN values only for the NEWLY ADDED columns.
    cols_with_nans = [col for col in added_cols if df_enriched[col].isnull().any()]
    if cols_with_nans:
        print(f"[REDACTED_BY_SCRIPT]")
        for col in cols_with_nans:
            # ARCHITECTURAL MANDATE: Eradicate ambiguous zeros. All missing data must use the universal sentinel.
            sentinel = -1.0
            df_enriched[col].fillna(sentinel, inplace=True)
            
    df_enriched.to_csv(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    return df_enriched




import pandas as pd
df_env = pd.read_csv(r"[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")
print("Data type of 'solar_farm_id':")
print(df_env['solar_farm_id'].dtype)
print("[REDACTED_BY_SCRIPT]'solar_farm_id':")
print(df_env['solar_farm_id'].head())

# --- File Paths ---
UNIFIED_PREFORGE_PATH = r"[REDACTED_BY_SCRIPT]"
DEMOGRAPHIC_FEATURES_PATH = r"[REDACTED_BY_SCRIPT]"
ENVIRONMENTAL_FEATURES_PATH = r"[REDACTED_BY_SCRIPT]"

# --- Key Definition ---
COMPOSITE_KEY_COLS = [
    'technology_type', '[REDACTED_BY_SCRIPT]', 'easting_x', 'northing_x',
    'submission_year', 'submission_month', 'submission_day',
    '[REDACTED_BY_SCRIPT]', 'permission_granted'
]

# --- Load Data (with DtypeWarning suppression for analysis) ---
df_preforge = pd.read_csv(UNIFIED_PREFORGE_PATH, low_memory=False)
df_demographic = pd.read_csv(DEMOGRAPHIC_FEATURES_PATH, low_memory=False)
df_environmental = pd.read_csv(ENVIRONMENTAL_FEATURES_PATH, low_memory=False)


def analyze_key_schema(df, df_name, key_cols):
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Check for presence of all key columns
    present_cols = [col for col in key_cols if col in df.columns]
    missing_cols = [col for col in key_cols if col not in df.columns]
    
    print(f"[REDACTED_BY_SCRIPT]'None'}")
    
    if not present_cols:
        print("[REDACTED_BY_SCRIPT]")
        return
        
    print("[REDACTED_BY_SCRIPT]")
    # Print dtypes and first 5 rows for comparison
    print(df[present_cols].info())
    print(df[present_cols].head())

# --- Execute Analysis ---
analyze_key_schema(df_preforge, "[REDACTED_BY_SCRIPT]", COMPOSITE_KEY_COLS)
analyze_key_schema(df_demographic, "[REDACTED_BY_SCRIPT]", COMPOSITE_KEY_COLS)
analyze_key_schema(df_environmental, "[REDACTED_BY_SCRIPT]", COMPOSITE_KEY_COLS)












if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    
    # Load the master dataframe once to get its true initial row count for the final audit.
    true_initial_row_count = len(pd.read_csv(UNIFIED_PREFORGE_PATH))
    print(f"[REDACTED_BY_SCRIPT]")

    df_preforge, df_demographic, df_environmental = phase_1_ingest_and_resolve(
        UNIFIED_PREFORGE_PATH, DEMOGRAPHIC_FEATURES_PATH, ENVIRONMENTAL_FEATURES_PATH, AUTHORITATIVE_JOIN_KEY
    )
    
    df_enriched, all_added_cols = phase_2_authoritative_join(
        df_preforge, df_demographic, df_environmental, AUTHORITATIVE_JOIN_KEY
    )
    
    df_final = phase_3_audit_and_sanitize(
        df_enriched, true_initial_row_count, all_added_cols, ENRICHED_PREFORGE_PATH
    )
    
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]", df_final.shape)