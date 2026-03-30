import pandas as pd
import numpy as np
import os

# --- Artifact & Project Constants ---
UKPN_STRATIFIED_PATH = r"[REDACTED_BY_SCRIPT]"
NGED_STRATIFIED_PATH = r"[REDACTED_BY_SCRIPT]"

# Mandate: The output is a single, unified artifact ready for the Forge Imputation Engine.
UNIFIED_PREFORGE_PATH = r"[REDACTED_BY_SCRIPT]"

# Sanctioned sentinel value for known intelligence gaps.
SENTINEL_VALUE = -1.0

def phase_1_ingest_and_audit(ukpn_path, nged_path):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    df_ukpn = pd.read_csv(ukpn_path)
    df_nged = pd.read_csv(nged_path)
    
    ukpn_cols = set(df_ukpn.columns)
    nged_cols = set(df_nged.columns)
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    return df_ukpn, df_nged, ukpn_cols, nged_cols

def phase_2_identify_cohorts(ukpn_cols, nged_cols):
    """
    Architectural Mandate: Dynamically identify feature cohorts using set operations.
    This is the core of the generalizable, non-hardcoded approach.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # The Intelligence Gap: Features in UKPN that are missing from NGED.
    ukpn_exclusive_cols = ukpn_cols - nged_cols
    
    # The Anomaly Cohort: Features in NGED that are not in the UKPN gold standard.
    nged_exclusive_cols = nged_cols - ukpn_cols
    
    # The Stable Spine: Features common to both datasets.
    common_cols = ukpn_cols.intersection(nged_cols)
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    return ukpn_exclusive_cols, nged_exclusive_cols

def phase_3_fill_nan_with_sentinel(df_ukpn, df_nged):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Make copies to avoid SettingWithCopyWarning
    df_ukpn_filled = df_ukpn.copy()
    df_nged_filled = df_nged.copy()

    df_ukpn_filled.fillna(SENTINEL_VALUE, inplace=True)
    df_nged_filled.fillna(SENTINEL_VALUE, inplace=True)
    
    print("[REDACTED_BY_SCRIPT]")
    return df_ukpn_filled, df_nged_filled

def phase_4_schema_expansion(df_nged, ukpn_exclusive_cols, nged_exclusive_cols):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Anomaly Quarantine
    if nged_exclusive_cols:
        print(f"[REDACTED_BY_SCRIPT]")
        print("    Anomalies:", sorted(list(nged_exclusive_cols)))
        # df_nged = df_nged.drop(columns=list(nged_exclusive_cols))
    
    # Schema Expansion
    print(f"[REDACTED_BY_SCRIPT]")
    df_nged_expanded = df_nged.copy()
    for col in sorted(list(ukpn_exclusive_cols)):
        # CRITICAL FIX: Create the new column and assign the sentinel value.
        # This correctly adds the missing UKPN-exclusive columns to the NGED dataframe.
        df_nged_expanded[col] = SENTINEL_VALUE
        
        
    return df_nged_expanded

def phase_5_unify_and_verify(df_ukpn, df_nged_expanded, output_path):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Enforce identical column order to prevent concatenation errors.
    df_nged_final = df_nged_expanded[df_ukpn.columns]
    
    # Store initial row counts for verification
    initial_ukpn_rows = len(df_ukpn)
    initial_nged_rows = len(df_nged_final)
    
    # The Final Unification
    df_unified = pd.concat([df_ukpn, df_nged_final], ignore_index=True)
    
    # --- Paranoid Verification Protocol ---
    print("[REDACTED_BY_SCRIPT]")
    # 1. Row count integrity
    assert len(df_unified) == initial_ukpn_rows + initial_nged_rows, "[REDACTED_BY_SCRIPT]"
    # 2. Column count integrity
    assert len(df_unified.columns) == len(df_ukpn.columns), "[REDACTED_BY_SCRIPT]"
    # 3. Sentinel value integrity
    nged_subset = df_unified[df_unified['dno_source'] == 'nged']
    missing_cols = set(df_ukpn.columns) - set(pd.read_csv(NGED_STRATIFIED_PATH).columns) # Re-calc for safety
    for col in missing_cols:
        if col in nged_subset.columns:
            # Check if all values in the column for the NGED subset are the sentinel value
            is_sentinel = (nged_subset[col] == SENTINEL_VALUE)
            assert is_sentinel.all(), f"[REDACTED_BY_SCRIPT]'{col}'."
    
    # 4. Primary Key Uniqueness
    # ARCHITECTURAL MANDATE: Final firewall against upstream duplication.
    if 'application_id' in df_unified.columns:
        assert df_unified['application_id'].is_unique, "FATAL: Primary key 'application_id'[REDACTED_BY_SCRIPT]"

    print("[REDACTED_BY_SCRIPT]")
    
    # Export the final artifact
    df_unified.to_csv(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    
    return df_unified

if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    
    # Phase 1: Ingest artifacts and perform the programmatic schema audit.
    df_ukpn, df_nged, ukpn_cols, nged_cols = phase_1_ingest_and_audit(UKPN_STRATIFIED_PATH, NGED_STRATIFIED_PATH)
    
    # Phase 2: Dynamically identify the feature cohorts.
    ukpn_exclusive_cols, nged_exclusive_cols = phase_2_identify_cohorts(ukpn_cols, nged_cols)
    
    # Phase 3: Fill all existing NaN values with the sentinel value.
    df_ukpn, df_nged = phase_3_fill_nan_with_sentinel(df_ukpn, df_nged)
    
    # Phase 4: Expand the NGED schema to match the gold standard.
    df_nged_expanded = phase_4_schema_expansion(df_nged, ukpn_exclusive_cols, nged_exclusive_cols)
    
    # Phase 5: Unify the two dataframes and run final verification.
    df_unified = phase_5_unify_and_verify(df_ukpn, df_nged_expanded, UNIFIED_PREFORGE_PATH)
    
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]", df_unified.shape)
    print("[REDACTED_BY_SCRIPT]")
    print(df_unified['dno_source'].value_counts())