import pandas as pd

# --- Artifact & Project Constants ---
L47_CONTAMINATED_PATH =  r"[REDACTED_BY_SCRIPT]"         #r"[REDACTED_BY_SCRIPT]"
L47_STERILIZED_PATH = r"[REDACTED_BY_SCRIPT]"

# The Purge List: A Non-Negotiable Edict from Directive 050
PURGE_LIST = [
    # Category 1: Direct Outcome Dates
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    # Category 2: Post-Decision Status Indicators
    '[REDACTED_BY_SCRIPT]',
    # Category 3: Post-Decision Technical & Chronological Data
    'datecommissioned',
    'assessmentdate',
    'time_since_last_assessment_days'
]

def sterilize_feature_set(input_path, output_path, purge_list):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    df = pd.read_csv(input_path, low_memory=False)
    original_column_count = df.shape[1]
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Execute Purge
    df_sterilized = df.drop(columns=purge_list, errors='ignore')
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Verify Purge
    final_column_count = df_sterilized.shape[1]
    num_dropped = original_column_count - final_column_count
    
    # Assertion 1: Check if the correct number of columns were dropped.
    # Note: We check against the number of columns *actually present* in the dataframe to avoid errors if a column was already missing.
    present_purge_cols = [col for col in purge_list if col in df.columns]
    assert len(present_purge_cols) == num_dropped, \
        f"[REDACTED_BY_SCRIPT]"

    # Assertion 2: Check for any remaining contaminants.
    remaining_contaminants = set(df_sterilized.columns).intersection(set(purge_list))
    assert not remaining_contaminants, \
        f"[REDACTED_BY_SCRIPT]"
        
    print("[REDACTED_BY_SCRIPT]")
    
    # Persist Sterilized Artifact
    df_sterilized.to_csv(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    sterilize_feature_set(L47_CONTAMINATED_PATH, L47_STERILIZED_PATH, PURGE_LIST)
    print("[REDACTED_BY_SCRIPT]")