import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# --- Artifact & Project Constants ---
L47_FINAL_DNO_PATH = r"[REDACTED_BY_SCRIPT]"
MODEL_READY_X_PATH = "[REDACTED_BY_SCRIPT]"
MODEL_READY_Y_CLASS_PATH = "[REDACTED_BY_SCRIPT]"
MODEL_READY_Y_REG_PATH = "[REDACTED_BY_SCRIPT]"
README_PATH = "[REDACTED_BY_SCRIPT]"

# Mandated thresholds for NaN triage
NAN_COL_THRESHOLD = 0.90
NAN_ROW_THRESHOLD = 0.40
HIGH_CARDINALITY_THRESHOLD = 100

def phase_1_2_triage_and_segregate(l47_path):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    df_raw = pd.read_csv(l47_path, low_memory=False)
    
    # Phase 1: Structural Triage
    df_cleaned = df_raw.loc[:, ~df_raw.columns.duplicated()]
    df_cleaned.set_index('amaryllis_id', inplace=True, drop=False)
    df_cleaned = df_cleaned.drop_duplicates(subset=['amaryllis_id'], keep='first')
    
    # Phase 2: Target Segregation
    y_class = df_cleaned['permission_granted']
    y_reg = df_cleaned['[REDACTED_BY_SCRIPT]']
    
    # ARCHITECTURAL MANDATE: Institute an Explicit Purge Manifest. Heuristic purging is forbidden.
    EXPLICIT_METADATA_PURGE_MANIFEST = [
        'solar_farm_id',
        'amaryllis_id',
        # All raw date columns are non-predictive; cyclical features are used instead.
        'submission_date',
        # All raw coordinate columns are non-predictive; spatial features are used instead.
        'easting_x',
        'northing_x',
        # Any other known non-predictive identifiers or names would be added here.
        # e.g., 'site_name', 'applicant_name', 'lpa_name'
    ]

    # Isolate and Purge Non-Predictive Metadata
    cols_to_drop = ['permission_granted', '[REDACTED_BY_SCRIPT]']
    
    # Filter the manifest to only include columns actually present in the dataframe
    purge_cols_present = [col for col in EXPLICIT_METADATA_PURGE_MANIFEST if col in df_cleaned.columns]
    
    metadata_cols = set(cols_to_drop + purge_cols_present)
    print(f"[REDACTED_BY_SCRIPT]")
    
    X = df_cleaned.drop(columns=list(metadata_cols), errors='ignore')
    
    print(f"[REDACTED_BY_SCRIPT]")
    return X, y_class, y_reg

def phase_3_handle_categoricals(X):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    cols_to_drop = []
    cols_to_encode = []
    
    for col in categorical_cols:
        if X[col].nunique() > HIGH_CARDINALITY_THRESHOLD:
            cols_to_drop.append(col)
        else:
            cols_to_encode.append(col)
            
    # Purge high-cardinality columns
    X.drop(columns=cols_to_drop, inplace=True)
    print(f"[REDACTED_BY_SCRIPT]")
    
    # One-hot encode low-cardinality columns
    if cols_to_encode:
        X = pd.get_dummies(X, columns=cols_to_encode, dummy_na=False, prefix=cols_to_encode)
        print(f"[REDACTED_BY_SCRIPT]")
    
    # Ensure all boolean-like columns are integers
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
        
    # CRITICAL: Sanitize infinite values produced by division-by-zero in feature engineering.
    # Replace them with NaN to be handled by the subsequent NaN triage and imputation steps.
    inf_count = np.isinf(X).sum().sum()
    if inf_count > 0:
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"[REDACTED_BY_SCRIPT]")
        
    return X

def phase_4_nan_triage(X, y_class, y_reg):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Column Purge
    nan_col_ratios = X.isnull().sum() / len(X)
    cols_to_drop = nan_col_ratios[nan_col_ratios > NAN_COL_THRESHOLD].index
    X.drop(columns=cols_to_drop, inplace=True)
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Row Purge
    nan_row_ratios = X.isnull().sum(axis=1) / X.shape[1]
    rows_to_drop_idx = nan_row_ratios[nan_row_ratios > NAN_ROW_THRESHOLD].index
    
    X.drop(index=rows_to_drop_idx, inplace=True)
    y_class = y_class.drop(index=rows_to_drop_idx)
    y_reg = y_reg.drop(index=rows_to_drop_idx)

    #replcae all '-1.0' with large number. when target is -1.0, it means it didnt get accepted, which we can treat as a massive planning duration.
    y_reg.replace(-1.0, 10000, inplace=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Verification
    assert len(X) == len(y_class) == len(y_reg), "[REDACTED_BY_SCRIPT]"
    
    return X, y_class, y_reg

def phase_5_persist_artifacts(X, y_class, y_reg):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Verification: Ensure data still contains NaNs as expected.
    assert X.isnull().sum().sum() > 0, "[REDACTED_BY_SCRIPT]"
    
    # Persist Final Artifacts
    output_dir = os.path.dirname(MODEL_READY_X_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    X.to_csv(MODEL_READY_X_PATH)
    y_class.to_csv(MODEL_READY_Y_CLASS_PATH, header=True)
    y_reg.to_csv(MODEL_READY_Y_REG_PATH, header=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    return X, y_class, y_reg

if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    
    X, y_class, y_reg = phase_1_2_triage_and_segregate(L47_FINAL_DNO_PATH)
    X = phase_3_handle_categoricals(X)
    X, y_class, y_reg = phase_4_nan_triage(X, y_class, y_reg)
    X_final, y_class_final, y_reg_final = phase_5_persist_artifacts(X, y_class, y_reg)
    
    print("[REDACTED_BY_SCRIPT]")