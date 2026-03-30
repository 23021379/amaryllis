import pandas as pd
import numpy as np
import json
import logging
import sys

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
SOURCE_L1_CSV = r'[REDACTED_BY_SCRIPT]'
OUTPUT_L2_CSV = r'[REDACTED_BY_SCRIPT]'
OUTPUT_MAPPINGS_JSON = r'[REDACTED_BY_SCRIPT]'

def load_and_prepare_data(source_path: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    try:
        df = pd.read_csv(source_path, low_memory=False)
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]'{source_path}'. Terminating.")
        sys.exit(1)

    # Coerce all potential date columns to datetime objects for subsequent operations.
    # This list is comprehensive based on the L1 schema.
    date_cols = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'appeal_lodged', 'appeal_withdrawn', 'appeal_refused', 'appeal_granted',
        '[REDACTED_BY_SCRIPT]', 'sos_intervened', 'sos_refusal', 'sos_granted',
        '[REDACTED_BY_SCRIPT]', 'under_construction', 'operational'
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df, date_cols

def generate_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]'permission_granted' and 'planning_duration_days'.")
    
    # Action 1: Establish the "[REDACTED_BY_SCRIPT]"
    df['[REDACTED_BY_SCRIPT]'] = df['appeal_granted'].fillna(df['[REDACTED_BY_SCRIPT]'])
    
    # Action 2: Create Classification Target `permission_granted`
    df['permission_granted'] = np.where(df['[REDACTED_BY_SCRIPT]'].notna(), 1, 0)
    
    # Action 3: Create Regression Target `planning_duration_days`
    # This will correctly result in NaT/NaN where the subtraction is not possible.
    time_delta = df['[REDACTED_BY_SCRIPT]'] - df['[REDACTED_BY_SCRIPT]']
    df['[REDACTED_BY_SCRIPT]'] = time_delta.dt.days
    
    logging.info(f"Target 'permission_granted'[REDACTED_BY_SCRIPT]'permission_granted'].mean():.2%}")
    logging.info(f"Target 'planning_duration_days'[REDACTED_BY_SCRIPT]'planning_duration_days'].mean():.0f} days.")
    
    return df

def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]'planning_application_submitted' and 'planning_permission_granted'.")
    
    # Submission date features
    submission_date = df['[REDACTED_BY_SCRIPT]']
    df['submission_year'] = submission_date.dt.year
    df['submission_month'] = submission_date.dt.month
    df['submission_day'] = submission_date.dt.day
    
    # Cyclical encoding for submission month
    df['submission_month_sin'] = np.sin(2 * np.pi * df['submission_month'] / 12)
    df['submission_month_cos'] = np.cos(2 * np.pi * df['submission_month'] / 12)
    
    # Granted date features
    granted_date = df['[REDACTED_BY_SCRIPT]']
    df['[REDACTED_BY_SCRIPT]'] = granted_date.dt.year
    df['[REDACTED_BY_SCRIPT]'] = granted_date.dt.month
    df['[REDACTED_BY_SCRIPT]'] = granted_date.dt.day
    
    return df

def triage_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Action 1: Column Culling
    initial_cols = df.shape[1]
    null_percentages = df.isnull().mean()
    cols_to_drop = null_percentages[null_percentages > 0.5].index
    if 'solar_site_area_sqm' in cols_to_drop:
        cols_to_drop = cols_to_drop.drop('solar_site_area_sqm') # CRITICAL: Retain this column if present
    if '[REDACTED_BY_SCRIPT]' in cols_to_drop:
        cols_to_drop = cols_to_drop.drop('[REDACTED_BY_SCRIPT]') # CRITICAL: Retain this column if present
    
    if not cols_to_drop.empty:
        df = df.drop(columns=cols_to_drop)
        logging.warning(f"[REDACTED_BY_SCRIPT]")
    
    # Action 2: Numeric Imputation (excluding the regression target)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if '[REDACTED_BY_SCRIPT]' in numeric_cols:
        numeric_cols.remove('[REDACTED_BY_SCRIPT]') # CRITICAL: Do not impute the target
    
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(-1)
            # Ensure integer columns remain integer after fillna if possible
            if pd.api.types.is_integer_dtype(df[col].dtype):
                 df[col] = df[col].astype(int)

    return df

def encode_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    categorical_cols = df.select_dtypes(include=['object']).columns
    master_mappings = {}

    for col in categorical_cols:
        # Impute missing categorical values with a standard 'missing' placeholder
        df[col] = df[col].fillna('missing').astype('category')
        
        # Create the correct {category: code} mapping for encoding future data
        master_mappings[col] = {cat: code for code, cat in enumerate(df[col].cat.categories)}
        
        # Apply the codes to the dataframe column
        df[col] = df[col].cat.codes

    with open(OUTPUT_MAPPINGS_JSON, 'w') as f:
        json.dump(master_mappings, f, indent=4)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return df

def const_imputation(series: pd.Series, constant: int = -1) -> pd.Series:
    """[REDACTED_BY_SCRIPT]"""
    return series.fillna(constant)

def clean_lpa_name(series: pd.Series) -> pd.Series:
    """[REDACTED_BY_SCRIPT]"""
    return series.astype(str).str.lower().str.strip().str.replace(r'\s*\(.*\)\s*', '', regex=True).str.strip()

def main():
    """[REDACTED_BY_SCRIPT]"""
    df, date_cols_to_drop = load_and_prepare_data(SOURCE_L1_CSV)
    df_ind=df['planning_authority']
    df.drop(columns=['planning_authority'],inplace=True)
    df = generate_target_variables(df)
    df = engineer_date_features(df)
    df = triage_missing_values(df)
    df = encode_categorical_features(df)
    df = df.apply(const_imputation, constant=-1)

    # Final cleanup: Remove all original date columns and any helper columns
    logging.info("[REDACTED_BY_SCRIPT]")
    df = df.drop(columns=[col for col in date_cols_to_drop if col in df.columns], errors='ignore')
    df = df.drop(columns=['[REDACTED_BY_SCRIPT]'], errors='ignore')
    
    # Verification step
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)
    df['planning_authority']=df_ind
    df['planning_authority']=clean_lpa_name(df['planning_authority'])
    # Save the final ML-ready dataset
    df.to_csv(OUTPUT_L2_CSV, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()