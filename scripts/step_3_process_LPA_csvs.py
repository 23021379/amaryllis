import pandas as pd
import numpy as np
import logging
import sys
import re
from functools import reduce

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input file paths (placeholders to be replaced by user)
L2_DATA_PATH = r'[REDACTED_BY_SCRIPT]'
PS1_PATH = r'[REDACTED_BY_SCRIPT]'
PS2_PATH = r'[REDACTED_BY_SCRIPT]'
CPS1_PATH = r"[REDACTED_BY_SCRIPT]"
CPS2_PATH = r'[REDACTED_BY_SCRIPT]'

# Output file path
L3_DATA_PATH = r'[REDACTED_BY_SCRIPT]'

def clean_lpa_name(series: pd.Series) -> pd.Series:
    """[REDACTED_BY_SCRIPT]"""
    return series.astype(str).str.lower().str.strip().str.replace(r'\s*\(.*\)\s*', '', regex=True).str.strip()

def calculate_trend(group):
    """[REDACTED_BY_SCRIPT]"""
    if len(group) < 3:
        return np.nan
    # Drop NaNs for robust regression
    group = group.dropna(subset=['year', 'workload'])
    if len(group) < 3:
        return np.nan
    
    # Simple linear regression: y = workload, x = year
    x = group['year']
    y = group['workload']
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope

def find_column_by_pattern(df: pd.DataFrame, pattern: str) -> str:
    """
    MANDATE 4.9: Finds a single column by matching a regex pattern against a
    semantically normalized version of the column names.
    """
    matches = []
    for col in df.columns:
        # Normalize the column name for matching: lowercase, replace non-alphanum with '_', collapse '_'
        normalized_col = re.sub(r'[^a-z0-9]+', '_', col.lower()).strip('_')
        
        if re.search(pattern, normalized_col):
            matches.append(col) # Append the ORIGINAL column name
            
    if len(matches) == 0:
        raise KeyError(f"[REDACTED_BY_SCRIPT]'{pattern}'.")
    if len(matches) > 1:
        raise ValueError(f"[REDACTED_BY_SCRIPT]'{pattern}': {matches}")
    return matches[0]

def safe_division(numerator, denominator):
    """[REDACTED_BY_SCRIPT]"""
    return np.divide(numerator.values, denominator.values, out=np.full_like(numerator.values, np.nan, dtype=float), where=(denominator.values!=0))

# --- MODULES ---

def process_ps1_workload(ps1_path: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(ps1_path, encoding='latin-1', dtype=str)
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False)
    df.replace('..', np.nan, inplace=True)
    
    try:
        lpa_name_handle = find_column_by_pattern(df, r'^lpanm$')
        received_handle = find_column_by_pattern(df, r'[REDACTED_BY_SCRIPT]')
        withdrawn_handle = find_column_by_pattern(df, r'[REDACTED_BY_SCRIPT]')
        year_handle = find_column_by_pattern(df, r'^f_year$')
        
        df['lpanm_clean'] = clean_lpa_name(df[lpa_name_handle])
        
        enforcement_keywords = ['enforcement', 'breach', 'stop', 'contravention']
        enforcement_handles = [col for col in df.columns if any(key in col for key in enforcement_keywords)]
    except (KeyError, ValueError) as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        raise
        
    # Select ALL required columns
    required_cols = ['lpanm_clean', year_handle, received_handle, withdrawn_handle] + enforcement_handles
    df_out = df[required_cols].copy()
    
    # Rename to the CANONICAL SCHEMA
    df_out.rename(columns={
        year_handle: 'year', 
        received_handle: 'received',
        withdrawn_handle: 'withdrawn'
    }, inplace=True)
    return df_out

def process_cps1_workload(cps1_path: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(cps1_path, encoding='latin-1', dtype=str)
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False)
    df.replace('..', np.nan, inplace=True)

    try:
        lpa_name_handle = find_column_by_pattern(df, r'^lpanm$')
        received_handle = find_column_by_pattern(df, r'[REDACTED_BY_SCRIPT]')
        withdrawn_handle = find_column_by_pattern(df, r'[REDACTED_BY_SCRIPT]')
        quarter_handle = find_column_by_pattern(df, r'^quarter$')
        
        df['lpanm_clean'] = clean_lpa_name(df[lpa_name_handle])
        
        enforcement_keywords = ['enforcement', 'breach', 'stop', 'contravention']
        enforcement_handles = [col for col in df.columns if any(key in col for key in enforcement_keywords)]
    except (KeyError, ValueError) as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        raise
    
    # Bespoke feature engineering: Derive year from quarter
    df['year'] = df[quarter_handle].str.split(' ').str[0]

    df_out = df[['lpanm_clean', 'year', received_handle]].copy()
    df_out.rename(columns={received_handle: 'workload'}, inplace=True)
    return df_out


def process_decision_data(ps2_path: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    df_ps2 = pd.read_csv(ps2_path, encoding='latin-1', dtype=str)
    df_ps2.replace('..', np.nan, inplace=True)

    try:
        # Resolve the LPA name column itself into a handle.
        lpa_name_handle = find_column_by_pattern(df_ps2, r'^lpanm$')
        
        # Define patterns for all required columns
        patterns = {
            'major_comm_granted': [r'[REDACTED_BY_SCRIPT]', r'[REDACTED_BY_SCRIPT]', r'[REDACTED_BY_SCRIPT]'],
            'major_comm_decisions': [r'[REDACTED_BY_SCRIPT]', r'[REDACTED_BY_SCRIPT]', r'[REDACTED_BY_SCRIPT]'],
            'major_in_time': [r'[REDACTED_BY_SCRIPT]'],
            '[REDACTED_BY_SCRIPT]': [r'[REDACTED_BY_SCRIPT]']
        }
        
        # Find all columns, storing them in a dictionary
        found_cols = {key: [find_column_by_pattern(df_ps2, p) for p in pat_list] for key, pat_list in patterns.items()}

        # Use the handle, not a hard-coded string, for cleaning.
        df_ps2['lpanm_clean'] = clean_lpa_name(df_ps2[lpa_name_handle])
        
        # Coerce all found columns to numeric
        all_cols_to_process = [col for sublist in found_cols.values() for col in sublist]
        for col in all_cols_to_process:
            df_ps2[col] = pd.to_numeric(df_ps2[col], errors='coerce')

        # Create composite helper columns
        df_ps2['[REDACTED_BY_SCRIPT]'] = df_ps2[found_cols['major_comm_granted']].sum(axis=1)
        df_ps2['[REDACTED_BY_SCRIPT]'] = df_ps2[found_cols['major_comm_decisions']].sum(axis=1)
        df_ps2['total_major_in_time'] = df_ps2[found_cols['major_in_time']].sum(axis=1)
        
        # Aggregate the composite and total columns
        lpa_profile = df_ps2.groupby('lpanm_clean').agg(
            [REDACTED_BY_SCRIPT]=('[REDACTED_BY_SCRIPT]', 'sum'),
            [REDACTED_BY_SCRIPT]=('[REDACTED_BY_SCRIPT]', 'sum'),
            total_major_in_time=('total_major_in_time', 'sum'),
            [REDACTED_BY_SCRIPT]_for_compliance=(found_cols['[REDACTED_BY_SCRIPT]'][0], 'sum')
        )

        # Calculate final features
        lpa_profile['lpa_major_commercial_approval_rate'] = safe_division(
            lpa_profile['[REDACTED_BY_SCRIPT]'], lpa_profile['[REDACTED_BY_SCRIPT]']
        )
        lpa_profile['[REDACTED_BY_SCRIPT]'] = safe_division(
            lpa_profile['total_major_in_time'], lpa_profile['[REDACTED_BY_SCRIPT]']
        )
        logging.info("[REDACTED_BY_SCRIPT]")
        return lpa_profile[['lpa_major_commercial_approval_rate', '[REDACTED_BY_SCRIPT]']].reset_index()

    except (KeyError, ValueError) as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        raise


def process_transactional_data(cps2_path: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    df_cps2 = pd.read_csv(cps2_path, encoding='latin-1', dtype=str)
    
    df_cps2.replace('..', np.nan, inplace=True)
    
    try:
        lpa_name_handle = find_column_by_pattern(df_cps2, r'^lpanm$')
        granted_col = find_column_by_pattern(df_cps2, r'[REDACTED_BY_SCRIPT]')
        received_col = find_column_by_pattern(df_cps2, r'^date_received$')
        dispatched_col = find_column_by_pattern(df_cps2, r'^date_dispatched$')
        logging.info("[REDACTED_BY_SCRIPT]")
    except (KeyError, ValueError) as e:
        logging.error(e)
        raise

    df_cps2['lpanm_clean'] = clean_lpa_name(df_cps2[lpa_name_handle])
    

    df_cps2['is_granted'] = df_cps2[granted_col].map({'Granted': 1, 'Refused': 0})
    df_cps2[received_col] = pd.to_datetime(df_cps2[received_col], errors='coerce', dayfirst=True)
    df_cps2[dispatched_col] = pd.to_datetime(df_cps2[dispatched_col], errors='coerce', dayfirst=True)
    df_cps2['decision_days'] = (df_cps2[dispatched_col] - df_cps2[received_col]).dt.days
    
    df_cps2 = df_cps2[(df_cps2['decision_days'] >= 0) & (df_cps2['is_granted'].notna())]

    # Main aggregation for standard metrics
    lpa_profile = df_cps2.groupby('lpanm_clean').agg(
        lpa_approval_rate_cps2=('is_granted', 'mean'),
        lpa_avg_decision_days=('decision_days', 'mean'),
        lpa_decision_speed_variance=('decision_days', 'std'),
        lpa_p90_decision_days=('decision_days', lambda x: x.quantile(0.9))
    )
    
    # Mandate 5.3: Cross-validate with industrial-specific approval rate
    try:
        scheme_type_col = find_column_by_pattern(df_cps2, r'^type_of_scheme$')
        industrial_apps = df_cps2[df_cps2[scheme_type_col].str.contains('Minerals|Waste', case=False, na=False)]
        if not industrial_apps.empty:
            industrial_approval_rate = industrial_apps.groupby('lpanm_clean')['is_granted'].mean().rename('[REDACTED_BY_SCRIPT]')
            lpa_profile = lpa_profile.merge(industrial_approval_rate, on='lpanm_clean', how='left')
    except KeyError:
        logging.warning("Could not find 'type_of_scheme'[REDACTED_BY_SCRIPT]")

    # Mandate 4.3: Calculate grant/refuse speed ratio separately for robustness
    speed_by_outcome = df_cps2.groupby(['lpanm_clean', 'is_granted'])['decision_days'].mean().unstack()
    speed_by_outcome.columns = ['refused_speed', 'granted_speed']
    
    if 'refused_speed' in speed_by_outcome.columns and 'granted_speed' in speed_by_outcome.columns:
        speed_by_outcome['[REDACTED_BY_SCRIPT]'] = safe_division(
            speed_by_outcome['granted_speed'], 
            speed_by_outcome['refused_speed']
        )
        lpa_profile = lpa_profile.merge(
            speed_by_outcome[['[REDACTED_BY_SCRIPT]']], 
            on='lpanm_clean', 
            how='left'
        )
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return lpa_profile.reset_index()

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    try:
        # Execute bespoke, file-specific modules
        ps1_data = process_ps1_workload(PS1_PATH)
        cps1_data = process_cps1_workload(CPS1_PATH)
        
        # Concatenate standardized outputs
        df_full_workload = pd.concat([ps1_data, cps1_data], ignore_index=True)

        # The bespoke modules now guarantee a CANONICAL schema.
        # The main pipeline can now be simple, robust, and use hard-coded canonical names.
        
        # Convert all potentially numeric columns to numeric, coercing errors
        cols_to_convert = ['year', 'received', 'withdrawn']
        enforcement_keywords = ['enforcement', 'breach', 'stop', 'contravention']
        enforcement_cols = [col for col in df_full_workload.columns if any(key in col for key in enforcement_keywords)]
        cols_to_convert.extend(enforcement_cols)

        for col in cols_to_convert:
            if col in df_full_workload.columns:
                df_full_workload[col] = pd.to_numeric(df_full_workload[col], errors='coerce')

        df_full_workload['[REDACTED_BY_SCRIPT]'] = df_full_workload[enforcement_cols].sum(axis=1)

        # Use the simple, guaranteed CANONICAL names in the aggregation
        workload_base_agg = df_full_workload.groupby('lpanm_clean').agg(
            total_applications_received=('received', 'sum'),
            total_applications_withdrawn=('withdrawn', 'sum'),
            [REDACTED_BY_SCRIPT]=('[REDACTED_BY_SCRIPT]', 'sum')
        )
        workload_base_agg['lpa_withdrawal_rate'] = safe_division(
            workload_base_agg['[REDACTED_BY_SCRIPT]'], workload_base_agg['[REDACTED_BY_SCRIPT]']
        )

        # Perform yearly aggregations using canonical names
        yearly_workload = df_full_workload.groupby(['lpanm_clean', 'year'])['received'].sum().reset_index()
        yearly_workload.rename(columns={'received': 'workload'}, inplace=True) # Standardize for trend func

        workload_dynamics_agg = yearly_workload.groupby('lpanm_clean')['workload'].agg(
            lpa_avg_yearly_workload='mean',
            lpa_workload_volatility=lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan,
            lpa_total_experience='sum'
        )
        workload_trend = yearly_workload.groupby('lpanm_clean').apply(calculate_trend).rename('lpa_workload_trend')
        
        # Merge all workload features
        workload_profile = reduce(lambda left, right: pd.merge(left, right, on='lpanm_clean', how='outer'), 
                                  [workload_base_agg, workload_dynamics_agg, workload_trend.reset_index()])
        
        workload_profile['[REDACTED_BY_SCRIPT]'] = safe_division(
            workload_profile['[REDACTED_BY_SCRIPT]'], workload_profile['lpa_total_experience']
        )
        logging.info(f"[REDACTED_BY_SCRIPT]")


        leniency_profile = process_decision_data(PS2_PATH)
        transactional_profile = process_transactional_data(CPS2_PATH)
    except (FileNotFoundError, KeyError, ValueError) as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)
    
    data_frames = [workload_profile, leniency_profile, transactional_profile]
    lpa_master_profile = reduce(lambda left, right: pd.merge(left, right, on='lpanm_clean', how='outer'), data_frames)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    df_main = pd.read_csv(L2_DATA_PATH)
    df_main['lpanm_clean'] = clean_lpa_name(df_main['planning_authority'])
    df_merged = pd.merge(df_main, lpa_master_profile, on='lpanm_clean', how='left')
    
    # Final Triage
    df_merged = df_merged.drop(columns=['lpanm_clean', 'planning_authority'])
    
    null_percentages = df_merged.isnull().mean()
    cols_to_drop = null_percentages[null_percentages > 0.9].index
    if not cols_to_drop.empty:
        df_merged = df_merged.drop(columns=cols_to_drop)
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    numeric_cols = df_merged.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df_merged[col].isnull().any():
            df_merged[col] = df_merged[col].fillna(-1)

    if df_merged.select_dtypes(exclude=np.number).shape[1] > 0:
        non_numeric = df_merged.select_dtypes(exclude=np.number).columns.tolist()
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)
            
    df_merged.to_csv(L3_DATA_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()