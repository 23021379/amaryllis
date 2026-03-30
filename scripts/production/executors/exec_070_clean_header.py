import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
LEGACY_SCHEMA_PATH = Path(r"[REDACTED_BY_SCRIPT]")

def derive_temporal_features(df):
    """
    Reconstructs cyclic temporal features required by the legacy model.
    """
    if 'submission_date' in df.columns:
        logging.info("[REDACTED_BY_SCRIPT]'submission_date'...")
        dt = pd.to_datetime(df['submission_date'], errors='coerce')
        df['submission_year'] = dt.dt.year
        df['submission_month'] = dt.dt.month
        df['submission_day'] = dt.dt.day
        # Cyclic encoding
        df['submission_month_sin'] = np.sin(2 * np.pi * df['submission_month'] / 12)
        df['submission_month_cos'] = np.cos(2 * np.pi * df['submission_month'] / 12)
    else:
        logging.warning("CRITICAL: 'submission_date'[REDACTED_BY_SCRIPT]")
    return df

def execute(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes schema cleaning and header alignment to match the legacy model expectations.
    
    This executor performs:
    1. Schema extraction from legacy model
    2. Creating amaryllis_id column (copy of hex_id for legacy compatibility)
    3. DNO-specific feature coalescing (expanded with new mappings)
    4. Temporal feature engineering
    5. LSOA composite indices from raw census data
    6. Strict schema enforcement and verification
    7. Adds hex_id back for pipeline tracking
    
    Args:
        master_df (pd.DataFrame): The dataframe from the previous pipeline step.
        
    Returns:
        pd.DataFrame: The cleaned and aligned dataframe with hex_id for pipeline tracking.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # ---------------------------------------------------------
    # 1. Schema Extraction (The Ground Truth)
    # ---------------------------------------------------------
    if not LEGACY_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"[REDACTED_BY_SCRIPT]")
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    # Read only the header to get the strict column order
    legacy_schema = pd.read_csv(LEGACY_SCHEMA_PATH, nrows=0).columns.tolist()
    
    df_new = master_df.copy()
    
    # ---------------------------------------------------------
    # 2. Create amaryllis_id (COPY, not rename!)
    # ---------------------------------------------------------
    # CRITICAL: The pipeline needs hex_id to remain. We create amaryllis_id as a COPY.
    if 'hex_id' in df_new.columns:
        df_new['amaryllis_id'] = df_new['hex_id']
        logging.info("Created 'amaryllis_id' as a copy of 'hex_id'[REDACTED_BY_SCRIPT]")
    else:
        logging.warning("'hex_id'[REDACTED_BY_SCRIPT]'amaryllis_id'.")
        
    # ---------------------------------------------------------
    # 3. Feature Coalescing (The Merge Logic) - EXPANDED
    # ---------------------------------------------------------
    # We must merge DNO-specific columns into the global legacy columns.
    # Logic: If dno_source is NG, take NG col; if UKPN, take UKPN col.
    
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Determine source masks
    if 'dno_source' in df_new.columns:
        is_ng = df_new['dno_source'].str.lower().str.contains('ng', na=False) | df_new['dno_source'].str.lower().str.contains('national', na=False)
        is_ukpn = df_new['dno_source'].str.lower().str.contains('ukpn', na=False)
    else:
        logging.warning("Feature 'dno_source'[REDACTED_BY_SCRIPT]")
        is_ng = pd.Series(False, index=df_new.index)
        is_ukpn = pd.Series(False, index=df_new.index)

    # Map: Legacy_Column -> (New_NG_Column, New_UKPN_Column)
    # EXPANDED based on user requirements
    coalesce_map = {
        # Substation Distances & Capacity
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]':   ('[REDACTED_BY_SCRIPT]', 'ukpn_tx_[REDACTED_BY_SCRIPT]'),
        'avg_total_kva_5nn':              ('ng_sub_avg_total_kva_5nn', 'ukpn_tx_avg_total_kva_5nn'),
        '[REDACTED_BY_SCRIPT]':       ('ng_sub_[REDACTED_BY_SCRIPT]', 'ukpn_tx_[REDACTED_BY_SCRIPT]'),
        
        # PRIMARY SUBSTATION (NEW - from user requirements)
        '[REDACTED_BY_SCRIPT]': ('ng_[REDACTED_BY_SCRIPT]', 'ukpn_psub_total_kva'),
        
        # Grid Structure (Poles/Towers)
        '[REDACTED_BY_SCRIPT]':     ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]':    ('dist_to_nearest_ng_tower', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        
        # OHL Sterilized Areas (NEW - from user requirements)
        '[REDACTED_BY_SCRIPT]': ('ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'),
        
        # Connection Path Intersections
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        
        # Grid Counts
        '[REDACTED_BY_SCRIPT]': ('count_ng_tower_2km', '[REDACTED_BY_SCRIPT]'),
        
        # LCT (Low Carbon Technology) - NEW with normalization note
        # Note: NG uses 5nn (nearest neighbors) while UKPN uses in_5km (radius) - may need distribution normalization
        'lct_total_connections_in_5km': ('ng_lct_avg_5nn_total_connections', '[REDACTED_BY_SCRIPT]'),
        
        # IDNO (NEW - from user requirements)
        '[REDACTED_BY_SCRIPT]': ('ng_idno_dist_to_nearest_m', '[REDACTED_BY_SCRIPT]'),
    }

    for target_col, (ng_col, ukpn_col) in coalesce_map.items():
        # Initialize target with NaN
        df_new[target_col] = np.nan
        
        # Fill NG data
        if ng_col in df_new.columns:
            df_new.loc[is_ng, target_col] = df_new.loc[is_ng, ng_col]
        
        # Fill UKPN data
        if ukpn_col in df_new.columns:
            df_new.loc[is_ukpn, target_col] = df_new.loc[is_ukpn, ukpn_col]

    # ---------------------------------------------------------
    # 4. Temporal Engineering & DNO Encoding
    # ---------------------------------------------------------
    df_new = derive_temporal_features(df_new)
    
    # One-Hot DNO Source
    df_new['dno_source_nged'] = is_ng.astype(int)
    df_new['dno_source_ukpn'] = is_ukpn.astype(int)
    
    # ---------------------------------------------------------
    # 5. LSOA Composite Indices (Demographics Feature Engineering)
    # ---------------------------------------------------------
    # The model expects composite indices, not raw census data (ah4... columns)
    # We calculate these from the raw columns if they exist
    
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # NOTE: These are PLACEHOLDER formulas using simple averaging.
    # You should replace with the actual PCA/weighting logic used during training.
    
    # Property Value Index
    property_cols = [c for c in df_new.columns if 'ah4' in c.lower() and any(x in c.lower() for x in ['prop', 'price', 'value'])]
    if property_cols and '[REDACTED_BY_SCRIPT]' not in df_new.columns:
        df_new['[REDACTED_BY_SCRIPT]'] = df_new[property_cols].mean(axis=1) if property_cols else np.nan
        logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Environmental Health Disadvantage Index
    env_health_cols = [c for c in df_new.columns if 'ah4' in c.lower() and any(x in c.lower() for x in ['poll', 'health', 'env'])]
    if env_health_cols and '[REDACTED_BY_SCRIPT]' not in df_new.columns:
        df_new['[REDACTED_BY_SCRIPT]'] = df_new[env_health_cols].mean(axis=1) if env_health_cols else np.nan
        logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Retail Vice Density Index (gambling, alcohol, etc.)
    retail_vice_cols = [c for c in df_new.columns if 'ah4' in c.lower() and any(x in c.lower() for x in ['gamb', 'alc', 'retail'])]
    if retail_vice_cols and '[REDACTED_BY_SCRIPT]' not in df_new.columns:
        df_new['[REDACTED_BY_SCRIPT]'] = df_new[retail_vice_cols].mean(axis=1) if retail_vice_cols else np.nan
        logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Landscape Amenity Index (blue space, green space, etc.)
    landscape_cols = [c for c in df_new.columns if 'ah4' in c.lower() and any(x in c.lower() for x in ['blue', 'green', 'park', 'natural'])]
    if landscape_cols and '[REDACTED_BY_SCRIPT]' not in df_new.columns:
        df_new['[REDACTED_BY_SCRIPT]'] = df_new[landscape_cols].mean(axis=1) if landscape_cols else np.nan
        logging.info(f"[REDACTED_BY_SCRIPT]")

    # ---------------------------------------------------------
    # 6. Strict Schema Enforcement (The Filter)
    # ---------------------------------------------------------
    logging.info("[REDACTED_BY_SCRIPT]")

    # Identify Missing Columns (e.g., 'technology_type', 'is_hot_site')
    missing_cols = [col for col in legacy_schema if col not in df_new.columns]
    if missing_cols:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        # Create a DataFrame of NaNs with the correct index and columns
        nan_df = pd.DataFrame(np.nan, index=df_new.index, columns=missing_cols)
        df_new = pd.concat([df_new, nan_df], axis=1)

    # Reorder to match legacy schema - ONLY legacy columns, no extras
    # This automatically drops any column NOT in legacy_schema (like raw 'ng_...' cols and 'ah4...' cols)  
    legacy_df = df_new[legacy_schema]

    # ---------------------------------------------------------
    # 7. Forensic Schema Verification (The Final Gate)
    # ---------------------------------------------------------
    logging.info("[REDACTED_BY_SCRIPT]")
    
    current_cols = legacy_df.columns.tolist()
    
    # Check 1: Set Integrity (Presence/Absence)
    missing_features = set(legacy_schema) - set(current_cols)
    extra_features = set(current_cols) - set(legacy_schema)
    
    if missing_features:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        logging.error(f"[REDACTED_BY_SCRIPT]")
    
    if extra_features:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        logging.error(f"[REDACTED_BY_SCRIPT]")
        
    if missing_features or extra_features:
        raise ValueError("[REDACTED_BY_SCRIPT]")

    # Check 2: Order Integrity (Crucial for XGBoost matrix inputs)
    if current_cols != legacy_schema:
        logging.error("[REDACTED_BY_SCRIPT]")
        # Diagnostic: Identify the first point of deviation
        for i, (col_actual, col_expected) in enumerate(zip(current_cols, legacy_schema)):
            if col_actual != col_expected:
                logging.error(f"[REDACTED_BY_SCRIPT]'{col_expected}', Found '{col_actual}'")
                break
        raise ValueError("[REDACTED_BY_SCRIPT]")

    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # ---------------------------------------------------------
    # 8. Add hex_id back for pipeline tracking
    # ---------------------------------------------------------
    # The orchestrator needs hex_id to track rows through the pipeline
    # This is NOT part of the model schema, just for pipeline continuity
    if 'hex_id' in df_new.columns and 'hex_id' not in legacy_df.columns:
        legacy_df['hex_id'] = df_new['hex_id']
        logging.info("Added 'hex_id'[REDACTED_BY_SCRIPT]")
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return legacy_df