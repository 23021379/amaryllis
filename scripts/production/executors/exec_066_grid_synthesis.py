import pandas as pd
import numpy as np
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Feature Groups for Synthesis Logic
# 1. SELECT: Choose the value from the DNO that is closest (based on dist_to_nearest_substation_km)
# 2. MIN: Take the minimum value (e.g. distance)
# 3. MAX: Take the maximum value (e.g. DER capacity, assuming overlap/subset)
# 4. SUM: Sum the values (e.g. OHL counts, assuming distinct assets)

SYNTHESIS_CONFIG = {
    'SELECT_BASED_ON_NEAREST': {
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        'avg_max_ef_fault_5nn_ka': {'ng': 'ng_avg_max_ef_fault_5nn_ka', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_tx_[REDACTED_BY_SCRIPT]'},
        'avg_total_kva_5nn': {'ng': 'ng_sub_avg_total_kva_5nn', 'ukpn': 'ukpn_tx_avg_total_kva_5nn'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_sub_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_tx_[REDACTED_BY_SCRIPT]'},
        
        # Primary Substation Details
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_psub_total_kva'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_psub_tx_count'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_psub_avg_tx_kva'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_psub_max_tx_kva'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'}, # FIXED: primary_sub -> psub
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'}, # FIXED: primary_sub -> psub
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        'sub_demandrag_green': {'ng': 'ng_sub_demandrag_green', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        'sub_demandrag_red': {'ng': 'ng_sub_demandrag_red', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        'dist_to_sec_sub_m': {'ng': 'ng_dist_to_sec_sub_m', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        
        # DNOA
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix
        'dnoa_avg_dist_knn_m': {'ng': 'ng_dnoa_avg_dist_knn_m', 'ukpn': '[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix
        'dnoa_avg_deferred_kva_knn': {'ng': 'ng_dnoa_avg_deferred_kva_knn', 'ukpn': 'ukpn_dnoa_avg_deferred_kva_knn'}, # FIXED: Added ng_ prefix
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'}, # FIXED: Added ng_ prefix

        # PQ
        'pq_idw_thd_knn5': {'ng': 'pq_idw_thd_knn5', 'ukpn': 'ukpn_pq_idw_thd_knn5'},
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        'pq_max_thd_in_knn5': {'ng': 'pq_max_thd_in_knn5', 'ukpn': 'ukpn_pq_max_thd_in_knn5'},
        'pq_std_thd_in_knn5': {'ng': 'pq_std_thd_in_knn5', 'ukpn': 'ukpn_pq_std_thd_in_knn5'},
        
        # LTDS
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        'ltds_count_in_10km': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        
        # LCT
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        'lct_total_connections_5km': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        'lct_total_import_kw_5km': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_lct_nearest_total_connections', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': '[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        'lct_avg_5nn_total_connections': {'ng': 'ng_lct_avg_5nn_total_connections', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        
        # IDNO
        'idno_is_within': {'ng': 'ng_idno_is_within', 'ukpn': 'ukpn_idno_is_within'},
        'idno_dist_to_nearest_m': {'ng': 'ng_idno_dist_to_nearest_m', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
        'lpa_idno_area_pct': {'ng': 'ng_lpa_idno_area_pct', 'ukpn': '[REDACTED_BY_SCRIPT]'},
        '[REDACTED_BY_SCRIPT]': {'ng': 'ng_[REDACTED_BY_SCRIPT]', 'ukpn': 'ukpn_[REDACTED_BY_SCRIPT]'},
    },
    
    'MIN': {
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
    },
    
    'MAX': {
        # DER Capacity & Counts (Assume National ECR vs UKPN ECR might overlap, so MAX is safest)
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
    },
    
    'SUM': {
        # OHL Counts (Distinct assets)
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        'total_connection_path_intersections': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        'pole_count_2km': ['count_ng_pole_2km', 'count_ukpn_pole_2km'],
        'pole_count_5km': ['count_ng_pole_5km', 'count_ukpn_pole_5km'],
        'pole_count_10km': ['count_ng_pole_10km', '[REDACTED_BY_SCRIPT]'],
        'tower_count_2km': ['count_ng_tower_2km', '[REDACTED_BY_SCRIPT]'],
        'tower_count_5km': ['count_ng_tower_5km', '[REDACTED_BY_SCRIPT]'],
        'tower_count_10km': ['count_ng_tower_10km', '[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        
        # Intersects Site (Boolean-ish, but sum works as OR if 0/1)
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
    },
    
    'MIN_DISTANCE_CORRIDOR': {
        # OHL Distances (Nearest corridor)
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        'pole_dist_to_nearest_m': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
        'tower_dist_to_nearest_m': ['dist_to_nearest_ng_tower', '[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
        '[REDACTED_BY_SCRIPT]': ['ng_[REDACTED_BY_SCRIPT]', 'ukpn_[REDACTED_BY_SCRIPT]'],
    }
}


def execute(master_gdf):
    """
    Synthesizes DNO-specific grid features into a single, canonical set of features.
    Uses robust logic to select or aggregate features based on the nearest DNO.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    id_col = 'hex_id'
    if master_gdf.index.name == id_col:
        gdf = master_gdf.reset_index()
    elif id_col in master_gdf.columns:
        gdf = master_gdf.copy()
    else:
        logging.error(f"FATAL: '{id_col}'[REDACTED_BY_SCRIPT]")
        return master_gdf

    # --- 1. Pre-processing: Clean Sentinels ---
    # Replace -1, 0 (where inappropriate), etc. with NaN to ensure correct min/max/mean
    # Note: For counts, 0 is valid. For distances, -1 or 0 might be sentinels.
    # We'll handle specific sentinels if needed, but generally coercing to numeric helps.
    
    # --- 1.5. Unit Standardization (Fixing Mismatch Pattern) ---
    # Fix PQ Distance Unit Mismatch (NG is KM, UKPN is M, Target is M)
    if '[REDACTED_BY_SCRIPT]' in gdf.columns:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf['[REDACTED_BY_SCRIPT]'] = gdf['[REDACTED_BY_SCRIPT]'] * 1000
        # Update map to use the new standardized column
        SYNTHESIS_CONFIG['SELECT_BASED_ON_NEAREST']['[REDACTED_BY_SCRIPT]']['ng'] = 'ng_[REDACTED_BY_SCRIPT]'

    # --- 2. Determine Nearest DNO (With Sentinel Scrubbing) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    ng_dist_col = '[REDACTED_BY_SCRIPT]'
    ukpn_dist_col = '[REDACTED_BY_SCRIPT]'
    
    # Ensure distance columns exist
    if ng_dist_col not in gdf.columns: gdf[ng_dist_col] = np.nan
    if ukpn_dist_col not in gdf.columns: gdf[ukpn_dist_col] = np.nan
    
    # Coerce to numeric
    gdf[ng_dist_col] = pd.to_numeric(gdf[ng_dist_col], errors='coerce')
    gdf[ukpn_dist_col] = pd.to_numeric(gdf[ukpn_dist_col], errors='coerce')

    # SENTINEL SCRUBBING: Treat values > 5000km or < 0 as Invalid/NaN
    # This prevents 'infinity' sentinels (e.g. 1.6M km) from skewing the nearest DNO logic
    sentinel_threshold = 5000 
    
    mask_ng_invalid = (gdf[ng_dist_col] > sentinel_threshold) | (gdf[ng_dist_col] < 0)
    gdf.loc[mask_ng_invalid, ng_dist_col] = np.nan
    
    mask_ukpn_invalid = (gdf[ukpn_dist_col] > sentinel_threshold) | (gdf[ukpn_dist_col] < 0)
    gdf.loc[mask_ukpn_invalid, ukpn_dist_col] = np.nan

    # Logic:
    # If both NaN -> NaN
    # If one NaN -> use the other
    # If both valid -> compare
    
    # Create a 'nearest_dno' column: 'ng', 'ukpn', or None
    def get_nearest_dno(row):
        ng_d = row[ng_dist_col]
        ukpn_d = row[ukpn_dist_col]
        
        if pd.isna(ng_d) and pd.isna(ukpn_d):
            return None
        if pd.isna(ng_d):
            return 'ukpn'
        if pd.isna(ukpn_d):
            return 'ng'
        
        # Both valid, compare
        return 'ng' if ng_d < ukpn_d else 'ukpn'

    gdf['nearest_dno'] = gdf.apply(get_nearest_dno, axis=1)
    
    # --- 3. Execute Synthesis Logic ---
    
    # Validation Helper
    def _validate_source_columns(df, needed_cols, feature_name):
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            logging.warning(f"[REDACTED_BY_SCRIPT]")

    # A. SELECT_BASED_ON_NEAREST
    for target_col, source_map in SYNTHESIS_CONFIG['SELECT_BASED_ON_NEAREST'].items():
        ng_col = source_map.get('ng')
        ukpn_col = source_map.get('ukpn')
        
        _validate_source_columns(gdf, [c for c in [ng_col, ukpn_col] if c], target_col)
        
        # Ensure columns exist locally to avoid KeyError
        if ng_col and ng_col not in gdf.columns: gdf[ng_col] = np.nan
        if ukpn_col and ukpn_col not in gdf.columns: gdf[ukpn_col] = np.nan
        
        def select_val(row):
            dno = row['nearest_dno']
            if dno == 'ng':
                return row[ng_col]
            elif dno == 'ukpn':
                return row[ukpn_col]
            else:
                return np.nan
        
        gdf[target_col] = gdf.apply(select_val, axis=1)
        # FORCE NUMERIC: Corrects variance issues caused by object-dtype contamination
        gdf[target_col] = pd.to_numeric(gdf[target_col], errors='coerce')

    # B. MIN (Distances)
    for target_col, sources in SYNTHESIS_CONFIG['MIN'].items():
        valid_sources = [c for c in sources if c in gdf.columns]
        if not valid_sources:
            gdf[target_col] = np.nan
            continue
        
        # Clean sentinels (e.g. -1) before min
        for col in valid_sources:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
            gdf[col] = gdf[col].replace(-1, np.nan)
            
        gdf[target_col] = gdf[valid_sources].min(axis=1)

    # C. MAX (DER / Capacity)
    for target_col, sources in SYNTHESIS_CONFIG['MAX'].items():
        valid_sources = [c for c in sources if c in gdf.columns]
        if not valid_sources:
            gdf[target_col] = 0 # Default for density is 0
            continue
            
        for col in valid_sources:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
            
        gdf[target_col] = gdf[valid_sources].max(axis=1)

    # D. SUM (OHL Counts)
    for target_col, sources in SYNTHESIS_CONFIG['SUM'].items():
        valid_sources = [c for c in sources if c in gdf.columns]
        if not valid_sources:
            gdf[target_col] = 0
            continue
            
        for col in valid_sources:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
            
        gdf[target_col] = gdf[valid_sources].sum(axis=1)

    # E. MIN_DISTANCE_CORRIDOR (OHL Distances)
    for target_col, sources in SYNTHESIS_CONFIG['MIN_DISTANCE_CORRIDOR'].items():
        valid_sources = [c for c in sources if c in gdf.columns]
        if not valid_sources:
            gdf[target_col] = np.nan
            continue
        
        for col in valid_sources:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
            # Replace 0 with NaN if it implies "no data" (though 0 dist is valid for OHL)
            # Assuming -1 is sentinel for OHL if used
            gdf[col] = gdf[col].replace(-1, np.nan)
            
        gdf[target_col] = gdf[valid_sources].min(axis=1)

    # --- Legacy Feature Renaming ---
    legacy_rename_map = {
        '[REDACTED_BY_SCRIPT]': 'powertransformercount',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]': 'transratingwinter',
        '[REDACTED_BY_SCRIPT]': 'transratingsummer'
    }
    gdf.rename(columns=legacy_rename_map, inplace=True)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- Finalization ---
    gdf.set_index(id_col, inplace=True)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    return gdf
