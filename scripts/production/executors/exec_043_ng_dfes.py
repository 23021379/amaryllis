import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

# --- CONFIGURATION ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
L1_LPA_RECONCILED_INPUT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
L1_DFES_LOOKUP_INPUT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
TARGET_CRS = "EPSG:27700"
HORIZON_YEARS = 5
NULL_SENTINEL = -1.0
EPSILON = 1e-6

# --- MODULE-LEVEL STATE ---
gdf_lpa, dfes_lookup, sindex_lpa = None, None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_lpa, dfes_lookup, sindex_lpa
    if sindex_lpa is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_lpa = gpd.read_file(L1_LPA_RECONCILED_INPUT)
        sindex_lpa = gdf_lpa.sindex

        gdf_lpa = gpd.read_file(L1_LPA_RECONCILED_INPUT)
        sindex_lpa = gdf_lpa.sindex

        # The DFES lookup is now correctly read as a GeoDataFrame.
        # The original logic intended a non-spatial lookup, so we convert it to a DataFrame.
        dfes_lookup_raw = gpd.read_file(L1_DFES_LOOKUP_INPUT)
        dfes_lookup = pd.DataFrame(dfes_lookup_raw.drop(columns='geometry'))

        # The rest of the original logic can now proceed safely.
        dfes_lookup.set_index(['reconciled_key', 'year'], inplace=True)
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_lpa = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_lpa is None: _initialize_module_state()
    if sindex_lpa == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        # --- Step 1: Geospatial Lookup ---
        geom = state['input_geom']
        possible_matches_idx = list(sindex_lpa.intersection(geom.bounds))
        possible_matches = gdf_lpa.iloc[possible_matches_idx]
        precise_match = possible_matches[possible_matches.intersects(geom)]
        
        if precise_match.empty:
            raise ValueError("[REDACTED_BY_SCRIPT]")
        reconciled_key = precise_match.iloc[0]['reconciled_key']
        
        # --- Step 2: Temporal Lookup (The Guard Protocol) ---
        submission_year = state['date_validated'].year
        target_future_year = submission_year + HORIZON_YEARS

        # High-performance, fault-tolerant lookup using multi-index
        present_row = dfes_lookup.loc[pd.IndexSlice[reconciled_key, :submission_year]].tail(1)
        future_row = dfes_lookup.loc[pd.IndexSlice[reconciled_key, target_future_year:]].head(1)
        
        if present_row.empty or future_row.empty:
            raise ValueError(f"[REDACTED_BY_SCRIPT]'{reconciled_key}'")

        # --- Step 3: Feature Synthesis ---
        gen_growth = future_row['generation_twh'].values[0] - present_row['generation_twh'].values[0]
        dem_growth = future_row['demand_twh'].values[0] - present_row['demand_twh'].values[0]

        state['features']['[REDACTED_BY_SCRIPT]'] = (gen_growth / (present_row['generation_twh'].values[0] + EPSILON)) * 100
        state['features']['[REDACTED_BY_SCRIPT]'] = gen_growth / (dem_growth + EPSILON)
        state['features']['[REDACTED_BY_SCRIPT]'] = future_row['[REDACTED_BY_SCRIPT]'].values[0]

    except Exception as e:
        error_msg = f"EXEC_007_FAIL: {e}"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        feature_keys = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
        for key in feature_keys: state['features'].setdefault(key, NULL_SENTINEL)
            
    return state