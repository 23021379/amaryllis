import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os
from scipy.spatial import cKDTree

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
SEC_SUBS_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PRIMARY_SUB_AGG_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
K_NEIGHBORS = 50
REINFORCEMENT_HORIZON_YRS = 5
NULL_SENTINEL = 0.0

# --- Module-level State for Performance ---
gdf_sec_subs, tree_sec_subs = None, None
df_primary_agg = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_sec_subs, tree_sec_subs, df_primary_agg
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        gdf_sec_subs = gpd.read_file(SEC_SUBS_L1_ARTIFACT)
        assert gdf_sec_subs.crs.to_string() == PROJECT_CRS
        coords = np.array(list(zip(gdf_sec_subs.geometry.x, gdf_sec_subs.geometry.y)))
        tree_sec_subs = cKDTree(coords)

        df_primary_agg = pd.read_parquet(PRIMARY_SUB_AGG_L1_ARTIFACT).set_index('[REDACTED_BY_SCRIPT]')

        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        tree_sec_subs = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if tree_sec_subs is None:
        _initialize_module_state()
    if tree_sec_subs == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    knn_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    agg_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    all_feature_keys = knn_keys + agg_keys
    
    try:
        geom = state['input_geom']
        submission_year = state['submission_year']

        # --- Phase 1: Local Environment Features via k-NN ---
        distances, indices = tree_sec_subs.query([geom.x, geom.y], k=K_NEIGHBORS)
        neighbors = gdf_sec_subs.iloc[indices]
        
        state['features']['[REDACTED_BY_SCRIPT]'] = neighbors['is_gmt'].mean()
        state['features']['[REDACTED_BY_SCRIPT]'] = neighbors['customer_count'].mean()
        state['features']['[REDACTED_BY_SCRIPT]'] = neighbors['utilisation_midpoint_pct'].mean()
        state['features']['[REDACTED_BY_SCRIPT]'] = (neighbors['utilisation_midpoint_pct'] > 0.8).sum()
        
        # CRITICAL TEMPORAL GUARD PROTOCOL (Pattern 3)
        imminent_mask = (neighbors['[REDACTED_BY_SCRIPT]'] >= submission_year) & \
                        (neighbors['[REDACTED_BY_SCRIPT]'] < submission_year + REINFORCEMENT_HORIZON_YRS)
        state['features']['[REDACTED_BY_SCRIPT]'] = imminent_mask.sum()

        # --- Phase 2: Aggregation to Primary Substation via Lookup ---
        # The governing primary sub ID is now guaranteed to be in the state from exec_018.
        primary_sub_id = state['features'].get('[REDACTED_BY_SCRIPT]') 
        if primary_sub_id and primary_sub_id in df_primary_agg.index:
            primary_profile = df_primary_agg.loc[primary_sub_id]
            for key in agg_keys:
                state['features'][key] = primary_profile.get(key, NULL_SENTINEL)
        else:
            for key in agg_keys:
                state['features'][key] = NULL_SENTINEL

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in all_feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state