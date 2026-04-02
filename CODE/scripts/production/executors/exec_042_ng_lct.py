import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
L2_LCT_SPATIOTEMPORAL_INPUT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
L1_PRIMARY_SUBS_INPUT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
TARGET_CRS = "EPSG:27700"
KNN_NEIGHBORS = 5
NULL_SENTINEL = -1.0

# --- MODULE-LEVEL STATE ---
gdf_lct_events, gdf_primary_subs, spatial_index = None, None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_lct_events, gdf_primary_subs, spatial_index
    if spatial_index is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_lct_events = gpd.read_file(L2_LCT_SPATIOTEMPORAL_INPUT)
        gdf_lct_events['connected_month'] = pd.to_datetime(gdf_lct_events['connected_month'])
        
        gdf_primary_subs = gpd.read_file(L1_PRIMARY_SUBS_INPUT)
        assert gdf_primary_subs.crs.to_string() == TARGET_CRS, "[REDACTED_BY_SCRIPT]"
        
        coords = np.array(list(zip(gdf_primary_subs.geometry.x, gdf_primary_subs.geometry.y)))
        spatial_index = cKDTree(coords)
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        spatial_index = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if spatial_index is None: _initialize_module_state()
    if spatial_index == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        # --- Temporal Guard Protocol ---
        submission_date = state['date_validated'] # Assumes orchestrator provides this key
        lct_snapshot = gdf_lct_events[gdf_lct_events['connected_month'] <= submission_date]

        if lct_snapshot.empty: # No historical LCT data, null-fill features
            for key in _get_feature_keys(): state['features'][key] = NULL_SENTINEL
            return state

        # --- Dynamic Point-in-Time Aggregation ---
        agg_df = lct_snapshot.groupby('primary_sub_id')[['ev', 'es', 'hp', 'pv']].sum()
        agg_df['total_connections'] = agg_df.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            agg_df['gen_dem_ratio'] = (agg_df['pv'] + agg_df['es']) / (agg_df['ev'] + agg_df['hp'] + 1)
        
        enriched_subs = gdf_primary_subs.merge(agg_df, on='primary_sub_id', how='left').fillna(0)
        
        # --- k-NN Feature Synthesis ---
        site_coord = (state['input_geom'].x, state['input_geom'].y)
        _, indices_knn = spatial_index.query(site_coord, k=KNN_NEIGHBORS)
        
        nearest_subs_data = enriched_subs.iloc[indices_knn]
        nearest_sub = nearest_subs_data.iloc[0]

        # Nearest Neighbor Features (n=1)
        state['features']['lct_nearest_total_connections'] = nearest_sub['total_connections']
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_sub['gen_dem_ratio']
        
        # Local Cluster Features (k-NN)
        state['features'][f'[REDACTED_BY_SCRIPT]'] = nearest_subs_data['total_connections'].mean()
        state['features'][f'[REDACTED_BY_SCRIPT]'] = nearest_subs_data['total_connections'].std()
        state['features'][f'[REDACTED_BY_SCRIPT]'] = nearest_subs_data['gen_dem_ratio'].mean()

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in _get_feature_keys(): state['features'].setdefault(key, NULL_SENTINEL)
            
    return state

def _get_feature_keys():
    """[REDACTED_BY_SCRIPT]"""
    return [
        'lct_nearest_total_connections', '[REDACTED_BY_SCRIPT]',
        f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]',
        f'[REDACTED_BY_SCRIPT]'
    ]