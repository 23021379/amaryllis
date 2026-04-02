import os
import logging
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
L1_DIST_SUBS_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
TARGET_CRS = "EPSG:27700"
PRIMARY_SUB_PROXY_NEIGHBORS = 10 # Number of secondary subs to aggregate for primary features
NULL_SENTINEL = -1.0

# --- MODULE-LEVEL STATE ---
gdf_dist_subs, spatial_index = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_dist_subs, spatial_index
    if spatial_index is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_dist_subs = gpd.read_file(L1_DIST_SUBS_ARTIFACT)
        assert gdf_dist_subs.crs.to_string() == TARGET_CRS, "[REDACTED_BY_SCRIPT]"
        
        coords = np.array(list(zip(gdf_dist_subs.geometry.x, gdf_dist_subs.geometry.y)))
        spatial_index = cKDTree(coords)
        logging.info(f"[REDACTED_BY_SCRIPT]")
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
        site_geom = state['input_geom']
        site_coord = (site_geom.x, site_geom.y)

        # --- Direct Features (Nearest Secondary Substation) ---
        _, idx_nn1 = spatial_index.query(site_coord, k=1)
        nearest_sec_sub = gdf_dist_subs.iloc[idx_nn1]
        
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_sec_sub['rating_kva'].values[0]
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_sec_sub['utilisation_pct'].values[0]

        # --- Aggregate Features (Primary Substation Proxy via k-NN) ---
        _, indices_knn = spatial_index.query(site_coord, k=PRIMARY_SUB_PROXY_NEIGHBORS)
        neighbor_data = gdf_dist_subs.iloc[indices_knn]
        
        state['features']['[REDACTED_BY_SCRIPT]'] = np.nansum(neighbor_data['rating_kva'])
        state['features']['[REDACTED_BY_SCRIPT]'] = PRIMARY_SUB_PROXY_NEIGHBORS
        state['features']['[REDACTED_BY_SCRIPT]'] = np.nanmean(neighbor_data['rating_kva'])
        state['features']['[REDACTED_BY_SCRIPT]'] = np.nanmax(neighbor_data['rating_kva'])
        state['features']['[REDACTED_BY_SCRIPT]'] = np.nanmin(neighbor_data['rating_kva'])
        state['features']['[REDACTED_BY_SCRIPT]'] = np.nanvar(neighbor_data['rating_kva'])
        state['features']['[REDACTED_BY_SCRIPT]'] = np.nanmean(neighbor_data['utilisation_pct'])
        
        # --- Second-Order Synthesis ---
        # This calculation is now robust as it only depends on features generated within this executor.
        with np.errstate(divide='ignore', invalid='ignore'):
            kva_per_tx = state['features']['[REDACTED_BY_SCRIPT]'] / state['features']['[REDACTED_BY_SCRIPT]']
            state['features']['kva_per_transformer'] = np.nan_to_num(kva_per_tx, nan=NULL_SENTINEL)

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        # Null-fill all features this executor is responsible for to maintain schema
        feature_keys = [
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]', 'kva_per_transformer'
        ]
        for key in feature_keys: state['features'].setdefault(key, NULL_SENTINEL)
            
    return state