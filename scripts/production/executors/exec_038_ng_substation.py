import os
import logging
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
L1_SUBSTATION_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
TARGET_CRS = "EPSG:27700"
KNN_NEIGHBORS = 5
RADIUS_METERS = 10000
NULL_SENTINEL = -1.0

# --- MODULE-LEVEL STATE ---
gdf_subs, spatial_index = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_subs, spatial_index
    if spatial_index is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_subs = gpd.read_file(L1_SUBSTATION_ARTIFACT)
        assert gdf_subs.crs.to_string() == TARGET_CRS, "[REDACTED_BY_SCRIPT]"
        
        coords = np.array(list(zip(gdf_subs.geometry.x, gdf_subs.geometry.y)))
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

        # --- Direct Nearest Neighbor Features ---
        dist_nn1, idx_nn1 = spatial_index.query(site_coord, k=1)
        nearest_sub = gdf_subs.iloc[idx_nn1]
        
        state['features']['[REDACTED_BY_SCRIPT]'] = dist_nn1 / 1000
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_sub['max_voltage_kv'].values[0]
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_sub['headroom_mw'].values[0]

        # --- K-Nearest Neighbors Aggregation ---
        distances_knn, indices_knn = spatial_index.query(site_coord, k=KNN_NEIGHBORS)
        neighbor_data = gdf_subs.iloc[indices_knn]

        state['features'][f'[REDACTED_BY_SCRIPT]'] = np.mean(distances_knn) / 1000
        state['features'][f'[REDACTED_BY_SCRIPT]'] = np.nanmean(neighbor_data['headroom_mw'])
        state['features'][f'[REDACTED_BY_SCRIPT]'] = np.nanstd(neighbor_data['headroom_mw'])

        # --- Radius-Based Aggregation ---
        indices_radius = spatial_index.query_ball_point(site_coord, r=RADIUS_METERS)
        if indices_radius:
            total_kva_in_radius = gdf_subs.iloc[indices_radius]['total_kva'].sum()
            state['features'][f'[REDACTED_BY_SCRIPT]'] = total_kva_in_radius
        else:
            state['features'][f'[REDACTED_BY_SCRIPT]'] = 0.0

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        # Ensure feature keys exist to maintain schema consistency
        feature_keys = [
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]',
            f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]'
        ]
        for key in feature_keys:
            state['features'].setdefault(key, NULL_SENTINEL)
            
    return state