import geopandas as gpd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
PQ_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
K_NEIGHBORS = 5
NULL_SENTINEL = 0.0 # Harmonic values and ratios default to 0

# --- Module-level State for Performance ---
gdf_pq_profiles, pq_tree = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_pq_profiles, pq_tree
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_pq_profiles = gpd.read_file(PQ_L1_ARTIFACT, engine='pyarrow')
        
        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        assert gdf_pq_profiles.crs.to_string() == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"
        
        # Prepare coordinates for BallTree, which expects radians for haversine
        coords_rad = np.deg2rad(np.array(list(zip(gdf_pq_profiles.geometry.y, gdf_pq_profiles.geometry.x))))
        pq_tree = BallTree(coords_rad, metric='haversine')
        
        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        pq_tree = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if pq_tree is None:
        _initialize_module_state()
    if pq_tree == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    feature_keys = [
        'pq_idw_thd_knn5', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'pq_max_thd_in_knn5', 'pq_std_thd_in_knn5'
    ]

    try:
        geom = state['input_geom']
        app_coords_rad = np.deg2rad([[geom.y, geom.x]])
        
        # Query the tree for k-nearest neighbors
        k = min(K_NEIGHBORS, len(gdf_pq_profiles))
        distances_rad, indices = pq_tree.query(app_coords_rad, k=k)
        
        # Get the cohort of neighbors and their distances in meters
        cohort = gdf_pq_profiles.iloc[indices[0]]
        distances_m = distances_rad[0] * 6371000 # Earth radius in meters
        
        # Guard against division-by-zero for sites co-located with a monitor
        distances_m[distances_m == 0] = 1e-6 
        
        # Calculate Inverse Distance Weights
        weights = 1 / (distances_m ** 2)

        # Synthesize IDW and other spatial features
        state['features']['pq_idw_thd_knn5'] = np.average(cohort['pq_thd_highest'], weights=weights)
        state['features']['[REDACTED_BY_SCRIPT]'] = np.average(cohort['pq_h5_highest'], weights=weights)
        state['features']['[REDACTED_BY_SCRIPT]'] = np.average(cohort['[REDACTED_BY_SCRIPT]'], weights=weights)
        state['features']['[REDACTED_BY_SCRIPT]'] = distances_m[0] / 1000.0
        state['features']['pq_max_thd_in_knn5'] = cohort['pq_thd_highest'].max()
        state['features']['pq_std_thd_in_knn5'] = cohort['pq_thd_highest'].std() if len(cohort) > 1 else 0.0

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state