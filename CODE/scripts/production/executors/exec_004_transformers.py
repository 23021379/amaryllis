import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
TRANSFORMER_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
K_NEIGHBORS = 5
RADIUS_METERS = 10000
NULL_SENTINEL = 0.0 # Capacity sums and averages default to 0

# --- Module-level State for Performance ---
gdf_transformers = None
transformer_tree = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_transformers, transformer_tree
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_transformers = gpd.read_file(TRANSFORMER_L1_ARTIFACT)
        
        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        assert gdf_transformers.crs.to_string() == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"
        
        # Example fix for geometry type error
        gdf_transformers = gdf_transformers[gdf_transformers.geometry.type == 'Point']
        
        coords = np.array(list(zip(gdf_transformers.geometry.x, gdf_transformers.geometry.y)))
        transformer_tree = cKDTree(coords)
        logging.info(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        gdf_transformers = gpd.GeoDataFrame()
        transformer_tree = None

def execute(state: dict) -> dict:
    """
    Calculates KNN and radius-based features for primary transformer capacity.

    Adds the following to state['features']:
    - nearest_sub_total_kva
    - avg_total_kva_5nn
    - total_kva_in_10km_radius
    """
    if transformer_tree is None:
        _initialize_module_state()
        if transformer_tree is None:
            state['errors'].append("[REDACTED_BY_SCRIPT]")
            return state

    try:
        target_coord = [state['input_geom'].x, state['input_geom'].y]
        
        # --- KNN Query (k=5) ---
        distances_knn, indices_knn = transformer_tree.query(target_coord, k=K_NEIGHBORS)
        valid_indices_knn = indices_knn[np.isfinite(distances_knn)]
        
        if len(valid_indices_knn) > 0:
            nearest_subs = gdf_transformers.iloc[valid_indices_knn]
            state['features']['[REDACTED_BY_SCRIPT]'] = nearest_subs.iloc[0]['total_onan_rating_kva']
            state['features']['avg_total_kva_5nn'] = nearest_subs['total_onan_rating_kva'].mean()
        else:
            state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            state['features']['avg_total_kva_5nn'] = NULL_SENTINEL
            
        # --- Radius Query (10km) ---
        indices_radius = transformer_tree.query_ball_point(target_coord, r=RADIUS_METERS)
        
        if len(indices_radius) > 0:
            subs_in_radius = gdf_transformers.iloc[indices_radius]
            state['features']['[REDACTED_BY_SCRIPT]'] = subs_in_radius['total_onan_rating_kva'].sum()
        else:
            state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            
    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        # Ensure schema consistency on failure
        feature_keys = ['[REDACTED_BY_SCRIPT]', 'avg_total_kva_5nn', '[REDACTED_BY_SCRIPT]']
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state