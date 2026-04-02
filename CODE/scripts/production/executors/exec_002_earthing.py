import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
FAULT_LEVEL_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]") # New authoritative artifact
PROJECT_CRS = "EPSG:27700"
K_NEIGHBORS = 5
NULL_SENTINEL = -1.0

# --- Module-level State for Performance ---
gdf_fault_profiles = None
fault_tree = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_fault_profiles, fault_tree
    try:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf_fault_profiles = gpd.read_file(FAULT_LEVEL_L1_ARTIFACT)
        
        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        assert gdf_fault_profiles.crs.to_string() == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"
        
        # PRE-OPERATIVE CHECK: GEOMETRY VALIDATION
        assert gdf_fault_profiles.geometry.notna().all(), "[REDACTED_BY_SCRIPT]"

        coords = np.array(list(zip(gdf_fault_profiles.geometry.x, gdf_fault_profiles.geometry.y)))
        fault_tree = cKDTree(coords)
        
        logging.info(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        gdf_fault_profiles = gpd.GeoDataFrame() # Prevent future calls from succeeding
        fault_tree = None

def execute(state: dict) -> dict:
    """
    Calculates grid stability features based on substation fault level data.
    
    Adds the following to state['features']:
    - nearest_sub_max_3ph_fault_ka
    - nearest_sub_max_ef_fault_ka
    - avg_max_3ph_fault_5nn_ka
    - avg_max_ef_fault_5nn_ka
    """
    if fault_tree is None:
        _initialize_module_state()
        if fault_tree is None:
            state['errors'].append("[REDACTED_BY_SCRIPT]")
            return state

    try:
        input_geom = state['input_geom']
        target_coord = [input_geom.x, input_geom.y]
        
        distances_m, indices = fault_tree.query(target_coord, k=K_NEIGHBORS)
        
        if np.isinf(distances_m).any():
            state['errors'].append("[REDACTED_BY_SCRIPT]")
            return state
            
        neighbors = gdf_fault_profiles.iloc[indices]
        
        # Nearest Substation (n=1) features
        nearest_neighbor = neighbors.iloc[0]
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get('threephasermsbreak', NULL_SENTINEL)
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get('earthfaultrmsbreak', NULL_SENTINEL)
        
        # Substation Cluster (n=k) features
        state['features']['[REDACTED_BY_SCRIPT]'] = neighbors['threephasermsbreak'].mean()
        state['features']['avg_max_ef_fault_5nn_ka'] = neighbors['earthfaultrmsbreak'].mean()

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        # Ensure schema consistency on failure
        feature_keys = [
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]', 'avg_max_ef_fault_5nn_ka'
        ]
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL
                
    return state