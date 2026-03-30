import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]" # As defined in monolith
SUBSTATION_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]") # Authoritative L1 artifact
PROJECT_CRS = "EPSG:27700"
K_NEIGHBORS = 5
NULL_SENTINEL = -1.0 # Use float for consistency

# --- Module-level State for Performance ---
# These variables are initialized once per worker process, not per function call.
# This prevents catastrophic I/O and indexing overhead.
substation_gdf = None
substation_tree = None
forecast_years = None

def _initialize_module_state():
    """
    Loads the L1 artifact and builds the spatial index.
    This is called only once when the first point is processed by a worker.
    """
    global substation_gdf, substation_tree, forecast_years
    try:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        substation_gdf = gpd.read_file(SUBSTATION_L1_ARTIFACT)
        
        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        assert substation_gdf.crs.to_string() == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"
        
        # Build spatial index
        substation_coords = np.array(list(zip(substation_gdf.geometry.x, substation_gdf.geometry.y)))
        substation_tree = cKDTree(substation_coords)
        
        # Pre-calculate available forecast years from artifact columns
        forecast_years = sorted([int(col.split('_')[-1]) for col in substation_gdf.columns if 'headroom_mw_' in col])
        
        logging.info(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        # Setting state to non-None but invalid values ensures subsequent calls fail cleanly.
        substation_gdf = gpd.GeoDataFrame()
        substation_tree = None
        forecast_years = []

def _get_relevant_headroom_col(submission_year: int) -> str:
    """[REDACTED_BY_SCRIPT]"""
    for forecast_year in forecast_years:
        if submission_year < forecast_year:
            return f'[REDACTED_BY_SCRIPT]'
    # Fallback to the latest available forecast if submission year is beyond all forecasts
    return f'[REDACTED_BY_SCRIPT]'

def execute(state: dict) -> dict:
    """
    Calculates grid capacity and proximity features based on DFES substation data.
    
    Adds the following to state['features']:
    - dist_to_nearest_sub_km
    - nearest_sub_voltage_kv
    - nearest_sub_headroom_mw
    - avg_dist_to_5nn_sub_km
    - avg_headroom_5nn_sub_mw
    - std_headroom_5nn_sub_mw
    """
    # Lazy initialization: Load data only when the first point is processed.
    if substation_tree is None:
        _initialize_module_state()
        if substation_tree is None: # Check if initialization failed
            state['errors'].append("[REDACTED_BY_SCRIPT]")
            return state

    try:
        input_geom = state['input_geom']
        submission_year = state['submission_year']
        
        target_coord = [input_geom.x, input_geom.y]
        distances_m, indices = substation_tree.query(target_coord, k=K_NEIGHBORS)
        
        if np.isinf(distances_m).any():
             state['errors'].append("[REDACTED_BY_SCRIPT]")
             return state
             
        neighbors = substation_gdf.iloc[indices]
        relevant_headroom_col = _get_relevant_headroom_col(submission_year)
        
        # --- Feature Synthesis ---
        # Nearest Substation (n=1) features
        nearest_neighbor = neighbors.iloc[0]
        state['features']['[REDACTED_BY_SCRIPT]'] = distances_m[0] / 1000.0
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get('voltage', NULL_SENTINEL)
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get(relevant_headroom_col, NULL_SENTINEL)
        
        # Substation Cluster (n=k) features
        state['features']['[REDACTED_BY_SCRIPT]'] = np.mean(distances_m) / 1000.0
        
        if relevant_headroom_col in neighbors.columns:
            headroom_values = neighbors[relevant_headroom_col].dropna()
            if not headroom_values.empty:
                state['features']['[REDACTED_BY_SCRIPT]'] = headroom_values.mean()
                state['features']['[REDACTED_BY_SCRIPT]'] = headroom_values.std() if len(headroom_values) > 1 else 0.0
            else:
                state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
                state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        else:
             state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
             state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
             
    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg) # Log full trace for debugging if needed
        state['errors'].append(error_msg)
        # Ensure feature keys exist even on failure to maintain consistent schema
        feature_keys = [
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
        ]
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state