import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
# This executor now depends on a pre-indexed L1 artifact, not a raw source file.
SUBSTATION_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
K_NEIGHBORS = 5
NULL_SENTINEL = -1

# --- Module-level Cache for Performance ---
# This dictionary will be populated once per worker process, avoiding repeated file I/O.
_G_DATA = {}

def _initialize_cache():
    """[REDACTED_BY_SCRIPT]"""
    if "substation_gdf" not in _G_DATA:
        substation_gdf = gpd.read_file(SUBSTATION_L1_ARTIFACT)
        _G_DATA["substation_gdf"] = substation_gdf
        
        coords = np.array(list(zip(substation_gdf.geometry.x, substation_gdf.geometry.y)))
        _G_DATA["substation_tree"] = cKDTree(coords)
        
        forecast_years = sorted([int(c.split('_')[-1]) for c in substation_gdf.columns if 'headroom_mw_' in c])
        _G_DATA["forecast_years"] = forecast_years

def _get_relevant_headroom_col(year: int) -> str:
    """[REDACTED_BY_SCRIPT]"""
    forecast_years = _G_DATA["forecast_years"]
    for fy in forecast_years:
        if year < fy:
            return f'headroom_mw_{fy}'
    # If the submission year is past the last forecast, use the last available forecast
    return f'[REDACTED_BY_SCRIPT]'

def execute(state: dict) -> dict:
    """
    Calculates features related to grid headroom from the DFES dataset.
    - state['input_geom']: The shapely Point geometry of the hex centroid.
    - state['submission_year']: The year of the application.
    - state['features']: The dictionary of features calculated so far.
    """
    _initialize_cache()
    
    substation_gdf = _G_DATA["substation_gdf"]
    substation_tree = _G_DATA["substation_tree"]

    input_geom = state['input_geom']
    # A prediction year must be provided in the state for time-sensitive features.
    submission_year = state.get('submission_year')
    
    if not submission_year or submission_year == NULL_SENTINEL:
        state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        state['features']['[REDACTED_BY_SCRIPT]'] = "MISSING_YEAR"
        state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        return state

    dist_m, indices = substation_tree.query([input_geom.x, input_geom.y], k=K_NEIGHBORS)
    
    # Get the single nearest neighbor (k=0)
    nearest_neighbor = substation_gdf.iloc[indices[0]]
    headroom_col = _get_relevant_headroom_col(submission_year)
    
    # Update the state dictionary with new features
    state['features']['[REDACTED_BY_SCRIPT]'] = dist_m[0] / 1000.0
    state['features']['[REDACTED_BY_SCRIPT]'] = nearest_neighbor['sitefunctionallocation']
    state['features']['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get(headroom_col, NULL_SENTINEL)
    
    return state