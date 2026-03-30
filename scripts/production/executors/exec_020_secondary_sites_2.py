import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree
from shapely.geometry import LineString

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
CANONICAL_SUBS_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = 0.0

# --- Module-level State for Performance ---
gdf_sec_sub_areas, sindex_sec_sub_areas = None, None
gdf_primary_subs, tree_primary_subs = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_sec_sub_areas, sindex_sec_sub_areas, gdf_primary_subs, tree_primary_subs
    try:
        logging.info("[REDACTED_BY_SCRIPT]")

        # Read the single authoritative artifact once
        gdf_master = gpd.read_file(CANONICAL_SUBS_L1_ARTIFACT)
        assert gdf_master.crs.to_string() == PROJECT_CRS

        # Populate all necessary module-level variables from the single master copy
        gdf_sec_sub_areas = gdf_master
        sindex_sec_sub_areas = gdf_sec_sub_areas.sindex

        gdf_primary_subs = gdf_master
        coords_subs_rad = np.deg2rad(np.array(list(zip(gdf_primary_subs.geometry.y, gdf_primary_subs.geometry.x))))
        tree_primary_subs = BallTree(coords_subs_rad, metric='haversine')
        
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_sec_sub_areas = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_sec_sub_areas is None:
        _initialize_module_state()
    if sindex_sec_sub_areas == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'sec_sub_gov_area_sqm', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]' # Upgraded version
    ]
    
    try:
        geom = state['input_geom']
        governing_area, join_method = None, None
        
        # --- Two-Stage, Fault-Tolerant Join (Amaryllis Alpha) ---
        possible_matches_idx = list(sindex_sec_sub_areas.intersection(geom.bounds))
        if possible_matches_idx:
            containing_areas = gdf_sec_sub_areas.iloc[possible_matches_idx]
            containing_areas = containing_areas[containing_areas.intersects(geom)]
            if not containing_areas.empty:
                governing_area = containing_areas.iloc[0]
                join_method = 'within'
        
        if governing_area is None:
            _, tree_indices = sindex_sec_sub_areas.nearest(geom, return_all=False)
            governing_area = gdf_sec_sub_areas.iloc[tree_indices[0]]
            join_method = 'nearest'
            
        state['features']['[REDACTED_BY_SCRIPT]'] = 1 if join_method == 'within' else 0
        
        # --- Direct Inheritance & Land Character Features ---
        state['features']['[REDACTED_BY_SCRIPT]'] = governing_area.get('utilisation_midpoint_pct', NULL_SENTINEL)
        state['features']['[REDACTED_BY_SCRIPT]'] = governing_area.get('customer_count', NULL_SENTINEL)
        state['features']['[REDACTED_BY_SCRIPT]'] = governing_area.get('onan_rating_kva', NULL_SENTINEL)
        area_sqm = governing_area.get('sec_sub_area_sqm', 0)
        state['features']['sec_sub_gov_area_sqm'] = area_sqm
        state['features']['[REDACTED_BY_SCRIPT]'] = (state['features']['[REDACTED_BY_SCRIPT]'] / (area_sqm / 1e6 + 1e-9))

        # --- Friction Feature Upgrade ---
        app_coords_rad = np.deg2rad([[geom.y, geom.x]])
        _, sub_idx = tree_primary_subs.query(app_coords_rad, k=1)
        nearest_primary_geom = gdf_primary_subs.geometry.iloc[sub_idx[0][0]]
        connection_path = LineString([geom, nearest_primary_geom])
        
        path_candidates_idx = list(sindex_sec_sub_areas.intersection(connection_path.bounds))
        path_candidates = gdf_sec_sub_areas.iloc[path_candidates_idx]
        intersection_count = path_candidates.intersects(connection_path).sum()
        state['features']['[REDACTED_BY_SCRIPT]'] = intersection_count

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL
                
    return state