import geopandas as gpd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree
from shapely.geometry import LineString

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
TOPOLOGICAL_ASSETS_CANONICAL_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PRIMARY_SUBSTATIONS_CANONICAL_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
LOCAL_NETWORK_BUFFER_METERS = 2000
NULL_SENTINEL = 0.0

# --- Module-level State for Performance ---
gdf_dlv_lines, gdf_dlv_poles, gdf_dlv_corridors, gdf_subs = None, None, None, None
sindex_dlv_lines, sindex_dlv_corridors = None, None
tree_dlv_poles, tree_subs = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_dlv_lines, gdf_dlv_poles, gdf_dlv_corridors, gdf_subs
    global sindex_dlv_lines, sindex_dlv_corridors, tree_dlv_poles, tree_subs
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load and triage topological assets for DLV
        gdf_points_master = gpd.read_file(TOPOLOGICAL_ASSETS_CANONICAL_ARTIFACT, layer='points')
        gdf_lines_master = gpd.read_file(TOPOLOGICAL_ASSETS_CANONICAL_ARTIFACT, layer='lines')
        
        gdf_dlv_lines = gdf_lines_master[gdf_lines_master['voltage'] == 'lv'].copy()
        sindex_dlv_lines = gdf_dlv_lines.sindex

        gdf_dlv_poles = gdf_points_master[gdf_points_master['voltage'] == 'lv'].copy()
        coords_poles = np.deg2rad(np.array(list(zip(gdf_dlv_poles.geometry.y, gdf_dlv_poles.geometry.x))))
        tree_dlv_poles = BallTree(coords_poles, metric='haversine')

        gdf_dlv_corridors = gdf_dlv_lines.copy()
        gdf_dlv_corridors['geometry'] = gdf_dlv_corridors.apply(lambda row: row.geometry.buffer(row.buffer_m), axis=1)
        sindex_dlv_corridors = gdf_dlv_corridors.sindex

        # Load authoritative substation data
        gdf_subs = gpd.read_parquet(PRIMARY_SUBSTATIONS_CANONICAL_ARTIFACT)
        assert gdf_subs.crs.to_string() == PROJECT_CRS
        coords_subs = np.deg2rad(np.array(list(zip(gdf_subs.geometry.y, gdf_subs.geometry.x))))
        tree_subs = BallTree(coords_subs, metric='haversine')

        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        tree_dlv_poles = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if tree_dlv_poles is None:
        _initialize_module_state()
    if tree_dlv_poles == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    
    try:
        geom = state['input_geom']
        app_coords_rad = np.deg2rad([[geom.y, geom.x]])
        buffer = geom.buffer(LOCAL_NETWORK_BUFFER_METERS)
        buffer_area_km2 = np.pi * ((LOCAL_NETWORK_BUFFER_METERS / 1000) ** 2)

        # --- D-LV Density Features ---
        line_candidates = gdf_dlv_lines.iloc[list(sindex_dlv_lines.intersection(buffer.bounds))]
        total_line_length_km = gpd.clip(line_candidates, buffer).geometry.length.sum() / 1000
        state['features']['[REDACTED_BY_SCRIPT]'] = total_line_length_km / buffer_area_km2
        
        radius_rad = LOCAL_NETWORK_BUFFER_METERS / 6371000
        pole_count = tree_dlv_poles.query_radius(app_coords_rad, r=radius_rad, count_only=True)[0]
        state['features']['[REDACTED_BY_SCRIPT]'] = pole_count / buffer_area_km2

        # --- D-LV Frictional Features ---
        _, sub_idx = tree_subs.query(app_coords_rad, k=1)
        nearest_sub_geom = gdf_subs.geometry.iloc[sub_idx[0][0]]
        connection_path = LineString([geom, nearest_sub_geom])
        
        path_candidates = gdf_dlv_corridors.iloc[list(sindex_dlv_corridors.intersection(connection_path.bounds))]
        dlv_intersection_count = path_candidates.intersects(connection_path).sum()
        state['features']['[REDACTED_BY_SCRIPT]'] = dlv_intersection_count

        # --- Capstone Synthesis (Amaryllis Alpha) ---
        dhv_intersection_count = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        state['features']['[REDACTED_BY_SCRIPT]'] = dhv_intersection_count + dlv_intersection_count

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state