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
gdf_dhv_lines, gdf_dhv_poles, gdf_dhv_corridors, gdf_subs = None, None, None, None
sindex_dhv_lines, sindex_dhv_corridors = None, None
tree_dhv_poles, tree_subs = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_dhv_lines, gdf_dhv_poles, gdf_dhv_corridors, gdf_subs
    global sindex_dhv_lines, sindex_dhv_corridors, tree_dhv_poles, tree_subs
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load and triage topological assets for DHV (assuming 11kV)
        gdf_points_master = gpd.read_file(TOPOLOGICAL_ASSETS_CANONICAL_ARTIFACT, layer='points')
        gdf_lines_master = gpd.read_file(TOPOLOGICAL_ASSETS_CANONICAL_ARTIFACT, layer='lines')
        
        gdf_dhv_lines = gdf_lines_master[gdf_lines_master['voltage'] == '11kv'].copy()
        sindex_dhv_lines = gdf_dhv_lines.sindex
        
        gdf_dhv_poles = gdf_points_master[gdf_points_master['voltage'] == '11kv'].copy()
        coords_poles = np.deg2rad(np.array(list(zip(gdf_dhv_poles.geometry.y, gdf_dhv_poles.geometry.x))))
        tree_dhv_poles = BallTree(coords_poles, metric='haversine')

        gdf_dhv_corridors = gdf_dhv_lines.copy()
        gdf_dhv_corridors['geometry'] = gdf_dhv_corridors.apply(lambda row: row.geometry.buffer(row.buffer_m), axis=1)
        sindex_dhv_corridors = gdf_dhv_corridors.sindex

        # Load authoritative substation data
        gdf_subs = gpd.read_parquet(PRIMARY_SUBSTATIONS_CANONICAL_ARTIFACT)
        assert gdf_subs.crs.to_string() == PROJECT_CRS
        coords_subs = np.deg2rad(np.array(list(zip(gdf_subs.geometry.y, gdf_subs.geometry.x))))
        tree_subs = BallTree(coords_subs, metric='haversine')

        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        tree_dhv_poles = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if tree_dhv_poles is None:
        _initialize_module_state()
    if tree_dhv_poles == "INIT_FAILED":
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

        # --- Foundational Density Features ---
        line_candidates_idx = list(sindex_dhv_lines.intersection(buffer.bounds))
        line_candidates = gdf_dhv_lines.iloc[line_candidates_idx]
        clipped_lines = gpd.clip(line_candidates, buffer)
        total_line_length_km = clipped_lines.geometry.length.sum() / 1000
        state['features']['[REDACTED_BY_SCRIPT]'] = total_line_length_km / buffer_area_km2
        
        radius_rad = LOCAL_NETWORK_BUFFER_METERS / 6371000
        pole_count = tree_dhv_poles.query_radius(app_coords_rad, r=radius_rad, count_only=True)[0]
        state['features']['[REDACTED_BY_SCRIPT]'] = pole_count / buffer_area_km2

        # --- Topological "Alpha" Features ---
        # 1. Connection Path Intersection
        _, sub_idx = tree_subs.query(app_coords_rad, k=1)
        nearest_sub_geom = gdf_subs.geometry.iloc[sub_idx[0][0]]
        connection_path = LineString([geom, nearest_sub_geom])
        
        path_candidates_idx = list(sindex_dhv_corridors.intersection(connection_path.bounds))
        path_candidates = gdf_dhv_corridors.iloc[path_candidates_idx]
        intersection_count = path_candidates.intersects(connection_path).sum()
        state['features']['[REDACTED_BY_SCRIPT]'] = intersection_count
        
        # 2. Network Complexity
        line_segment_count = len(clipped_lines)
        total_line_length_m = total_line_length_km * 1000
        state['features']['[REDACTED_BY_SCRIPT]'] = (line_segment_count / (total_line_length_m + 1e-6))

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state