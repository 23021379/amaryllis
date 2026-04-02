import geopandas as gpd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
OHL_CANONICAL_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
LOCAL_NETWORK_BUFFER_METERS = 2000
NULL_SENTINEL = 0.0

# --- Module-level State for Performance ---
gdf_towers, gdf_corridors = None, None
tree_towers, sindex_corridors = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_towers, gdf_corridors, tree_towers, sindex_corridors
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_points_master = gpd.read_file(OHL_CANONICAL_ARTIFACT, layer='points')
        gdf_lines_master = gpd.read_file(OHL_CANONICAL_ARTIFACT, layer='lines')
        assert gdf_points_master.crs.to_string() == PROJECT_CRS and gdf_lines_master.crs.to_string() == PROJECT_CRS

        # Filter for 132kV towers and build BallTree
        gdf_towers = gdf_points_master[gdf_points_master['voltage'] == '132kv'].copy()
        coords_towers_rad = np.deg2rad(np.array(list(zip(gdf_towers.geometry.y, gdf_towers.geometry.x))))
        tree_towers = BallTree(coords_towers_rad, metric='haversine')

        # Filter for 132kV lines, generate corridors, and build sindex
        gdf_lines_132kv = gdf_lines_master[gdf_lines_master['voltage'] == '132kv'].copy()
        gdf_corridors = gdf_lines_132kv.copy()
        gdf_corridors['geometry'] = gdf_corridors.apply(lambda row: row.geometry.buffer(row.buffer_m), axis=1)
        sindex_corridors = gdf_corridors.sindex

        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        tree_towers = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if tree_towers is None:
        _initialize_module_state()
    if tree_towers == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'ohl_nearest_voltage', '[REDACTED_BY_SCRIPT]'
    ]

    try:
        geom = state['input_geom']
        app_coords_rad = np.deg2rad([[geom.y, geom.x]])

        # --- Core 132kV Features ---
        dist_tower_rad, _ = tree_towers.query(app_coords_rad, k=1)
        dist_tower_m = dist_tower_rad[0][0] * 6371000
        state['features']['[REDACTED_BY_SCRIPT]'] = dist_tower_m

        is_within = gdf_corridors.iloc[list(sindex_corridors.intersection(geom.bounds))].intersects(geom).any()
        state['features']['[REDACTED_BY_SCRIPT]'] = 1 if is_within else 0
        state['features']['[REDACTED_BY_SCRIPT]'] = 0.0 if is_within else geom.distance(gdf_corridors.unary_union)

        radius_rad = LOCAL_NETWORK_BUFFER_METERS / 6371000
        tower_count_2km = tree_towers.query_radius(app_coords_rad, r=radius_rad, count_only=True)[0]
        buffer_area_km2 = np.pi * ((LOCAL_NETWORK_BUFFER_METERS / 1000) ** 2)
        state['features']['[REDACTED_BY_SCRIPT]'] = tower_count_2km
        state['features']['[REDACTED_BY_SCRIPT]'] = tower_count_2km / buffer_area_km2

        # --- Cross-Voltage "Alpha" Features ---
        dist_33kv_m = state['features'].get('[REDACTED_BY_SCRIPT]', NULL_SENTINEL)
        dist_132kv_m = state['features']['[REDACTED_BY_SCRIPT]']
        state['features']['[REDACTED_BY_SCRIPT]'] = dist_33kv_m / (dist_132kv_m + 1e-6)

        state['features']['ohl_nearest_voltage'] = 132 if dist_132kv_m < dist_33kv_m else 33
        
        tower_count_33kv = state['features'].get('ohl_local_tower_count', 0)
        state['features']['[REDACTED_BY_SCRIPT]'] = tower_count_33kv + tower_count_2km

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL
                
    return state