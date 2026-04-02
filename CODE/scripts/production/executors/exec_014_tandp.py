import geopandas as gpd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
# Pre-stratified L1 artifacts are now mandated
TOPOLOGICAL_ASSETS_CANONICAL_ARTIFACT = os.path.join(BASE_DATA_DIR, "NG_data", "artifacts", "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
LOCAL_NETWORK_BUFFER_METERS = 2000
NULL_SENTINEL = 0.0

# --- Module-level State for Performance ---
gdf_poles, gdf_towers, gdf_structures_all, gdf_lines = None, None, None, None
tree_poles, tree_towers, tree_structures_all = None, None, None
sindex_lines = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_poles, gdf_towers, gdf_structures_all, gdf_lines
    global tree_poles, tree_towers, tree_structures_all, sindex_lines
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_points_master = gpd.read_file(TOPOLOGICAL_ASSETS_CANONICAL_ARTIFACT, layer='points')
        gdf_lines_master = gpd.read_file(TOPOLOGICAL_ASSETS_CANONICAL_ARTIFACT, layer='lines')
        assert gdf_points_master.crs.to_string() == PROJECT_CRS and gdf_lines_master.crs.to_string() == PROJECT_CRS

        # Filter master points for 33kV structures
        gdf_structures_all = gdf_points_master[gdf_points_master['voltage'] == '33kv'].copy()
        gdf_poles = gdf_structures_all[gdf_structures_all['asset_tag'] == 'pole'].copy()
        gdf_towers = gdf_structures_all[gdf_structures_all['asset_tag'] == 'tower'].copy()

        # Build BallTrees for point geometries
        for gdf, tree_name in [(gdf_poles, 'tree_poles'), (gdf_towers, 'tree_towers'), (gdf_structures_all, 'tree_structures_all')]:
            coords = np.deg2rad(np.array(list(zip(gdf.geometry.y, gdf.geometry.x))))
            globals()[tree_name] = BallTree(coords, metric='haversine')

        # Filter master lines for 33kV lines and build sindex
        gdf_lines = gdf_lines_master[gdf_lines_master['voltage'] == '33kv'].copy()
        sindex_lines = gdf_lines.sindex

        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        tree_structures_all = "INIT_FAILED"


def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if tree_structures_all is None:
        _initialize_module_state()
    if tree_structures_all == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'ohl_local_structure_count',
        'ohl_local_tower_count', 'ohl_local_tower_ratio', '[REDACTED_BY_SCRIPT]'
    ]

    try:
        geom = state['input_geom']
        app_coords_rad = np.deg2rad([[geom.y, geom.x]])
        
        # --- Specific Proximity Features ---
        dist_pole_rad, _ = tree_poles.query(app_coords_rad, k=1)
        state['features']['[REDACTED_BY_SCRIPT]'] = dist_pole_rad[0][0] * 6371000

        dist_tower_rad, _ = tree_towers.query(app_coords_rad, k=1)
        state['features']['[REDACTED_BY_SCRIPT]'] = dist_tower_rad[0][0] * 6371000

        # --- Local Network Character Features ---
        radius_rad = LOCAL_NETWORK_BUFFER_METERS / 6371000
        struct_count = tree_structures_all.query_radius(app_coords_rad, r=radius_rad, count_only=True)[0]
        tower_count = tree_towers.query_radius(app_coords_rad, r=radius_rad, count_only=True)[0]
        
        state['features']['ohl_local_structure_count'] = struct_count
        state['features']['ohl_local_tower_count'] = tower_count
        state['features']['ohl_local_tower_ratio'] = tower_count / (struct_count + 1e-6)

        # --- Reconciliation Feature (Performant Re-architecture) ---
        # Find the single nearest structure, then find its distance to the nearest line.
        _, nearest_struct_idx = tree_structures_all.query(app_coords_rad, k=1)
        nearest_structure_geom = gdf_structures_all.geometry.iloc[nearest_struct_idx[0][0]]
        state['features']['[REDACTED_BY_SCRIPT]'] = gdf_lines.distance(nearest_structure_geom).min()
        
    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL
                
    return state