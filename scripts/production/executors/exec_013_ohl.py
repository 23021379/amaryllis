import geopandas as gpd
import numpy as np
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
OHL_CANONICAL_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
DENSITY_BUFFER_KM = 2
NULL_SENTINEL = 0.0

# --- Module-level State for Performance ---
gdf_ohl_lines, gdf_ohl_corridors = None, None
sindex_ohl_lines, sindex_ohl_corridors = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_ohl_lines, gdf_ohl_corridors, sindex_ohl_lines, sindex_ohl_corridors
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_master = gpd.read_file(OHL_CANONICAL_ARTIFACT, layer='lines')
        assert gdf_master.crs.to_string() == PROJECT_CRS

        # Surgically filter the master artifact for 33kV lines
        gdf_ohl_lines = gdf_master[gdf_master['voltage'] == '33kv'].copy()
        sindex_ohl_lines = gdf_ohl_lines.sindex
        
        # Generate corridor polygons on-the-fly from the canonical source
        gdf_ohl_corridors = gdf_ohl_lines.copy()
        gdf_ohl_corridors['geometry'] = gdf_ohl_corridors.apply(lambda row: row.geometry.buffer(row.buffer_m), axis=1)
        sindex_ohl_corridors = gdf_ohl_corridors.sindex
        
        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_ohl_lines = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_ohl_lines is None:
        _initialize_module_state()
    if sindex_ohl_lines == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]

    try:
        geom = state['input_geom']
        
        # --- Proximity & Sterilization Features (using Corridors) ---
        # Find if the point itself is within a corridor
        possible_matches_idx = list(sindex_ohl_corridors.intersection(geom.bounds))
        containing_corridors = gdf_ohl_corridors.iloc[possible_matches_idx]
        is_within = containing_corridors.intersects(geom).any()
        
        state['features']['[REDACTED_BY_SCRIPT]'] = 1 if is_within else 0
        
        # Calculate distance to nearest corridor boundary
        if is_within:
            state['features']['[REDACTED_BY_SCRIPT]'] = 0.0
        else:
            # Use the more robust sindex.nearest for finding the closest geometry
            _, tree_indices = sindex_ohl_corridors.nearest(geom, return_all=False)
            nearest_corridor = gdf_ohl_corridors.geometry.iloc[tree_indices[0]]
            state['features']['[REDACTED_BY_SCRIPT]'] = geom.distance(nearest_corridor)

        # --- Network Topology & Density Features (using Lines) ---
        buffer = geom.buffer(DENSITY_BUFFER_KM * 1000)
        buffer_area_km2 = np.pi * (DENSITY_BUFFER_KM ** 2)
        
        candidate_lines_idx = list(sindex_ohl_lines.intersection(buffer.bounds))
        if candidate_lines_idx:
            candidate_lines = gdf_ohl_lines.iloc[candidate_lines_idx]
            # Clip the candidate lines to the buffer to get the precise length inside
            clipped_lines = gpd.clip(candidate_lines, buffer)
            total_length_km = clipped_lines.geometry.length.sum() / 1000
            state['features']['[REDACTED_BY_SCRIPT]'] = total_length_km / buffer_area_km2
        else:
            state['features']['[REDACTED_BY_SCRIPT]'] = 0.0

        # --- Final Interaction Feature ---
        dist_to_sub_km = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        dist_to_ohl_m = state['features']['[REDACTED_BY_SCRIPT]']
        
        if dist_to_sub_km > 0:
            state['features']['[REDACTED_BY_SCRIPT]'] = dist_to_ohl_m / (dist_to_sub_km * 1000)
        else:
            state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL
                
    return state