import geopandas as gpd
import pandas as pd
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
LCT_SPATIAL_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
BUFFER_RADIUS_METERS = 5000
NULL_SENTINEL = 0.0 # Counts and sums default to 0

# --- Module-level State for Performance ---
gdf_lct_points, gdf_primary_subs = None, None
sindex_lct_points, sindex_primary_subs = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_primary_subs, sindex_primary_subs
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load the single, authoritative, pre-aggregated LCT artifact
        gdf_primary_subs = gpd.read_file(LCT_SPATIAL_L1_ARTIFACT, engine='pyarrow')
        assert gdf_primary_subs.crs.to_string() == PROJECT_CRS
        sindex_primary_subs = gdf_primary_subs.sindex
        
        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_primary_subs = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_primary_subs is None:
        _initialize_module_state()
    if sindex_primary_subs == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    # Define feature keys for schema consistency on failure
    buffer_feature_keys = [
        '[REDACTED_BY_SCRIPT]', 'lct_total_connections_in_5km', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'lct_ev_connections_in_5km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    primary_sub_feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    all_feature_keys = buffer_feature_keys + primary_sub_feature_keys

    try:
        geom = state['input_geom']
        
        # --- Ambient LCT Density Features (5km Buffer) ---
        buffer = geom.buffer(BUFFER_RADIUS_METERS)
        possible_matches_idx = list(sindex_lct_points.intersection(buffer.bounds))
        lcts_in_buffer = gdf_lct_points.iloc[possible_matches_idx]
        lcts_in_buffer = lcts_in_buffer[lcts_in_buffer.intersects(buffer)]
        
        if not lcts_in_buffer.empty:
            state['features']['[REDACTED_BY_SCRIPT]'] = len(lcts_in_buffer)
            state['features']['lct_total_connections_in_5km'] = lcts_in_buffer['lct_connections'].sum()
            state['features']['[REDACTED_BY_SCRIPT]'] = lcts_in_buffer['import'].sum()
            state['features']['[REDACTED_BY_SCRIPT]'] = lcts_in_buffer['export'].sum()
            
            demand_conns = lcts_in_buffer[lcts_in_buffer['category'] == 'Demand']['lct_connections'].sum()
            gen_conns = lcts_in_buffer[lcts_in_buffer['category'] == 'Generation']['lct_connections'].sum()
            
            state['features']['[REDACTED_BY_SCRIPT]'] = demand_conns
            state['features']['[REDACTED_BY_SCRIPT]'] = gen_conns
            state['features']['lct_ev_connections_in_5km'] = lcts_in_buffer[lcts_in_buffer['type'] == 'Ev Charging Point']['lct_connections'].sum()
            state['features']['[REDACTED_BY_SCRIPT]'] = lcts_in_buffer[lcts_in_buffer['type'] == 'Solar']['lct_connections'].sum()
            state['features']['[REDACTED_BY_SCRIPT]'] = gen_conns / (demand_conns + 1) # Guard against division by zero
        else:
            for key in buffer_feature_keys: state['features'][key] = NULL_SENTINEL

        # --- Primary Substation LCT Profile Features (Nearest Substation Lookup) ---
        nearest_idx = list(sindex_primary_subs.nearest(geom))
        if nearest_idx:
            nearest_sub_profile = gdf_primary_subs.iloc[nearest_idx[0]]
            for key in primary_sub_feature_keys:
                state['features'][key] = nearest_sub_profile.get(key, NULL_SENTINEL)
        else:
            for key in primary_sub_feature_keys: state['features'][key] = NULL_SENTINEL

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in all_feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state