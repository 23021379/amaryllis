import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
# This L1 artifact contains the pre-aggregated transformer asset stats per primary substation
PRIMARY_SUB_ASSETS_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = 0.0 # Asset counts and stats default to 0

# --- Module-level State for Performance ---
gdf_primary_subs, sindex_primary_subs = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_primary_subs, sindex_primary_subs
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load the pre-aggregated spatial artifact and set index for fast lookups
        gdf_primary_subs = gpd.read_file(PRIMARY_SUB_ASSETS_L1_ARTIFACT, engine='pyarrow').set_index('[REDACTED_BY_SCRIPT]')
        
        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        assert gdf_primary_subs.crs.to_string() == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"
        
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

    feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]

    try:
        geom = state['input_geom']
        
        # Step 1: Spatially identify the nearest primary substation
        nearest_idx = list(sindex_primary_subs.nearest(geom))
        if not nearest_idx:
            raise ValueError("[REDACTED_BY_SCRIPT]")
        
        # Step 2: Retrieve the pre-aggregated profile using the index (substation ID)
        substation_id = gdf_primary_subs.index[nearest_idx[0]]
        asset_profile = gdf_primary_subs.loc[substation_id]
        
        # Step 3: Populate features directly from the looked-up profile
        for key in feature_keys:
            state['features'][key] = asset_profile.get(key, NULL_SENTINEL)

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL
                
    return state