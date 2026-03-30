import geopandas as gpd
import pandas as pd
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
# Artifact from previous LCT step (for nearest sub lookup)
LCT_CANONICAL_ARTIFACT = os.path.join(BASE_DATA_DIR, "NG_data", "artifacts", "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = -1.0

# --- Module-level State for Performance ---
gdf_primary_subs, df_direct_profiles = None, None
sindex_primary_subs = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_primary_subs, sindex_primary_subs
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load the single, authoritative LCT artifact for all lookups.
        # It serves as both the spatial and profile data source.
        gdf_primary_subs = gpd.read_file(LCT_CANONICAL_ARTIFACT, engine='pyarrow').set_index('primary_sub_id')
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

    # Define all new feature keys for schema consistency
    feature_keys = [
        'lct_reconciliation_delta_connections', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]

    try:
        geom = state['input_geom']
        
        # Step 1: Spatially identify the nearest primary substation to get its ID
        nearest_idx = list(sindex_primary_subs.nearest(geom))
        if not nearest_idx:
            raise ValueError("[REDACTED_BY_SCRIPT]")
        
        nearest_sub = gdf_primary_subs.iloc[nearest_idx[0]]
        substation_id = nearest_sub['[REDACTED_BY_SCRIPT]']
        
        # Step 2: Use the ID to perform a direct, non-spatial lookup for the new features
        direct_profile = df_direct_profiles.loc[substation_id]
        
        # Step 3: Retrieve features from the PREVIOUS step, now in the state dictionary
        secondary_total_conns = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        secondary_total_export = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        
        # Step 4: Synthesize the "Alpha" features
        direct_total_conns = direct_profile.get('[REDACTED_BY_SCRIPT]', 0)
        direct_total_export = direct_profile.get('[REDACTED_BY_SCRIPT]', 0)
        max_secondary_conns = direct_profile.get('[REDACTED_BY_SCRIPT]', 0)

        state['features']['lct_reconciliation_delta_connections'] = direct_total_conns - secondary_total_conns
        state['features']['[REDACTED_BY_SCRIPT]'] = direct_total_export - secondary_total_export
        state['features']['[REDACTED_BY_SCRIPT]'] = max_secondary_conns / (direct_total_conns + 1)
        state['features']['[REDACTED_BY_SCRIPT]'] = secondary_total_conns / (direct_total_conns + 1)
        
    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state
