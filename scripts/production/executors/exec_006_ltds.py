import geopandas as gpd
import pandas as pd
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
LTDS_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RADIUS_METERS = 10000
NULL_SENTINEL = 0.0 # Counts default to 0

# --- Module-level State for Performance ---
gdf_ltds_master = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_ltds_master
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_ltds_master = gpd.read_file(LTDS_L1_ARTIFACT, engine='pyarrow')
        
        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        assert gdf_ltds_master.crs.to_string() == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"
        
        # Ensure temporal columns are correctly typed for comparison
        gdf_ltds_master['[REDACTED_BY_SCRIPT]'] = pd.to_datetime(gdf_ltds_master['[REDACTED_BY_SCRIPT]'])
        gdf_ltds_master['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(gdf_ltds_master['[REDACTED_BY_SCRIPT]'], errors='coerce')

        logging.info(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        gdf_ltds_master = "INIT_FAILED"

def execute(state: dict) -> dict:
    """
    Calculates features based on forward-looking grid upgrades from LTDS,
    while strictly enforcing temporal integrity.
    """
    if gdf_ltds_master is None:
        _initialize_module_state()
    if isinstance(gdf_ltds_master, str) and gdf_ltds_master == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    # Define feature keys for schema consistency
    feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'ltds_count_in_10km',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]

    try:
        app_geom = state['input_geom']
        submission_date = pd.to_datetime(f"{state['submission_year']}-01-01")

        # CRITICAL TEMPORAL GUARDRAIL (Pattern 3: Feature Leakage)
        # 1. Filter to only include LTDS publications that existed at the time of application.
        valid_ltds = gdf_ltds_master[gdf_ltds_master['[REDACTED_BY_SCRIPT]'] <= submission_date]
        
        if valid_ltds.empty:
            # No LTDS data was available at this point in time. This is valid.
            for key in feature_keys:
                state['features'][key] = NULL_SENTINEL
            return state

        # 2. Of the valid publications, select the most recent one.
        latest_pub_date = valid_ltds['[REDACTED_BY_SCRIPT]'].max()
        point_in_time_ltds = valid_ltds[valid_ltds['[REDACTED_BY_SCRIPT]'] == latest_pub_date].copy()

        # --- Feature Synthesis on the temporally-correct data slice ---
        # Proximity Features
        point_in_time_ltds['distance'] = point_in_time_ltds.geometry.distance(app_geom)
        nearest = point_in_time_ltds.loc[point_in_time_ltds['distance'].idxmin()]
        
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest['distance']
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest['[REDACTED_BY_SCRIPT]']
        state['features']['[REDACTED_BY_SCRIPT]'] = nearest['[REDACTED_BY_SCRIPT]'] - submission_date.year

        # Density & Character Features (10km radius)
        buffer = app_geom.buffer(RADIUS_METERS)
        upgrades_in_10km = point_in_time_ltds[point_in_time_ltds.intersects(buffer)]
        
        state['features']['ltds_count_in_10km'] = len(upgrades_in_10km)
        
        if not upgrades_in_10km.empty:
            asset_text = upgrades_in_10km['[REDACTED_BY_SCRIPT]'].str.lower().fillna('')
            state['features']['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('[REDACTED_BY_SCRIPT]').sum()
            state['features']['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('transformer|tx').sum()
            state['features']['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('[REDACTED_BY_SCRIPT]').sum()
        else:
            state['features']['[REDACTED_BY_SCRIPT]'] = 0
            state['features']['[REDACTED_BY_SCRIPT]'] = 0
            state['features']['[REDACTED_BY_SCRIPT]'] = 0
            
    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state