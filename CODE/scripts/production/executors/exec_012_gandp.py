import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
# A lightweight spatial artifact for finding the nearest substation ID
PRIMARY_SUBSTATION_CANONICAL_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = 0.0

# --- Module-level State for Performance ---
gdf_sub_locations, sindex_sub_locations = None, None
df_bio_profiles = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_sub_locations, sindex_sub_locations, df_bio_profiles
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load the single canonical artifact containing both spatial and biography data.
        gdf_master = gpd.read_file(PRIMARY_SUBSTATION_CANONICAL_ARTIFACT, engine='pyarrow')
        assert gdf_master.crs.to_string() == PROJECT_CRS
        
        # Populate both module variables from the same source.
        gdf_sub_locations = gdf_master
        sindex_sub_locations = gdf_master.sindex
        df_bio_profiles = gdf_master.set_index('[REDACTED_BY_SCRIPT]')
        
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_sub_locations = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_sub_locations is None:
        _initialize_module_state()
    if sindex_sub_locations == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    feature_keys = [
        'substation_age_at_submission', 'time_since_last_assessment_days', 'is_hot_site',
        '[REDACTED_BY_SCRIPT]', 'has_demand_data', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'kva_per_transformer'
    ]

    try:
        geom = state['input_geom']
        submission_date = pd.to_datetime(f"{state['submission_year']}-01-01")

        # Step 1: Spatially find the nearest substation to get its ID
        nearest_idx = list(sindex_sub_locations.nearest(geom))
        if not nearest_idx:
            raise ValueError("[REDACTED_BY_SCRIPT]")
        
        substation_id = gdf_sub_locations.iloc[nearest_idx[0]]['[REDACTED_BY_SCRIPT]']
        
        # Step 2: Look up the pre-processed biography for that substation
        bio = df_bio_profiles.loc[substation_id]
        
        # Step 3: Synthesize point-in-time and interaction features
        state['features']['substation_age_at_submission'] = submission_date.year - bio['datecommissioned'].year if pd.notna(bio['datecommissioned']) else NULL_SENTINEL
        state['features']['time_since_last_assessment_days'] = (submission_date - bio['assessmentdate']).days if pd.notna(bio['assessmentdate']) else NULL_SENTINEL
        state['features']['is_hot_site'] = 1 if bio.get('siteclassification') == 'HOT' else 0
        
        # Interaction features require data from previous executors, accessed via the state dict
        prev_tx_count = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        total_kva = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        
        bio_tx_count = bio.get('powertransformercount', 0)
        state['features']['[REDACTED_BY_SCRIPT]'] = 1 if bio_tx_count != prev_tx_count else 0
        state['features']['has_demand_data'] = 1 if pd.notna(bio.get('maxdemandwinter')) else 0
        
        # Calculate loading ratios, converting MW (demand) to kVA
        state['features']['[REDACTED_BY_SCRIPT]'] = (bio.get('maxdemandwinter', 0) * 1000) / (total_kva + 1e-6)
        state['features']['[REDACTED_BY_SCRIPT]'] = (bio.get('maxdemandsummer', 0) * 1000) / (total_kva + 1e-6)
        state['features']['kva_per_transformer'] = total_kva / (bio_tx_count + 1e-6)

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state