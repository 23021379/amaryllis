import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
SERVICE_AREAS_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = 0.0

# --- Module-level State for Performance ---
gdf_service_areas, sindex_service_areas = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_service_areas, sindex_service_areas
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        gdf_service_areas = gpd.read_file(SERVICE_AREAS_L1_ARTIFACT)
        assert gdf_service_areas.crs.to_string() == PROJECT_CRS
        sindex_service_areas = gdf_service_areas.sindex
        
        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_service_areas = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_service_areas is None:
        _initialize_module_state()
    if sindex_service_areas == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    # Add one-hot encoded features from the artifact to the list
    ohe_features = [c for c in gdf_service_areas.columns if 'demandrag_' in c or 'constraint_season_' in c]
    feature_keys.extend(ohe_features)
    
    try:
        geom = state['input_geom']
        governing_area = None
        join_method = None
        
        # --- Two-Stage, Fault-Tolerant Join (Amaryllis Alpha) ---
        # Stage 1: Primary Containment Query ('within')
        possible_matches_idx = list(sindex_service_areas.intersection(geom.bounds))
        if possible_matches_idx:
            containing_areas = gdf_service_areas.iloc[possible_matches_idx]
            containing_areas = containing_areas[containing_areas.intersects(geom)]
            if not containing_areas.empty:
                governing_area = containing_areas.iloc[0]
                join_method = 'within'
        
        # Stage 2: Fallback Proximity Query ('nearest') if Stage 1 failed
        if governing_area is None:
            _, tree_indices = sindex_service_areas.nearest(geom, return_all=False)
            governing_area = gdf_service_areas.iloc[tree_indices[0]]
            join_method = 'nearest'
            
        state['features']['[REDACTED_BY_SCRIPT]'] = 1 if join_method == 'within' else 0
        
        # --- Direct & Synthetic Feature Generation ---
        # Direct features from the L1 artifact
        state['features']['[REDACTED_BY_SCRIPT]'] = governing_area.get('[REDACTED_BY_SCRIPT]', NULL_SENTINEL)
        for col in ohe_features:
            state['features'][col] = governing_area.get(col, 0)

        # Add critical identifiers to the state for downstream consumption
        state['features']['primary_site_floc'] = governing_area.get('primary_site_floc', None)
        state['features']['[REDACTED_BY_SCRIPT]'] = governing_area.get('[REDACTED_BY_SCRIPT]', None)
            
        # Spatially derived features
        state['features']['[REDACTED_BY_SCRIPT]'] = geom.distance(governing_area.geometry.centroid) / 1000.0
        
        # Synthetic features combining with previous state
        forecast_headroom_mw = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        total_kva = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        forecast_headroom_pct = (forecast_headroom_mw * 1000) / (total_kva + 1e-6)
        state['features']['[REDACTED_BY_SCRIPT]'] = state['features']['[REDACTED_BY_SCRIPT]'] - forecast_headroom_pct
        

        # The `dist_to_nearest_sub_km` feature implies a different "nearest" substation from a different dataset.
        # The `primary_site_floc` is the ID of the substation at the *centroid* of the service area.
        # The original logic's intent was likely to check if the point-based nearest substation is the same as the area-based one.
        # Since we cannot guarantee the point-based FLOC is available, this feature is unreliable and removed.
        # The correct feature is generated in `exec_012_gandp`, which compares two different data sources for the *same* substation.
        # This logic is therefore decommissioned from this executor to prevent confusion.

        # # A previous step must populate a '[REDACTED_BY_SCRIPT]' for this to work
        # nearest_floc = state['features'].get('[REDACTED_BY_SCRIPT]', None) 
        # governing_floc = governing_area.get('primary_site_floc', None)
        # state['features']['[REDACTED_BY_SCRIPT]'] = 1 if nearest_floc != governing_floc else 0

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL if 'pct' in key else 0
                
    return state