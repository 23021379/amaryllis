import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os
from shapely.geometry import LineString, Point

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
IDNO_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
LPA_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
TRANSFORMER_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = -1.0

# --- Module-level State for Performance ---
gdf_idno, gdf_lpa, gdf_transformers = None, None, None
sindex_idno, sindex_lpa, sindex_transformers = None, None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_idno, gdf_lpa, gdf_transformers, sindex_idno, sindex_lpa, sindex_transformers
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load IDNO data
        gdf_idno = gpd.read_file(IDNO_L1_ARTIFACT)
        assert gdf_idno.crs.to_string() == PROJECT_CRS
        sindex_idno = gdf_idno.sindex
        
        # Load enriched LPA data
        gdf_lpa = gpd.read_file(LPA_L1_ARTIFACT)
        assert gdf_lpa.crs.to_string() == PROJECT_CRS
        sindex_lpa = gdf_lpa.sindex
        
        # Load transformer data
        gdf_transformers = gpd.read_file(TRANSFORMER_L1_ARTIFACT)
        assert gdf_transformers.crs.to_string() == PROJECT_CRS
        sindex_transformers = gdf_transformers.sindex
        
        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        # Set a sentinel to prevent execution
        sindex_idno = "INIT_FAILED" 

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_idno is None:
        _initialize_module_state()
        if sindex_idno == "INIT_FAILED":
            state['errors'].append("[REDACTED_BY_SCRIPT]")
            return state

    try:
        geom = state['input_geom']
        
        # --- Core IDNO Features ---
        possible_matches = gdf_idno.iloc[list(sindex_idno.intersection(geom.bounds))]
        containing_idno = possible_matches[possible_matches.intersects(geom)]
        is_within = not containing_idno.empty
        state['features']['idno_is_within'] = int(is_within)
        
        if is_within:
            state['features']['[REDACTED_BY_SCRIPT]'] = 0.0
        else:
            state['features']['[REDACTED_BY_SCRIPT]'] = gdf_idno.distance(geom).min()

        buffer_5km = geom.buffer(5000)
        hits_5km = sindex_idno.intersection(buffer_5km.bounds)
        state['features']['idno_count_in_5km'] = len(list(hits_5km))
        
        buffer_10km = geom.buffer(10000)
        hits_10km_idx = list(sindex_idno.intersection(buffer_10km.bounds))
        if hits_10km_idx:
            possible_matches_10km = gdf_idno.iloc[hits_10km_idx]
            intersecting_idnos = possible_matches_10km[possible_matches_10km.intersects(buffer_10km)]
            state['features']['[REDACTED_BY_SCRIPT]'] = intersecting_idnos.area.sum()
        else:
            state['features']['[REDACTED_BY_SCRIPT]'] = 0.0

        # --- Grid Interaction Features ---
        nearest_idx = list(sindex_transformers.nearest(geom))
        nearest_sub = gdf_transformers.iloc[nearest_idx[0]]
        
        sub_matches = gdf_idno.iloc[list(sindex_idno.intersection(nearest_sub.geometry.bounds))]
        state['features']['[REDACTED_BY_SCRIPT]'] = int(not sub_matches[sub_matches.intersects(nearest_sub.geometry)].empty)
        
        connection_path = LineString([geom, nearest_sub.geometry])
        path_matches = gdf_idno.iloc[list(sindex_idno.intersection(connection_path.bounds))]
        state['features']['[REDACTED_BY_SCRIPT]'] = int(path_matches.intersects(connection_path).any())

        # --- LPA Interaction Features (Robust Lookup) ---
        lpa_idx = list(sindex_lpa.intersection(geom.bounds))
        if lpa_idx:
            containing_lpa = gdf_lpa.iloc[lpa_idx]
            containing_lpa = containing_lpa[containing_lpa.intersects(geom)]
            if not containing_lpa.empty:
                lpa_row = containing_lpa.iloc[0]
                state['features']['lpa_idno_area_as_percent_of_total_area'] = lpa_row.get('idno_area_pct', NULL_SENTINEL)
                state['features']['[REDACTED_BY_SCRIPT]'] = lpa_row.get('[REDACTED_BY_SCRIPT]', NULL_SENTINEL)
            else:
                 state['features']['lpa_idno_area_as_percent_of_total_area'] = NULL_SENTINEL
                 state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        else:
            state['features']['lpa_idno_area_as_percent_of_total_area'] = NULL_SENTINEL
            state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            
    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        # Ensure schema consistency on failure
        feature_keys = [
            'idno_is_within', '[REDACTED_BY_SCRIPT]', 'idno_count_in_5km', '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            'lpa_idno_area_as_percent_of_total_area', '[REDACTED_BY_SCRIPT]'
        ]
        for key in feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL

    return state