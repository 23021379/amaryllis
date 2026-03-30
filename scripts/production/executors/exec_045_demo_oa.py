import os
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
pd.options.mode.chained_assignment = None # Suppress controlled warning

# --- CONFIGURATION ---
BASE_DATA_DIR = Path(r"[REDACTED_BY_SCRIPT]")
L2_ENRICHED_OA_INPUT = BASE_DATA_DIR / "[REDACTED_BY_SCRIPT]"
NULL_SENTINEL = -1.0
BUFFER_RADIUS_M = 1000

# --- MODULE-LEVEL STATE ---
gdf_oa_enriched, sindex_oa = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_oa_enriched, sindex_oa
    if sindex_oa is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_oa_enriched = gpd.read_file(L2_ENRICHED_OA_INPUT)
        sindex_oa = gdf_oa_enriched.sindex
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_oa = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_oa == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        geom = state['input_geom']
        feature_cols = [c for c in gdf_oa_enriched.columns if c.startswith(('oa_', 'oac_'))]

        # --- Part A: Buffered Area-Weighted Average ---
        buffer = geom.buffer(BUFFER_RADIUS_M)
        possible_matches_idx = list(sindex_oa.intersection(buffer.bounds))
        nearby_oas = gdf_oa_enriched.iloc[possible_matches_idx]
        
        # Perform overlay on the small subset of nearby OAs for performance
        intersection = gpd.overlay(gpd.GeoDataFrame(geometry=[buffer], crs=gdf_oa_enriched.crs), nearby_oas, how='intersection')
        
        if not intersection.empty:
            intersection['weight'] = intersection.geometry.area / buffer.area
            for col in feature_cols:
                weighted_val = (intersection[col] * intersection['weight']).sum()
                state['features'][f"oa_buffered_{col}"] = weighted_val

        # --- Part B: Dominant Character (Centroid) ---
        centroid_idx = list(sindex_oa.query(geom.centroid, predicate='intersects'))
        if centroid_idx:
            dominant_oa_data = gdf_oa_enriched.iloc[centroid_idx[0]]
            for col in feature_cols:
                state['features'][f"oa_centroid_{col}"] = dominant_oa_data[col]

        # --- Part C: Final SIC Synthesis ---
        pop_trend = state['features'].get('oa_buffered_oa_pop_trend', 0)
        dep_idx = state['features'].get('[REDACTED_BY_SCRIPT]', 0)
        state['features']['[REDACTED_BY_SCRIPT]'] = pop_trend * dep_idx

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)

    return state