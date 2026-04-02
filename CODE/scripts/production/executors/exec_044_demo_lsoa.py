import os
import logging
import numpy as np
import geopandas as gpd

# --- CONFIGURATION ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
L2_LSOA_INPUT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
L2_LPA_AGG_INPUT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
NULL_SENTINEL = -1.0

# --- MODULE-LEVEL STATE ---
gdf_lsoa, gdf_lpa_agg, sindex_lsoa, sindex_lpa = None, None, None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_lsoa, gdf_lpa_agg, sindex_lsoa, sindex_lpa
    if sindex_lsoa is not None and sindex_lpa is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_lsoa = gpd.read_file(L2_LSOA_INPUT)
        sindex_lsoa = gdf_lsoa.sindex
        gdf_lpa_agg = gpd.read_file(L2_LPA_AGG_INPUT)
        sindex_lpa = gdf_lpa_agg.sindex
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_lsoa = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_lsoa == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        geom = state['input_geom']
        
        # --- Site-Level LSOA Feature Lookup ---
        lsoa_idx = list(sindex_lsoa.query(geom, predicate='intersects'))
        if not lsoa_idx: raise ValueError("[REDACTED_BY_SCRIPT]")
        site_lsoa_data = gdf_lsoa.iloc[lsoa_idx[0]]

        # --- LPA-Level Aggregate Feature Lookup ---
        lpa_idx = list(sindex_lpa.query(geom, predicate='intersects'))
        if not lpa_idx: raise ValueError("[REDACTED_BY_SCRIPT]")
        lpa_agg_data = gdf_lpa_agg.iloc[lpa_idx[0]]

        # --- On-Demand Synthesis ---
        # Calculate site-level property value index
        bands = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7}
        num = sum(w * site_lsoa_data.get(f'ct_band_{b}', 0) for b, w in bands.items())
        den = site_lsoa_data.get('ct_total', 0)
        site_prop_idx = num / den if den > 0 else 0
        state['features']['[REDACTED_BY_SCRIPT]'] = site_prop_idx

        # Calculate Site-LPA Delta features
        lpa_prop_idx_mean = lpa_agg_data.get('[REDACTED_BY_SCRIPT]', 0)
        state['features']['delta_property_value'] = site_prop_idx / (lpa_prop_idx_mean + 0.01)

        # Add other direct LSOA features
        state['features']['site_lsoa_ahah_rank'] = site_lsoa_data.get('ahah_rank', NULL_SENTINEL)
        
        # Add other pre-aggregated LPA features
        state['features']['[REDACTED_BY_SCRIPT]'] = lpa_agg_data.get('[REDACTED_BY_SCRIPT]', NULL_SENTINEL)

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)

    return state