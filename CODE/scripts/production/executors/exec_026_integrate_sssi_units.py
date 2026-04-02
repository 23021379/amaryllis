import logging
import pandas as pd
import geopandas as gpd
import os
from datetime import datetime

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
SSSI_UNITS_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "geopackage", "SSSI_Units_L1.gpkg")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL_INT = 0
NULL_SENTINEL_FLOAT = 0.0
ACUTE_ZONE_METERS = 2000

# --- Module-level State for Performance ---
gdf_sssi_units, sindex_sssi_units = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_sssi_units, sindex_sssi_units
    if gdf_sssi_units is not None:
        return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf = gpd.read_file(SSSI_UNITS_L1_ARTIFACT)

        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        if gdf.crs.to_string() != PROJECT_CRS:
            logging.error(f"[REDACTED_BY_SCRIPT]")
            sindex_sssi_units = "INIT_FAILED"
            return
        
        # Pre-process dates for performance
        gdf['cond_date_dt'] = pd.to_datetime(gdf['cond_date'], errors='coerce')

        gdf_sssi_units = gdf
        sindex_sssi_units = gdf_sssi_units.sindex
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_sssi_units = "INIT_FAILED"

def execute(state: dict) -> dict:
    """
    Calculates SSSI Unit condition features and data reconciliation checks.
    """
    if sindex_sssi_units is None:
        _initialize_module_state()
    if sindex_sssi_units == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        geom = state['input_geom']
        features = state['features']
        
        # --- Nearest Unit Features ---
        nearest_idx = sindex_sssi_units.nearest(geom, return_all=False)[1][0]
        nearest_unit = gdf_sssi_units.iloc[nearest_idx]
        
        features['sssi_unit_dist_to_nearest_m'] = geom.distance(nearest_unit.geometry)
        features['sssi_unit_nearest_condition'] = nearest_unit.get('condition', 'UNKNOWN')
        
        # Recency Calculation (Temporal Integrity)
        submission_year = state.get('submission_year')
        cond_date = nearest_unit.get('cond_date_dt')
        if pd.notna(cond_date) and submission_year is not None:
            # Assume submission date is mid-year for calculation
            submission_datetime = datetime(submission_year, 6, 30)
            recency_days = (submission_datetime - cond_date).days
            features['[REDACTED_BY_SCRIPT]'] = recency_days
        else:
            features['[REDACTED_BY_SCRIPT]'] = -1 # Mandated null sentinel

        # --- Worst-Case Condition in Acute Zone ---
        buffer = geom.buffer(ACUTE_ZONE_METERS)
        possible_matches_idx = list(sindex_sssi_units.intersection(buffer.bounds))
        candidates = gdf_sssi_units.iloc[possible_matches_idx]
        candidates_in_zone = candidates[candidates.intersects(buffer)]
        
        if not candidates_in_zone.empty:
            worst_condition = candidates_in_zone['condition_ordinal'].max()
            features['sssi_unit_worst_condition_in_2km'] = int(worst_condition)
        else:
            features['sssi_unit_worst_condition_in_2km'] = NULL_SENTINEL_INT

        # --- Reconciliation Feature ---
        # Consume feature from prior SSSI executor, assuming it exists
        parent_name = features.get('sssi_nearest_name', '').strip().lower()
        child_name = nearest_unit.get('sssi_name', '').strip().lower()
        
        if parent_name and child_name:
             features['[REDACTED_BY_SCRIPT]'] = 1 if parent_name != child_name else 0
        else:
             features['[REDACTED_BY_SCRIPT]'] = 1 # Mismatch if either is missing

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)

    return state