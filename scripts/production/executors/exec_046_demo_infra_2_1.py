import os
import logging
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from pathlib import Path

# --- CONFIGURATION ---
BASE_DATA_DIR = Path(r"[REDACTED_BY_SCRIPT]") # Assuming a standard artifact location
L1_PLACES_INPUT = BASE_DATA_DIR / "L1_osm_places.gpkg" # Assumes this L1 artifact exists
TARGET_CRS = "EPSG:27700"
NULL_SENTINEL = -1.0

# --- MODULE-LEVEL STATE ---
places_tree = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global places_tree
    if places_tree is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_places = gpd.read_file(L1_PLACES_INPUT)
        assert gdf_places.crs.to_string() == TARGET_CRS, "[REDACTED_BY_SCRIPT]"
        
        # Filter for high-value targets to build a lean index
        major_places = gdf_places[gdf_places['super_class'].isin(['city', 'town', 'national_capital'])]
        if major_places.empty:
            raise ValueError("[REDACTED_BY_SCRIPT]")
            
        coords = np.array(list(zip(major_places.geometry.x, major_places.geometry.y)))
        places_tree = cKDTree(coords)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        places_tree = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if places_tree == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        site_coord = (state['input_geom'].x, state['input_geom'].y)
        distance, _ = places_tree.query(site_coord, k=1)
        state['features']['[REDACTED_BY_SCRIPT]'] = distance / 1000

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            
    return state