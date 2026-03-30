import os
import logging
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
L2_PRIMARY_SUBS_ENRICHED_INPUT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
TARGET_CRS = "EPSG:27700"
NULL_SENTINEL = 0 # Downstream count of 0 is a valid feature, not an error

# --- MODULE-LEVEL STATE ---
gdf_primary_enriched, spatial_index = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_primary_enriched, spatial_index
    if spatial_index is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_primary_enriched = gpd.read_file(L2_PRIMARY_SUBS_ENRICHED_INPUT)
        assert gdf_primary_enriched.crs.to_string() == TARGET_CRS, "[REDACTED_BY_SCRIPT]"
        
        coords = np.array(list(zip(gdf_primary_enriched.geometry.x, gdf_primary_enriched.geometry.y)))
        spatial_index = cKDTree(coords)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        spatial_index = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if spatial_index is None: _initialize_module_state()
    if spatial_index == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        site_geom = state['input_geom']
        site_coord = (site_geom.x, site_geom.y)

        # Perform a single nearest-neighbor query
        _, idx_nn1 = spatial_index.query(site_coord, k=1)
        
        # Look up the pre-computed value from the enriched artifact
        downstream_count = gdf_primary_enriched.iloc[idx_nn1]['[REDACTED_BY_SCRIPT]'].values[0]
        
        state['features']['[REDACTED_BY_SCRIPT]'] = int(downstream_count)

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        state['features']['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            
    return state