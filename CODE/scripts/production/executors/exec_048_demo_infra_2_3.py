import os
import logging
import geopandas as gpd
from pathlib import Path

# --- CONFIGURATION ---
BASE_DATA_DIR = Path(r"[REDACTED_BY_SCRIPT]")
L1_ROADS_INPUT = BASE_DATA_DIR / "L1_osm_roads.gpkg"
L1_RAILWAYS_INPUT = BASE_DATA_DIR / "[REDACTED_BY_SCRIPT]"
TARGET_CRS = "EPSG:27700"
RADII_METERS = [1000, 2000, 5000, 10000]

# --- MODULE-LEVEL STATE ---
gdf_roads, sindex_roads, gdf_rail, sindex_rail = [None] * 4

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_roads, sindex_roads, gdf_rail, sindex_rail
    if sindex_roads is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_roads, sindex_roads = gpd.read_file(L1_ROADS_INPUT), gpd.read_file(L1_ROADS_INPUT).sindex
        gdf_rail, sindex_rail = gpd.read_file(L1_RAILWAYS_INPUT), gpd.read_file(L1_RAILWAYS_INPUT).sindex
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_roads = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_roads == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        geom = state['input_geom']
        for r_m in RADII_METERS:
            r_km = r_m // 1000
            buffer = gpd.GeoDataFrame(geometry=[geom.buffer(r_m)], crs=TARGET_CRS)
            
            # Helper for indexed clip, generalized for different infrastructure types
            def get_length(gdf, sindex, super_class_list):
                candidates_idx = list(sindex.intersection(buffer.bounds))
                if not candidates_idx: return 0.0
                candidates = gdf.iloc[candidates_idx]
                
                filtered = candidates[candidates['super_class'].isin(super_class_list)]
                if filtered.empty: return 0.0
                
                clipped = gpd.clip(filtered, buffer)
                return clipped.length.sum() / 1000 if not clipped.empty else 0.0

            # Calculate features using the helper
            state['features'][f'[REDACTED_BY_SCRIPT]'] = get_length(gdf_roads, sindex_roads, ['major_road'])
            state['features'][f'[REDACTED_BY_SCRIPT]'] = get_length(gdf_rail, sindex_rail, ['rail', 'light_rail'])
            
    except Exception as e:
        state['errors'].append(f"EXEC_025_FAIL: {e}")
            
    return state