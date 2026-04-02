import os
import logging
import geopandas as gpd
from pathlib import Path

# --- CONFIGURATION ---
BASE_DATA_DIR = Path(r"[REDACTED_BY_SCRIPT]")
L1_LAND_COVER_INPUT = BASE_DATA_DIR / "[REDACTED_BY_SCRIPT]"
L1_BUILDINGS_INPUT = BASE_DATA_DIR / "[REDACTED_BY_SCRIPT]"
L1_LANDUSE_INPUT = BASE_DATA_DIR / "L1_osm_landuse.gpkg"
TARGET_CRS = "EPSG:27700"
RADII_METERS = [1000, 2000, 5000, 10000]

# --- MODULE-LEVEL STATE ---
gdf_lc, gdf_bld, gdf_lu, sindex_lc, sindex_bld, sindex_lu = [None] * 6

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_lc, gdf_bld, gdf_lu, sindex_lc, sindex_bld, sindex_lu
    if sindex_lc is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_lc, sindex_lc = gpd.read_file(L1_LAND_COVER_INPUT), gpd.read_file(L1_LAND_COVER_INPUT).sindex
        gdf_bld, sindex_bld = gpd.read_file(L1_BUILDINGS_INPUT), gpd.read_file(L1_BUILDINGS_INPUT).sindex
        gdf_lu, sindex_lu = gpd.read_file(L1_LANDUSE_INPUT), gpd.read_file(L1_LANDUSE_INPUT).sindex
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_lc = "INIT_FAILED"
def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_lc == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        geom = state['input_geom']
        for r_m in RADII_METERS:
            r_km = r_m // 1000
            buffer = gpd.GeoDataFrame(geometry=[geom.buffer(r_m)], crs=TARGET_CRS)
            
            # Helper for indexed overlay, ensuring robust calculation
            def get_coverage(gdf, sindex, super_class):
                candidates_idx = list(sindex.intersection(buffer.bounds))
                if not candidates_idx: return 0.0
                candidates = gdf.iloc[candidates_idx]
                
                # Filter for the specific class and ensure there's something to intersect
                filtered_candidates = candidates[candidates['super_class'] == super_class]
                if filtered_candidates.empty: return 0.0
                
                intersected = gpd.overlay(buffer, filtered_candidates, how='intersection')
                return intersected.area.sum() / buffer.area.iloc[0] if not intersected.empty else 0.0

            # Calculate land fabric features using correct data sources
            state['features'][f'[REDACTED_BY_SCRIPT]'] = get_coverage(gdf_lc, sindex_lc, 'urban_fabric')
            state['features'][f'[REDACTED_BY_SCRIPT]'] = get_coverage(gdf_lc, sindex_lc, 'agricultural')
            state['features'][f'[REDACTED_BY_SCRIPT]'] = get_coverage(gdf_lc, sindex_lc, 'natural_protected_clc')
            state['features'][f'[REDACTED_BY_SCRIPT]'] = get_coverage(gdf_lu, sindex_lu, '[REDACTED_BY_SCRIPT]')
            
            # Building density calculation
            bld_candidates = gdf_bld.iloc[list(sindex_bld.intersection(buffer.bounds))]
            res_bld = bld_candidates[bld_candidates['super_class'] == '[REDACTED_BY_SCRIPT]']
            if not res_bld.empty:
                bld_intersected = gpd.overlay(buffer, res_bld, how='intersection')
                buffer_area_sqkm = buffer.area.iloc[0] / 1e6
                state['features'][f'[REDACTED_BY_SCRIPT]'] = bld_intersected.area.sum() / buffer_area_sqkm if not bld_intersected.empty else 0.0
            else:
                state['features'][f'[REDACTED_BY_SCRIPT]'] = 0.0

    except Exception as e:
        state['errors'].append(f"EXEC_024_FAIL: {e}")
            
    return state