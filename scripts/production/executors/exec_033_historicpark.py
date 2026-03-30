import logging
import numpy as np
import geopandas as gpd
import pandas as pd
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
HP_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "geopackage", "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2000, 5000, 10000, 20000]
NULL_SENTINEL_FLOAT = -1.0

# --- Module-level State for Performance ---
gdf_hp, sindex_hp = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_hp, sindex_hp
    if gdf_hp is not None:
        return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        gdf_hp_local = gpd.read_file(HP_L1_ARTIFACT)
        if gdf_hp_local.crs.to_string() != PROJECT_CRS:
             gdf_hp_local = gdf_hp_local.to_crs(PROJECT_CRS)

        gdf_hp_local['hp_area_ha'] = gdf_hp_local.geometry.area / 10_000
        
        # Reset index to prevent potential iloc errors
        gdf_hp_local.reset_index(drop=True, inplace=True)
        
        gdf_hp = gdf_hp_local
        sindex_hp = gdf_hp.sindex

        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        gdf_hp = "INIT_FAILED" # Use string to indicate failure

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    if gdf_hp is None:
        _initialize_module_state()
    if isinstance(gdf_hp, str) and gdf_hp == "INIT_FAILED":
        logging.error("[REDACTED_BY_SCRIPT]")
        # Add empty columns to ensure schema consistency
        # (This part can be enhanced to be more robust)
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Proximity & Direct Impact Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    joined_gdf = gpd.sjoin_nearest(master_gdf, gdf_hp, how='left', distance_col='hp_dist_to_nearest_m')
    joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep='first')]

    master_gdf['hp_dist_to_nearest_m'] = joined_gdf['hp_dist_to_nearest_m']
    master_gdf['hp_nearest_name'] = joined_gdf.get('hp_name', 'UNKNOWN')
    master_gdf['hp_nearest_area_ha'] = joined_gdf.get('hp_area_ha', 0.0)
    
    # Direct within check
    within_join = gpd.sjoin(master_gdf, gdf_hp, how='left', predicate='within')
    is_within = within_join['index_right'].notna()
    master_gdf['hp_is_within'] = is_within[~is_within.index.duplicated(keep='first')].astype(int)

    # --- Multi-Radii Density Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    def calculate_density(geom):
        results = {}
        max_radius = max(RADII_METERS)
        buffer = geom.buffer(max_radius)
        possible_matches_idx = list(sindex_hp.intersection(buffer.bounds))
        
        if not possible_matches_idx:
            for r in RADII_METERS:
                r_km = r // 1000
                results[f'[REDACTED_BY_SCRIPT]'] = 0.0
                results[f'[REDACTED_BY_SCRIPT]'] = 0
            return pd.Series(results)

        candidates = gdf_hp.iloc[possible_matches_idx]
        
        for r in RADII_METERS:
            r_km = r // 1000
            radius_buffer = geom.buffer(r)
            intersecting_hp = candidates[candidates.intersects(radius_buffer)]
            
            total_area_in_radius = 0
            if not intersecting_hp.empty:
                intersections = intersecting_hp.geometry.intersection(radius_buffer)
                total_area_in_radius = intersections.area.sum()
            
            results[f'[REDACTED_BY_SCRIPT]'] = total_area_in_radius / 10_000
            results[f'[REDACTED_BY_SCRIPT]'] = len(intersecting_hp)
        return pd.Series(results)

    density_features = master_gdf['geometry'].apply(calculate_density)
    master_gdf = pd.concat([master_gdf, density_features], axis=1)

    # --- Capstone Synthesis Update ---
    logging.info("[REDACTED_BY_SCRIPT]")
    dist_cols = [col for col in master_gdf.columns if '_dist_to_nearest_m' in col]
    dist_df = master_gdf[dist_cols].copy().fillna(np.inf)

    master_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.min(axis=1).replace(np.inf, NULL_SENTINEL_FLOAT)
    
    constraint_map = {col: col.split('_')[0].upper() for col in dist_cols}
    master_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.idxmin(axis=1).map(constraint_map).fillna('None')

    # --- Finalization and Cleanup ---
    logging.info("[REDACTED_BY_SCRIPT]")
    fill_values = {
        'hp_dist_to_nearest_m': NULL_SENTINEL_FLOAT,
        'hp_nearest_name': 'None',
        'hp_nearest_area_ha': 0.0,
        'hp_is_within': 0
    }
    master_gdf.fillna(fill_values, inplace=True)
    for col in master_gdf.columns:
        if 'hp_' in col and ('count' in col or 'area_in' in col):
            master_gdf[col].fillna(0, inplace=True)

    logging.info("[REDACTED_BY_SCRIPT]")
    return master_gdf