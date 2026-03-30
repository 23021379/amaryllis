import logging
import numpy as np
import geopandas as gpd
import pandas as pd
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
PHI_L2_ARTIFACT = os.path.join(BASE_DATA_DIR, "geopackage", "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2000, 5000, 10000]
NULL_SENTINEL_FLOAT = -1.0

# --- Module-level State for Performance ---
gdf_phi, sindex_phi = None, None
gdf_phi_priority, sindex_phi_priority = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_phi, sindex_phi, gdf_phi_priority, sindex_phi_priority
    if gdf_phi is not None:
        return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        gdf_phi_local = gpd.read_file(PHI_L2_ARTIFACT)
        if gdf_phi_local.crs.to_string() != PROJECT_CRS:
            gdf_phi_local = gdf_phi_local.to_crs(PROJECT_CRS)
        
        gdf_phi_local['area_ha'] = gdf_phi_local.geometry.area / 10_000
        gdf_phi_local.reset_index(drop=True, inplace=True)

        gdf_phi_priority_local = gdf_phi_local[gdf_phi_local['is_priority'] == 1].copy()
        gdf_phi_priority_local.reset_index(drop=True, inplace=True)
        
        gdf_phi = gdf_phi_local
        sindex_phi = gdf_phi.sindex
        gdf_phi_priority = gdf_phi_priority_local
        sindex_phi_priority = gdf_phi_priority.sindex

        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        gdf_phi = "INIT_FAILED"

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    if gdf_phi is None:
        _initialize_module_state()
    if isinstance(gdf_phi, str) and gdf_phi == "INIT_FAILED":
        logging.error("[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Proximity & Direct Impact Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    joined_gdf = gpd.sjoin_nearest(master_gdf, gdf_phi, how='left', distance_col='ph_dist_to_nearest_m')
    joined_gdf = joined_gdf[~joined_gdf.index.duplicated(keep='first')]

    master_gdf['ph_dist_to_nearest_m'] = joined_gdf['ph_dist_to_nearest_m']
    master_gdf['ph_nearest_is_priority'] = joined_gdf.get('is_priority', 0).astype(int)
    master_gdf['[REDACTED_BY_SCRIPT]'] = joined_gdf.get('hab_group', 'None')
    
    within_join = gpd.sjoin(master_gdf, gdf_phi, how='left', predicate='within')
    is_within = within_join['index_right'].notna()
    master_gdf['ph_is_within'] = is_within[~is_within.index.duplicated(keep='first')].astype(int)

    # --- Multi-Radii Density Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    def calculate_density(geom):
        results = {}
        max_radius = max(RADII_METERS)
        buffer = geom.buffer(max_radius)
        
        possible_matches_idx = list(sindex_phi.intersection(buffer.bounds))
        candidates = gdf_phi.iloc[possible_matches_idx]
        
        priority_matches_idx = list(sindex_phi_priority.intersection(buffer.bounds))
        priority_candidates = gdf_phi_priority.iloc[priority_matches_idx]
        
        for r in RADII_METERS:
            r_km = r // 1000
            radius_buffer = geom.buffer(r)
            
            intersecting_phi = candidates[candidates.intersects(radius_buffer)]
            total_area_ha = (intersecting_phi.geometry.intersection(radius_buffer).area.sum()) / 10_000
            results[f'[REDACTED_BY_SCRIPT]'] = total_area_ha
            
            intersecting_priority = priority_candidates[priority_candidates.intersects(radius_buffer)]
            priority_area_ha = (intersecting_priority.geometry.intersection(radius_buffer).area.sum()) / 10_000
            
            pct = (priority_area_ha / total_area_ha * 100) if total_area_ha > 0 else 0
            results[f'[REDACTED_BY_SCRIPT]'] = pct
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
        'ph_dist_to_nearest_m': NULL_SENTINEL_FLOAT,
        'ph_nearest_is_priority': 0,
        '[REDACTED_BY_SCRIPT]': 'None',
        'ph_is_within': 0
    }
    master_gdf.fillna(fill_values, inplace=True)
    for col in master_gdf.columns:
        if 'ph_' in col and ('pct_in' in col or 'area_in' in col):
            master_gdf[col].fillna(0, inplace=True)

    logging.info("[REDACTED_BY_SCRIPT]")
    return master_gdf