import logging
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL_FLOAT = -1.0
RISK_RADII_M = [2000, 5000, 10000, 20000]
INPUT_GPKG = Path(r"[REDACTED_BY_SCRIPT]") # Assumed L1 artifact path

# --- Helper Functions ---
def _load_and_prepare_spa_data() -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    if not INPUT_GPKG.exists():
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return gpd.GeoDataFrame(geometry=[], crs=PROJECT_CRS)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(INPUT_GPKG)
    if gdf.crs != PROJECT_CRS:
        gdf = gdf.to_crs(PROJECT_CRS)
    
    if 'spa_area_ha' not in gdf.columns:
        gdf['spa_area_ha'] = gdf.geometry.area / 10_000
        
    return gdf

def calculate_proximity_and_containment(solar_farms: gpd.GeoDataFrame, constraint: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    if constraint.empty:
        return pd.DataFrame({
            f'[REDACTED_BY_SCRIPT]': NULL_SENTINEL_FLOAT,
            f'{prefix}_is_within': 0
        }, index=solar_farms.index)

    prox_join = gpd.sjoin_nearest(solar_farms, constraint, how='left', distance_col=f'[REDACTED_BY_SCRIPT]')
    prox_join = prox_join[~prox_join.index.duplicated(keep='first')]
    
    within_join = gpd.sjoin(solar_farms, constraint, how='left', predicate='within')
    
    # --- FIX START ---
    # Check for a successful join using the 'index_right' column.
    # Group by the original index and check if any join was successful.
    is_within_series = within_join['index_right'].notna()
    is_within = is_within_series.groupby(is_within_series.index).any().astype(int)
    # --- FIX END ---

    features = prox_join[[f'[REDACTED_BY_SCRIPT]']].copy()
    features[f'{prefix}_is_within'] = is_within
    features[f'{prefix}_is_within'].fillna(0, inplace=True)
    features.loc[features[f'{prefix}_is_within'] == 1, f'[REDACTED_BY_SCRIPT]'] = 0
    
    return features

def calculate_multi_radius_density(solar_farms: gpd.GeoDataFrame, constraint: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    all_radii_features = []
    max_radius = max(RISK_RADII_M)
    
    if constraint.empty:
        # Return a dataframe of zeros if no constraint data
        df = pd.DataFrame(index=solar_farms.index)
        for r_m in RISK_RADII_M:
            r_km = int(r_m / 1000)
            df[f'[REDACTED_BY_SCRIPT]'] = 0
            df[f'[REDACTED_BY_SCRIPT]'] = 0.0
        return df

    candidates = gpd.sjoin_nearest(solar_farms, constraint, how='left', max_distance=max_radius)
    candidates_with_dist = candidates.join(solar_farms[['geometry']], rsuffix='_farm')
    candidates_with_dist = candidates_with_dist[candidates_with_dist.index_right.notna()].copy()
    candidates_with_dist['distance'] = candidates_with_dist.geometry.distance(candidates_with_dist.geometry_farm)

    for radius_m in RISK_RADII_M:
        radius_km = int(radius_m / 1000)
        radius_subset = candidates_with_dist[candidates_with_dist['distance'] <= radius_m]
        radius_features = pd.DataFrame(index=solar_farms.index)
        
        if not radius_subset.empty:
            grouped_by_farm = radius_subset.groupby(radius_subset.index)
            radius_features[f'[REDACTED_BY_SCRIPT]'] = grouped_by_farm.size()
            radius_features[f'[REDACTED_BY_SCRIPT]'] = grouped_by_farm['spa_area_ha'].sum()
            
        all_radii_features.append(radius_features)

    final_density_features = pd.concat(all_radii_features, axis=1).fillna(0)
    return final_density_features

# --- Executor Function ---
def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Integrates Special Protection Areas (SPA) constraint features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    spa_gdf = _load_and_prepare_spa_data()

    # --- Proximity and Containment ---
    prox_features = calculate_proximity_and_containment(master_gdf, spa_gdf, 'spa')
    
    # --- Density ---
    density_features = calculate_multi_radius_density(master_gdf, spa_gdf, 'spa')

    # --- Merge Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    final_gdf = master_gdf.join([prox_features, density_features])
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf