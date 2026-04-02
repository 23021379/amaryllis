import logging
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL_FLOAT = -1.0
NULL_SENTINEL_INT = -1
RISK_RADII_M = [2000, 5000, 10000, 20000]
# This path assumes an L1 artifact has been created for AONB
INPUT_GPKG = Path(r"[REDACTED_BY_SCRIPT]")

def _load_and_prepare_data() -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    if not INPUT_GPKG.exists():
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return gpd.GeoDataFrame(geometry=[], crs=PROJECT_CRS)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(INPUT_GPKG)
    if gdf.crs != PROJECT_CRS:
        gdf = gdf.to_crs(PROJECT_CRS)
    
    # Pre-calculate area in square kilometers
    if 'aonb_area_sqkm' not in gdf.columns:
        gdf['aonb_area_sqkm'] = gdf.geometry.area / 1_000_000
        
    return gdf

def calculate_proximity_and_containment(sites_gdf: gpd.GeoDataFrame, constraint_gdf: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    if constraint_gdf.empty:
        return pd.DataFrame({
            f'[REDACTED_BY_SCRIPT]': NULL_SENTINEL_FLOAT,
            f'{prefix}_is_within': 0,
            f'[REDACTED_BY_SCRIPT]': NULL_SENTINEL_FLOAT
        }, index=sites_gdf.index)

    # Proximity and nearest feature attributes
    prox_join = gpd.sjoin_nearest(sites_gdf, constraint_gdf, how='left', distance_col=f'[REDACTED_BY_SCRIPT]')
    prox_join = prox_join.rename(columns={'aonb_area_sqkm': f'[REDACTED_BY_SCRIPT]'})
    prox_join = prox_join[~prox_join.index.duplicated(keep='first')]
    
    # Containment
    within_join = gpd.sjoin(sites_gdf, constraint_gdf, how='left', predicate='within')
    is_within_series = within_join['index_right'].notna()
    is_within = is_within_series.groupby(is_within_series.index).any().astype(int)

    features = prox_join[[f'[REDACTED_BY_SCRIPT]', f'[REDACTED_BY_SCRIPT]']].copy()
    features[f'{prefix}_is_within'] = is_within
    features[f'{prefix}_is_within'].fillna(0, inplace=True)
    features.loc[features[f'{prefix}_is_within'] == 1, f'[REDACTED_BY_SCRIPT]'] = 0
    
    return features

def calculate_multi_radius_density(sites_gdf: gpd.GeoDataFrame, constraint_gdf: gpd.GeoDataFrame, prefix: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    final_features = pd.DataFrame(index=sites_gdf.index)
    
    if constraint_gdf.empty:
        for r_m in RISK_RADII_M:
            r_km = int(r_m / 1000)
            final_features[f'[REDACTED_BY_SCRIPT]'] = 0
            final_features[f'[REDACTED_BY_SCRIPT]'] = 0.0
        return final_features

    for r_m in RISK_RADII_M:
        r_km = int(r_m / 1000)
        buffers = sites_gdf.copy()
        buffers['geometry'] = sites_gdf.geometry.buffer(r_m)
        
        # Intersect buffers with constraints
        intersected = gpd.overlay(buffers, constraint_gdf, how='intersection', keep_geom_type=False)
        
        if intersected.empty:
            final_features[f'[REDACTED_BY_SCRIPT]'] = 0
            final_features[f'[REDACTED_BY_SCRIPT]'] = 0.0
            continue

        # Calculate total area of intersection per site
        intersected['intersected_area'] = intersected.geometry.area / 1_000_000 # to sqkm
        area_sum = intersected.groupby(intersected.index).agg(
            total_area=('intersected_area', 'sum')
        ).rename(columns={'total_area': f'[REDACTED_BY_SCRIPT]'})

        # Calculate counts
        counts = intersected.groupby(intersected.index).agg(
            total_count=('hex_id', 'size')
        ).rename(columns={'total_count': f'[REDACTED_BY_SCRIPT]'})
        
        radius_features = pd.concat([area_sum, counts], axis=1)
        final_features = final_features.join(radius_features, how='left')

    return final_features.fillna(0)

def calculate_density_ratios(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    # Ratio of 5km area to 20km area
    num = df[f'[REDACTED_BY_SCRIPT]']
    den = df[f'[REDACTED_BY_SCRIPT]']
    df[f'[REDACTED_BY_SCRIPT]'] = (num / den).fillna(0)
    return df

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")

    aonb_gdf = _load_and_prepare_data()

    # --- Feature Generation ---
    prox_features = calculate_proximity_and_containment(master_gdf, aonb_gdf, 'aonb')
    density_features = calculate_multi_radius_density(master_gdf, aonb_gdf, 'aonb')

    # --- Merge Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    final_gdf = master_gdf.join([prox_features, density_features])
    
    # --- Post-calculation of Ratios ---
    final_gdf = calculate_density_ratios(final_gdf, 'aonb')
    
    # Fill any NaNs that may have slipped through
    new_cols = list(prox_features.columns) + list(density_features.columns) + [f'[REDACTED_BY_SCRIPT]']
    for col in new_cols:
        if col in final_gdf.columns and pd.api.types.is_numeric_dtype(final_gdf[col]):
            final_gdf[col] = final_gdf[col].fillna(NULL_SENTINEL_FLOAT if '.' in str(final_gdf[col].dtype) else NULL_SENTINEL_INT)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf