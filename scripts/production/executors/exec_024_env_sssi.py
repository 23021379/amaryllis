import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Executor Configuration ---
SSSI_PATH = Path(r"[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RISK_RADII_M = [2000, 5000, 10000, 20000]

# --- Helper Functions (Reused from other environmental executors) ---

def load_and_prepare_constraint(path: Path, area_unit_converter: float = 1.0, area_col_name: str = 'area_sqkm') -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(path)
    if gdf.crs.to_string() != PROJECT_CRS:
        logging.warning(f"Hostile CRS '{gdf.crs}'[REDACTED_BY_SCRIPT]'{PROJECT_CRS}'.")
        gdf = gdf.to_crs(PROJECT_CRS)
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf[area_col_name] = gdf.geometry.area / area_unit_converter
    return gdf

def calculate_proximity_and_containment(solar_farms: gpd.GeoDataFrame, constraint: gpd.GeoDataFrame, prefix: str, attribute_cols: dict) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]'{prefix}'...")
    
    if not solar_farms.index.is_unique:
        solar_farms = solar_farms.reset_index(drop=True)

    prox_join = gpd.sjoin_nearest(solar_farms, constraint, how='left', distance_col=f'[REDACTED_BY_SCRIPT]')
    prox_join = prox_join[~prox_join.index.duplicated(keep='first')]
    
    within_join = gpd.sjoin(solar_farms, constraint, how='left', predicate='within')
    is_within = within_join['index_right'].notna().astype(int)
    is_within = is_within[~is_within.index.duplicated(keep='first')]

    cols_to_select = [f'[REDACTED_BY_SCRIPT]']
    valid_attribute_cols = {k: v for k, v in attribute_cols.items() if k in prox_join.columns}
    cols_to_select.extend(valid_attribute_cols.keys())
    
    features = prox_join[cols_to_select].copy()
    features.rename(columns=valid_attribute_cols, inplace=True)
    
    features[f'{prefix}_is_within'] = is_within
    features.loc[features[f'{prefix}_is_within'] == 1, f'[REDACTED_BY_SCRIPT]'] = 0
    
    return features

def calculate_multi_radius_density(solar_farms: gpd.GeoDataFrame, constraint: gpd.GeoDataFrame, prefix: str, area_col: str, subtype_col: str = None) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]'{prefix}'...")
    max_radius = max(RISK_RADII_M)
    
    search_areas = solar_farms.copy()
    search_areas['geometry'] = search_areas.geometry.buffer(max_radius)
    
    candidates = gpd.sjoin(constraint, search_areas, how='inner', predicate='intersects')
    
    farm_geometries_array = solar_farms.loc[candidates['index_right']]['geometry'].values
    farm_geometries_series = gpd.GeoSeries(farm_geometries_array, index=candidates.index)
    candidates['distance'] = candidates.geometry.distance(farm_geometries_series)

    all_radii_features = []
    for radius_m in RISK_RADII_M:
        radius_km = int(radius_m / 1000)
        
        radius_subset = candidates[candidates['distance'] <= radius_m]
        radius_features = pd.DataFrame(index=solar_farms.index)

        if not radius_subset.empty:
            grouped_by_farm = radius_subset.groupby('index_right')
            radius_features[f'[REDACTED_BY_SCRIPT]'] = grouped_by_farm.size()
            radius_features[f'[REDACTED_BY_SCRIPT]'] = grouped_by_farm[area_col].sum()

            if subtype_col and subtype_col in radius_subset.columns:
                subtype_counts = radius_subset.groupby(['index_right', subtype_col]).size().unstack(fill_value=0)
                subtype_counts.columns = [f"[REDACTED_BY_SCRIPT]" for col in subtype_counts.columns]
                radius_features = radius_features.join(subtype_counts)

        all_radii_features.append(radius_features)

    final_density_features = pd.concat(all_radii_features, axis=1).fillna(0)
    for radius_m in RISK_RADII_M:
        radius_km = int(radius_m / 1000)
        if f'[REDACTED_BY_SCRIPT]' not in final_density_features:
            final_density_features[f'[REDACTED_BY_SCRIPT]'] = 0
        if f'[REDACTED_BY_SCRIPT]' not in final_density_features:
            final_density_features[f'[REDACTED_BY_SCRIPT]'] = 0
            
    return final_density_features

def calculate_density_focus_ratios(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]'Alpha'[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]'{prefix}'...")
    focus_df = pd.DataFrame(index=df.index)
    
    count_2km_col = f'[REDACTED_BY_SCRIPT]'
    count_20km_col = f'[REDACTED_BY_SCRIPT]'
    
    if count_2km_col in df.columns and count_20km_col in df.columns:
        focus_df[f'[REDACTED_BY_SCRIPT]'] = np.divide(
            df[count_2km_col], df[count_20km_col]
        ).fillna(0)
    else:
        focus_df[f'[REDACTED_BY_SCRIPT]'] = 0

    return focus_df

# --- Executor Function ---

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Orchestrator-compatible execution function for SSSI analysis.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    try:
        solar_farms_base = master_gdf.copy()

        # --- Feature Engineering ---
        sssi_gdf = load_and_prepare_constraint(SSSI_PATH, area_unit_converter=10000, area_col_name='area_ha')
        
        # Proximity and Containment
        sssi_prox_contain = calculate_proximity_and_containment(
            solar_farms_base, sssi_gdf, 'sssi', 
            attribute_cols={'area_ha': 'sssi_nearest_area_ha', 'sssi_name': 'sssi_nearest_name'}
        )
        
        # Multi-Radius Density
        # Note the area column name change to '[REDACTED_BY_SCRIPT]' is handled by the new helper
        sssi_density = calculate_multi_radius_density(solar_farms_base, sssi_gdf, 'sssi', 'area_ha')

        # --- Assembly & Synthesis ---
        logging.info("[REDACTED_BY_SCRIPT]")
        master_gdf = master_gdf.join([sssi_prox_contain, sssi_density])
        
        # Density Focus Ratio
        sssi_focus = calculate_density_focus_ratios(master_gdf, 'sssi')
        master_gdf = master_gdf.join(sssi_focus)

        # --- Final QA ---
        expected_cols = [f'[REDACTED_BY_SCRIPT]' for r in RISK_RADII_M] + \
                        [f'[REDACTED_BY_SCRIPT]' for r in RISK_RADII_M] + \
                        ['sssi_dist_to_nearest_m', 'sssi_is_within', 'sssi_nearest_area_ha', '[REDACTED_BY_SCRIPT]']
        for col in expected_cols:
            if col not in master_gdf.columns:
                master_gdf[col] = 0

        # The density function now creates '..._total_area_in_Xkm', so we rename to match the old schema if needed
        for r in RISK_RADII_M:
            radius_km = int(r/1000)
            old_col_name = f'[REDACTED_BY_SCRIPT]'
            new_col_name = f'[REDACTED_BY_SCRIPT]'
            if old_col_name in master_gdf.columns and new_col_name not in master_gdf.columns:
                    master_gdf.rename(columns={old_col_name: new_col_name}, inplace=True)

                
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    except Exception as e:
        logging.critical(f"[REDACTED_BY_SCRIPT]", exc_info=True)
        raise e