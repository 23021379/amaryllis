"""
Directive 033: Environmental Constraint Integration (Phase 3: SSSI)

This script evolves the L31 environmental artifact by integrating features
derived from Sites of Special Scientific Interest (SSSI).

It applies the "[REDACTED_BY_SCRIPT]" to the SSSI data and updates the
capstone environmental synthesis features.

The final output is the `Amaryllis_L32_ENV.csv` artifact.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
RISK_RADII_M = [2000, 5000, 10000, 20000]

logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Reusable Functions from Previous Directives (Assumed to be in a shared module) ---
def load_and_restore_geo_state(csv_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(csv_path, index_col='solar_farm_id')
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['easting_x'], df['northing_x']), crs=PROJECT_CRS
    )
    return gdf

def calculate_proximity_and_containment(solar_farms: gpd.GeoDataFrame, constraint: gpd.GeoDataFrame, prefix: str, attribute_cols: dict) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    # ... (Function as defined and corrected in previous step) ...
    logging.info(f"[REDACTED_BY_SCRIPT]'{prefix}'...")
    prox_join = gpd.sjoin_nearest(solar_farms, constraint, how='left', distance_col=f'[REDACTED_BY_SCRIPT]')
    prox_join = prox_join[~prox_join.index.duplicated(keep='first')]
    within_join = gpd.sjoin(solar_farms, constraint, how='left', predicate='within')
    is_within = within_join['index_right'].notna().astype(int)
    is_within = is_within[~is_within.index.duplicated(keep='first')]
    cols_to_select = [f'[REDACTED_BY_SCRIPT]'] + list(attribute_cols.keys())
    features = prox_join[cols_to_select].copy()
    features.rename(columns=attribute_cols, inplace=True)
    features[f'{prefix}_is_within'] = is_within
    features.loc[features[f'{prefix}_is_within'] == 1, f'[REDACTED_BY_SCRIPT]'] = 0
    return features

def calculate_multi_radius_density(solar_farms: gpd.GeoDataFrame, constraint: gpd.GeoDataFrame, prefix: str, area_col: str) -> pd.DataFrame:
    """Implements the 'Cascading Filter'[REDACTED_BY_SCRIPT]"""
    # ... (Function as defined and corrected in previous step, simplified as SSSI has no subtypes) ...
    logging.info(f"[REDACTED_BY_SCRIPT]'{prefix}'...")
    max_radius = max(RISK_RADII_M)
    candidates = gpd.sjoin_nearest(solar_farms, constraint, how='left', max_distance=max_radius)
    candidates_with_dist = candidates.join(solar_farms[['geometry']], rsuffix='_farm')
    candidates_with_dist = candidates_with_dist[candidates_with_dist.index_right.notna()].copy()
    candidates_with_dist['distance'] = candidates_with_dist.geometry.distance(candidates_with_dist.geometry_farm)
    all_radii_features = []
    for radius_m in RISK_RADII_M:
        radius_km = int(radius_m / 1000)
        radius_subset = candidates_with_dist[candidates_with_dist['distance'] <= radius_m]
        radius_features = pd.DataFrame(index=solar_farms.index)
        if not radius_subset.empty:
            grouped_by_farm = radius_subset.groupby(radius_subset.index)
            radius_features[f'[REDACTED_BY_SCRIPT]'] = grouped_by_farm.size()
            radius_features[f'[REDACTED_BY_SCRIPT]'] = grouped_by_farm[area_col].sum()
        all_radii_features.append(radius_features)
    final_density_features = pd.concat(all_radii_features, axis=1).fillna(0)
    # Ensure all columns exist
    for r_m in RISK_RADII_M:
        r_km = int(r_m / 1000)
        for col_suffix in [f'count_in_{r_km}km', f'[REDACTED_BY_SCRIPT]']:
            col_name = f'[REDACTED_BY_SCRIPT]'
            if col_name not in final_density_features:
                final_density_features[col_name] = 0
    return final_density_features

def calculate_updated_synthesis(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    synth_df = pd.DataFrame(index=df.index)
    
    # Update min distance calculation
    dist_cols = ['aw_dist_to_nearest_m', 'aonb_dist_to_nearest_m', 'sssi_dist_to_nearest_m']
    synth_df['[REDACTED_BY_SCRIPT]'] = df[dist_cols].min(axis=1)
    
    # Update dominant constraint logic
    idxmin_series = df[dist_cols].idxmin(axis=1)
    synth_df['[REDACTED_BY_SCRIPT]'] = idxmin_series.apply(
        lambda x: 'AncientWoodland' if 'aw_' in x else ('AONB' if 'aonb_' in x else 'SSSI')
    )
    return synth_df

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Load Inputs ---
    l31_path = r"[REDACTED_BY_SCRIPT]"
    sssi_l1_path = r"[REDACTED_BY_SCRIPT]"
    output_path = r"[REDACTED_BY_SCRIPT]"

    solar_farms_gdf = load_and_restore_geo_state(l31_path)
    sssi_gdf = gpd.read_file(sssi_l1_path)
    sssi_gdf['area_ha'] = sssi_gdf.geometry.area / 10000 # Calculate area in hectares

    # --- Phase 2: SSSI Feature Engineering ---
    sssi_prox_contain = calculate_proximity_and_containment(
        solar_farms_gdf, sssi_gdf, 'sssi',
        attribute_cols={'name': 'sssi_nearest_name', 'area_ha': 'sssi_nearest_area_ha'}
    )
    sssi_density = calculate_multi_radius_density(solar_farms_gdf, sssi_gdf, 'sssi', 'area_ha')

    # --- Phase 3: Assembly & Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Join the new SSSI-specific features
    df_with_sssi = solar_farms_gdf.join([sssi_prox_contain, sssi_density])
    
    # Calculate SSSI alpha feature
    df_with_sssi['[REDACTED_BY_SCRIPT]'] = np.divide(
        df_with_sssi['sssi_count_in_2km'], (df_with_sssi['sssi_count_in_20km'] + 1)
    ).fillna(0)
    
    # Update the capstone synthesis features
    updated_synthesis_features = calculate_updated_synthesis(df_with_sssi)
    
    # MANDATED FIX: Drop the obsolete synthesis columns before joining the new ones.
    obsolete_cols = updated_synthesis_features.columns.tolist()
    final_df = df_with_sssi.drop(columns=obsolete_cols, errors='ignore').join(updated_synthesis_features)

    # --- Phase 4: QA and Persistence ---
    logging.info("[REDACTED_BY_SCRIPT]")
    if not (final_df['sssi_count_in_2km'] <= final_df['sssi_count_in_5km']).all():
        logging.error("[REDACTED_BY_SCRIPT]")
    else:
        logging.info("[REDACTED_BY_SCRIPT]")

    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.drop(columns=['geometry']).to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()