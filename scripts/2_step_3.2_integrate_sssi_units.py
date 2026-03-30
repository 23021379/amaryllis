"""
Directive 034: SSSI Unit Condition Enrichment

This script enriches the L32 environmental artifact with granular data on SSSI
Unit condition, evolving it to the L33 artifact.

It engineers features modeling:
1.  The condition and assessment recency of the nearest SSSI Unit.
2.  The "worst-case" ecological condition within the 2km acute zone.
3.  Data consistency between the SSSI parent and unit datasets.
"""

import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"

logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

CONDITION_ORDINAL_MAP = {
    'Favourable': 1,
    '[REDACTED_BY_SCRIPT]': 2,
    '[REDACTED_BY_SCRIPT]': 3,
    '[REDACTED_BY_SCRIPT]': 4,
    'Partially Destroyed': 5,
    'Destroyed': 5
}

def load_and_restore_l32_state(l32_path: Path, l1_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(l32_path, index_col='solar_farm_id')
    
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['easting_x'], df['northing_x']), crs=PROJECT_CRS
    )
    
    logging.info("[REDACTED_BY_SCRIPT]")
    l1_df = pd.read_csv(l1_path, usecols=['solar_farm_id', 'submission_year'], index_col='solar_farm_id')
    gdf = gdf.join(l1_df)
    gdf['submission_year'] = pd.to_datetime(gdf['submission_year'], errors='coerce')
    
    return gdf

def calculate_nearest_unit_features(solar_farms: gpd.GeoDataFrame, sssi_units: gpd.GeoDataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    joined = gpd.sjoin_nearest(solar_farms, sssi_units, how='left', distance_col='sssi_unit_dist_to_nearest_m')
    joined = joined[~joined.index.duplicated(keep='first')]
    
    features = joined[['sssi_unit_dist_to_nearest_m', 'condition', 'cond_date', 'sssi_name']].copy()
    features.rename(columns={'condition': 'sssi_unit_nearest_condition'}, inplace=True)
    
    # Calculate recency
    features['cond_date'] = pd.to_datetime(features['cond_date'], errors='coerce')
    recency = (solar_farms['submission_year'] - features['cond_date']).dt.days
    features['[REDACTED_BY_SCRIPT]'] = recency
    
    return features

def calculate_worst_case_feature(solar_farms: gpd.GeoDataFrame, sssi_units: gpd.GeoDataFrame) -> pd.Series:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # ARCHITECTURAL NOTE (Category 3 Pattern): The following `sjoin_nearest` is a
    # deliberate one-to-many join. It finds all SSSI units within the 2km acute
    # zone for each solar farm, creating duplicate solar farm indices in the result.
    # This is essential for the subsequent `groupby().max()` operation, which
    # correctly identifies the single worst ecological condition from all candidates.
    # DO NOT de-duplicate this join.
    joined = gpd.sjoin_nearest(solar_farms, sssi_units, how='left', max_distance=2000)
    
    # Group by solar farm and find the max ordinal value
    worst_condition = joined.groupby(joined.index)['condition_ordinal'].max()
    worst_condition = worst_condition.reindex(solar_farms.index).fillna(0).astype(int)
    worst_condition.name = 'sssi_unit_worst_condition_in_2km'
    
    return worst_condition

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Load Inputs ---
    l32_path = r"[REDACTED_BY_SCRIPT]"
    l1_path = r"[REDACTED_BY_SCRIPT]"
    sssi_units_l1_path = r"[REDACTED_BY_SCRIPT]"
    output_path = r"[REDACTED_BY_SCRIPT]"

    solar_farms_gdf = load_and_restore_l32_state(l32_path, l1_path)
    sssi_units_gdf = gpd.read_file(sssi_units_l1_path)

    # --- Phase 2: Feature Engineering ---
    nearest_unit_features = calculate_nearest_unit_features(solar_farms_gdf, sssi_units_gdf)
    worst_case_feature = calculate_worst_case_feature(solar_farms_gdf, sssi_units_gdf)
    
    # --- Phase 3: Assembly & Reconciliation ---
    logging.info("[REDACTED_BY_SCRIPT]")
    final_gdf = solar_farms_gdf.join([nearest_unit_features, worst_case_feature])

    logging.info("[REDACTED_BY_SCRIPT]")
    # Standardize strings for robust comparison
    parent_name = final_gdf['sssi_nearest_name'].str.strip().str.lower()
    child_name = final_gdf['sssi_name'].str.strip().str.lower()
    final_gdf['[REDACTED_BY_SCRIPT]'] = (parent_name != child_name).astype(int)
    
    # --- Phase 4: QA and Persistence ---
    final_df = final_gdf.drop(columns=['geometry', 'submission_year', 'cond_date', 'sssi_name'])
    
    # Handle nulls as per directive
    final_df['[REDACTED_BY_SCRIPT]'].fillna(-1, inplace=True)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    # Ordinal value of nearest condition
    ordinal_map = {v: k for k, v in CONDITION_ORDINAL_MAP.items()} # Invert map for lookup
    nearest_ordinal = final_df['sssi_unit_nearest_condition'].map(CONDITION_ORDINAL_MAP).fillna(0)
    if not (final_df['sssi_unit_worst_condition_in_2km'] >= nearest_ordinal).all():
        logging.error("[REDACTED_BY_SCRIPT]")
    else:
        logging.info("[REDACTED_BY_SCRIPT]")

    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()