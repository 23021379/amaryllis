import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')


# Inputs for this directive
OHL_132KV_LINES_INPUT = r"[REDACTED_BY_SCRIPT]"
OHL_132KV_TOWERS_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
WAYLEAVE_BUFFER_METERS_132KV = 30 # 60m total corridor for 132kV
LOCAL_NETWORK_BUFFER_METERS = 2000

from typing import Tuple
def load_and_prepare_132kv_data(lines_path: str, towers_path: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads, re-projects, and synthesizes 132kV wayleave corridors and tower data.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_lines = gpd.read_file(lines_path)
    gdf_towers = gpd.read_file(towers_path)

    # MANDATE: Unconditional CRS Unification for both datasets.
    gdf_lines = gdf_lines.to_crs("EPSG:27700")
    gdf_towers = gdf_towers.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Schema Normalization.
    gdf_lines.columns = [col.lower().strip() for col in gdf_lines.columns]
    gdf_towers.columns = [col.lower().strip().replace('ob_', '') for col in gdf_towers.columns]

    # MANDATE: Create larger 132kV Wayleave Corridors.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_corridors = gdf_lines.copy()
    gdf_corridors['geometry'] = gdf_lines.geometry.buffer(WAYLEAVE_BUFFER_METERS_132KV)
    
    return gdf_towers, gdf_corridors


def calculate_132kv_core_features(gdf_solar: gpd.GeoDataFrame, gdf_towers: gpd.GeoDataFrame, gdf_corridors: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculates the core parallel feature set for the 132kV network.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Proximity Features
    gdf_solar_with_dist = gpd.sjoin_nearest(gdf_solar, gdf_corridors, distance_col="[REDACTED_BY_SCRIPT]", how="left").drop_duplicates(subset='solar_farm_id')
    gdf_solar_with_towers = gpd.sjoin_nearest(gdf_solar, gdf_towers, distance_col="[REDACTED_BY_SCRIPT]", how="left").drop_duplicates(subset='solar_farm_id')

    # Land Sterilization Features
    intersecting = gpd.overlay(gdf_solar, gdf_corridors, how='intersection')
    intersecting['[REDACTED_BY_SCRIPT]'] = intersecting.geometry.area
    sterilized_summary = intersecting.groupby('solar_farm_id')['[REDACTED_BY_SCRIPT]'].sum().reset_index()

    # Local Network Topology
    buffer_area_km2 = np.pi * ((LOCAL_NETWORK_BUFFER_METERS / 1000) ** 2)
    gdf_solar_buffers = gdf_solar.copy()
    gdf_solar_buffers['geometry'] = gdf_solar_buffers.geometry.buffer(LOCAL_NETWORK_BUFFER_METERS)
    towers_in_buffer = gpd.sjoin(gdf_towers, gdf_solar_buffers, how="inner", predicate="within")
    tower_counts = towers_in_buffer.groupby('solar_farm_id').size().reset_index(name='[REDACTED_BY_SCRIPT]')
    tower_counts['[REDACTED_BY_SCRIPT]'] = tower_counts['[REDACTED_BY_SCRIPT]'] / buffer_area_km2

    # Combine all features
    df_features = gdf_solar[['solar_farm_id', 'solar_site_area_sqm']].copy()
    df_features = df_features.merge(gdf_solar_with_dist[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(gdf_solar_with_towers[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(sterilized_summary, on='solar_farm_id', how='left')
    df_features = df_features.merge(tower_counts, on='solar_farm_id', how='left')
    
    df_features['[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)
    df_features['[REDACTED_BY_SCRIPT]'] = (df_features['[REDACTED_BY_SCRIPT]'] > 0).astype(int)
    df_features.loc[df_features['[REDACTED_BY_SCRIPT]'] == 1, '[REDACTED_BY_SCRIPT]'] = 0
    df_features['[REDACTED_BY_SCRIPT]'] = df_features['[REDACTED_BY_SCRIPT]'] / df_features['solar_site_area_sqm']
    
    return df_features.drop(columns=['solar_site_area_sqm'])


def main():
    """
    Main function to execute the consolidated 132kV network integration and synthesis pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest and Prepare 132kV Data ---
    gdf_towers, gdf_corridors = load_and_prepare_132kv_data(
        OHL_132KV_LINES_INPUT,
        OHL_132KV_TOWERS_INPUT
    )
    
    # --- Load Solar Data ---
    df_solar_l20 = pd.read_csv(SOLAR_DATA_INPUT)
    gdf_solar = gpd.GeoDataFrame(
        df_solar_l20, geometry=gpd.points_from_xy(df_solar_l20.easting_x, df_solar_l20.northing_x), crs="EPSG:27700"
    )
    gdf_solar['solar_farm_id'] = gdf_solar.index

    # --- Phase 2: Core 132kV Feature Generation ---
    core_132kv_features = calculate_132kv_core_features(gdf_solar, gdf_towers, gdf_corridors)
    
    # --- Integration ---
    df_enriched = df_solar_l20.merge(core_132kv_features, left_index=True, right_on='solar_farm_id', how='left')

    # --- Phase 3: Cross-Voltage Synthetic Features (The Alpha) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] / (df_enriched['[REDACTED_BY_SCRIPT]'] + 1)
    
    df_enriched['ohl_nearest_voltage'] = np.where(
        df_enriched['[REDACTED_BY_SCRIPT]'] < df_enriched['[REDACTED_BY_SCRIPT]'],
        132, 33
    )
    
    df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['ohl_local_tower_count'] + df_enriched['[REDACTED_BY_SCRIPT]']

    # --- Finalization ---
    new_cols = [col for col in df_enriched.columns if 'ohl_132kv' in col or 'ohl_total' in col or 'ohl_nearest' in col or 'dist_ratio' in col]
    df_enriched[new_cols] = df_enriched[new_cols].fillna(0)
    df_enriched.drop(columns=['solar_farm_id'], inplace=True, errors='ignore')
    
    # --- Persist Final Artifact ---
    output_path = OUTPUT_ARTIFACT
    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()