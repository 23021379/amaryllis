import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')


# Input for this directive
OHL_33KV_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
WAYLEAVE_BUFFER_METERS = 20 # 20m buffer on each side creates a 40m corridor


from typing import Tuple

def load_and_prepare_ohl_data(filepath: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads, re-projects, and synthesizes OHL wayleave corridors.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_ohl_lines = gpd.read_file(filepath)

    # MANDATE: Unconditional CRS Unification.
    gdf_ohl_lines = gdf_ohl_lines.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Schema Normalization.
    gdf_ohl_lines.columns = [col.lower().strip() for col in gdf_ohl_lines.columns]

    # MANDATE: Create Wayleave Corridors.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_ohl_corridors = gdf_ohl_lines.copy()
    gdf_ohl_corridors['geometry'] = gdf_ohl_lines.geometry.buffer(WAYLEAVE_BUFFER_METERS)
    
    # MANDATE: Reset index to guarantee alignment between sindex results and iloc positions.
    # This prevents IndexError when the source data has a non-standard or non-monotonic index.
    gdf_ohl_corridors.reset_index(drop=True, inplace=True)
    
    return gdf_ohl_lines, gdf_ohl_corridors

def calculate_sterilization_features(gdf_solar: gpd.GeoDataFrame, gdf_ohl_corridors: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculates proximity and land sterilization features using indexed spatial queries.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Proximity Feature (Robust Implementation)
    # The sjoin_nearest pattern is brittle if a site is equidistant to multiple corridors.
    # This implementation is deterministic: it finds the index of the nearest geometry and
    # calculates the distance explicitly, avoiding any ambiguous attribute joins.
    # sindex.nearest returns a tuple of (input_geometry_indices, tree_geometry_indices)
    input_indices, tree_indices = gdf_ohl_corridors.sindex.nearest(gdf_solar.geometry, return_distance=False, return_all=False)
    
    # Extract the actual nearest geometry from the right dataframe using the tree_indices.
    # .iloc is now safe to use because the index of gdf_ohl_corridors was reset upon creation.
    nearest_ohl_geoms = gdf_ohl_corridors.geometry.iloc[tree_indices]
    
    # Calculate the distance between each solar farm and its single, deterministically-found nearest corridor.
    # We must reset the index of nearest_ohl_geoms to align with gdf_solar's index for the vectorized operation.
    distances = gdf_solar.geometry.distance(nearest_ohl_geoms.reset_index(drop=True), align=True)
    
    # Create the feature DataFrame directly from the calculated distances.
    gdf_solar_with_dist = gdf_solar[['solar_farm_id']].copy()
    gdf_solar_with_dist['[REDACTED_BY_SCRIPT]'] = distances

    # Intersection and Sterilized Area Features
    intersecting_sites = gpd.overlay(gdf_solar, gdf_ohl_corridors, how='intersection')
    intersecting_sites['[REDACTED_BY_SCRIPT]'] = intersecting_sites.geometry.area
    
    # Sum sterilized area for sites intersected by multiple corridors
    sterilized_summary = intersecting_sites.groupby('solar_farm_id')['[REDACTED_BY_SCRIPT]'].sum().reset_index()
    
    # Merge features back to the main solar dataframe
    df_features = gdf_solar[['solar_farm_id', 'solar_site_area_sqm']].copy()
    df_features = df_features.merge(gdf_solar_with_dist[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(sterilized_summary, on='solar_farm_id', how='left')
    
    df_features['[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)
    df_features['[REDACTED_BY_SCRIPT]'] = (df_features['[REDACTED_BY_SCRIPT]'] > 0).astype(int)
    
    # A site that is intersected has a distance of 0 to the corridor.
    df_features.loc[df_features['[REDACTED_BY_SCRIPT]'] == 1, '[REDACTED_BY_SCRIPT]'] = 0
    
    df_features['[REDACTED_BY_SCRIPT]'] = df_features['[REDACTED_BY_SCRIPT]'] / df_features['solar_site_area_sqm']
    
    return df_features.drop(columns=['solar_site_area_sqm'])


def calculate_topology_features(gdf_solar: gpd.GeoDataFrame, gdf_ohl_lines: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculates OHL density and connection strategy ratio features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Density Feature
    buffer_radius_km = 2
    buffer_area_km2 = np.pi * (buffer_radius_km ** 2)
    
    gdf_solar_buffers = gdf_solar.copy()
    gdf_solar_buffers['geometry'] = gdf_solar_buffers.geometry.buffer(buffer_radius_km * 1000)
    
    clipped_lines = gpd.overlay(gdf_ohl_lines, gdf_solar_buffers, how='intersection')
    clipped_lines['length_km'] = clipped_lines.geometry.length / 1000
    
    density_summary = clipped_lines.groupby('solar_farm_id')['length_km'].sum().reset_index()
    density_summary.rename(columns={'length_km': '[REDACTED_BY_SCRIPT]'}, inplace=True)
    
    density_summary['[REDACTED_BY_SCRIPT]'] = density_summary['[REDACTED_BY_SCRIPT]'] / buffer_area_km2
    
    # Merge features back
    df_features = gdf_solar[['solar_farm_id', '[REDACTED_BY_SCRIPT]']].copy()
    df_features = df_features.merge(density_summary[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    
    return df_features.drop(columns=['[REDACTED_BY_SCRIPT]'])


def main():
    """
    Main function to execute the OHL corridor feature enrichment pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest and Synthesize ---
    gdf_ohl_lines, gdf_ohl_corridors = load_and_prepare_ohl_data(OHL_33KV_INPUT)
    
    # --- Load Solar Data ---
    df_solar_l18 = pd.read_csv(SOLAR_DATA_INPUT)
    # Create a GeoDataFrame from the solar site polygons (assuming WKT format)
    # This step assumes a 'geometry' column with WKT strings exists from a previous step.
    # For this exercise, we'll create points as a proxy if no polygon geometry is present.
    if 'geometry' not in df_solar_l18.columns:
         gdf_solar = gpd.GeoDataFrame(
            df_solar_l18, geometry=gpd.points_from_xy(df_solar_l18.easting_x, df_solar_l18.northing_x), crs="EPSG:27700"
        )
    else:
        # This is the expected path for real data with polygon geometries
        gdf_solar = gpd.GeoDataFrame(
            df_solar_l18, geometry=gpd.GeoSeries.from_wkt(df_solar_l18['geometry']), crs="EPSG:27700"
        )
    gdf_solar['solar_farm_id'] = gdf_solar.index

    # --- Execute Feature Generation Phases ---
    sterilization_features = calculate_sterilization_features(gdf_solar, gdf_ohl_corridors)
    topology_features = calculate_topology_features(gdf_solar, gdf_ohl_lines)

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_enriched = df_solar_l18.merge(sterilization_features, left_index=True, right_on='solar_farm_id', how='left')
    df_enriched = df_enriched.merge(topology_features, on='solar_farm_id', how='left')
    
    # Final Ratio Feature
    df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] / (df_enriched['[REDACTED_BY_SCRIPT]'] * 1000)
    df_enriched['[REDACTED_BY_SCRIPT]'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Finalization ---
    new_cols = [col for col in df_enriched.columns if 'ohl_33kv' in col]
    df_enriched[new_cols] = df_enriched[new_cols].fillna(0)
    df_enriched.drop(columns=['solar_farm_id'], inplace=True, errors='ignore')
    
    # --- Persist Final Artifact ---
    output_path = OUTPUT_ARTIFACT
    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()