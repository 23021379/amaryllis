import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')


# Inputs for this directive
HV_LINES_INPUT = r"[REDACTED_BY_SCRIPT]"
HV_POLES_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'
SUBSTATION_L1_ARTIFACT = '[REDACTED_BY_SCRIPT]' # Mandated authoritative substation artifact

# Architectural Parameters
WAYLEAVE_BUFFER_METERS_DHV = 20 # Mandated 20m buffer for D-HV wayleave corridors
LOCAL_NETWORK_BUFFER_METERS = 2000

from typing import Tuple
from shapely.geometry import LineString

def load_and_prepare_dhv_assets(lines_path: str, poles_path: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads and prepares the pure D-HV datasets, enforcing CRS and synthesizing wayleaves.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    dhv_lines = gpd.read_file(lines_path)
    dhv_poles = gpd.read_file(poles_path)

    # MANDATE: Unconditional CRS Unification.
    dhv_lines = dhv_lines.to_crs("EPSG:27700")
    dhv_poles = dhv_poles.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Schema Normalization.
    dhv_lines.columns = [col.lower().strip() for col in dhv_lines.columns]
    dhv_poles.columns = [col.lower().strip() for col in dhv_poles.columns]
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Create D-HV Wayleave Corridors.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    dhv_corridors = dhv_lines.copy()
    dhv_corridors['geometry'] = dhv_lines.geometry.buffer(WAYLEAVE_BUFFER_METERS_DHV)
    
    return dhv_lines, dhv_poles, dhv_corridors


def generate_dhv_features(gdf_solar: gpd.GeoDataFrame, dhv_lines: gpd.GeoDataFrame, dhv_poles: gpd.GeoDataFrame, dhv_corridors: gpd.GeoDataFrame, gdf_substations: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Generates foundational, topological, and frictional features for the D-HV network.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Foundational Features (Logic is sound and retained) ---
    buffer_area_km2 = np.pi * ((LOCAL_NETWORK_BUFFER_METERS / 1000) ** 2)
    gdf_solar_buffers = gdf_solar.copy()
    gdf_solar_buffers['geometry'] = gdf_solar_buffers.geometry.buffer(LOCAL_NETWORK_BUFFER_METERS)
    
    clipped_lines_for_density = gpd.overlay(dhv_lines, gdf_solar_buffers, how='intersection')
    clipped_lines_for_density['length_km'] = clipped_lines_for_density.geometry.length / 1000
    line_density_summary = clipped_lines_for_density.groupby('solar_farm_id')['length_km'].sum().reset_index()
    line_density_summary['[REDACTED_BY_SCRIPT]'] = line_density_summary['length_km'] / buffer_area_km2

    poles_in_buffer = gpd.sjoin(dhv_poles, gdf_solar_buffers, how="inner", predicate="within")
    pole_counts = poles_in_buffer.groupby('solar_farm_id').size().reset_index(name='dhv_pole_count_2km')
    pole_counts['[REDACTED_BY_SCRIPT]'] = pole_counts['dhv_pole_count_2km'] / buffer_area_km2

    # --- Topological & Frictional Features (The Alpha) - CORRECTED IMPLEMENTATION ---
    logging.info("[REDACTED_BY_SCRIPT]'Alpha' features...")
    
    # 1. Connection Path Intersection Count
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # ARCHITECTURAL NOTE (MANDATE AD-GRID-AUDIT-003): This operation is based on the "nearest is governing"
    # approximation. It assumes the connection path for a solar farm is a straight line to the
    # geographically nearest primary substation. This is a powerful proxy for connection friction,
    # but it is not ground truth, as the actual connection point may differ due to engineering,
    # land access, or other constraints. The resulting feature's meaning must be interpreted
    # with this accepted architectural approximation in mind.
    gdf_nearest_sub = gpd.sjoin_nearest(gdf_solar, gdf_substations, how='left', distance_col="dist_to_sub")
    gdf_nearest_sub = gdf_nearest_sub.drop_duplicates(subset=['solar_farm_id'], keep='first')

    # Create a clean lookup Series for substation geometries, indexed by their original index.
    substation_geoms = gdf_substations.geometry

    # Map the 'index_right' (the index of the nearest substation) to its geometry.
    # This is a vectorized lookup, far more efficient and correct than the previous .apply().
    nearest_substation_geoms = gdf_nearest_sub['index_right'].map(substation_geoms)

    # Construct the LineString geometries. Handle cases where no substation was found (geom is NaT).
    # The solar site geometry is the primary geometry of the gdf_nearest_sub frame.
    connection_paths = [
        LineString([site_geom, sub_geom]) if pd.notna(sub_geom) else None
        for site_geom, sub_geom in zip(gdf_nearest_sub.geometry, nearest_substation_geoms)
    ]
    
    # Create a clean GeoDataFrame of the connection paths.
    gdf_paths = gpd.GeoDataFrame(
        gdf_nearest_sub[['solar_farm_id']],
        geometry=connection_paths,
        crs="EPSG:27700"
    ).dropna(subset=['geometry'])

    if not gdf_paths.empty:
        path_intersections = gpd.sjoin(gdf_paths, dhv_corridors, how='inner', predicate='intersects')
        intersection_counts = path_intersections.groupby('solar_farm_id').size().reset_index(name='[REDACTED_BY_SCRIPT]')
    else:
        intersection_counts = pd.DataFrame(columns=['solar_farm_id', '[REDACTED_BY_SCRIPT]'])

    # 2. Network Complexity (Logic is sound and retained)
    clipped_lines_for_complexity = gpd.sjoin(dhv_lines, gdf_solar_buffers, how="inner", predicate="intersects")
    line_counts = clipped_lines_for_complexity.groupby('solar_farm_id').size().reset_index(name='line_segment_count')
    clipped_lines_for_complexity['length_m'] = clipped_lines_for_complexity.geometry.length
    line_lengths = clipped_lines_for_complexity.groupby('solar_farm_id')['length_m'].sum().reset_index()
    complexity_df = pd.merge(line_counts, line_lengths, on='solar_farm_id')
    complexity_df['[REDACTED_BY_SCRIPT]'] = (complexity_df['line_segment_count'] / complexity_df['length_m']).fillna(0)
    complexity_df.replace([np.inf, -np.inf], 0, inplace=True)

    # --- Combine all features ---
    df_features = gdf_solar[['solar_farm_id']].copy() # Site area is irrelevant for point data.
    df_features = df_features.merge(line_density_summary[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(pole_counts[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(intersection_counts, on='solar_farm_id', how='left')
    df_features = df_features.merge(complexity_df[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    
    return df_features



def main():
    """
    Main function to execute the D-HV network topological analysis pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest and Prepare D-HV Assets ---
    dhv_lines, dhv_poles, dhv_corridors = load_and_prepare_dhv_assets(
        HV_LINES_INPUT,
        HV_POLES_INPUT
    )
    
    # --- Load and Prepare Solar Data ---
    df_solar_l21 = pd.read_csv(SOLAR_DATA_INPUT)
    # Reverting to point geometry creation based on the verified L21 artifact contract.
    gdf_solar = gpd.GeoDataFrame(
        df_solar_l21,
        geometry=gpd.points_from_xy(df_solar_l21.easting_x, df_solar_l21.northing_x),
        crs="EPSG:27700"
    )
    gdf_solar['solar_farm_id'] = gdf_solar.index

    # --- NEW MANDATE: Load the Authoritative Substation Artifact ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    try:
        gdf_substations = gpd.read_file(SUBSTATION_L1_ARTIFACT)
        if gdf_substations.crs != "EPSG:27700":
            logging.warning("[REDACTED_BY_SCRIPT]")
            gdf_substations = gdf_substations.to_crs("EPSG:27700")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- Phase 2: D-HV Feature Generation ---
    dhv_features = generate_dhv_features(
        gdf_solar, dhv_lines, dhv_poles, dhv_corridors, gdf_substations
    )
    
    # --- Integration ---
    # Merge the new features back to the authoritative GeoDataFrame that holds the key.
    df_enriched = gdf_solar.merge(dhv_features, on='solar_farm_id', how='left')

    # --- Finalization ---
    # The cross-strata sterilization feature is removed as its D-HV component is invalid.
    new_cols = [col for col in df_enriched.columns if 'dhv_' in col]
    df_enriched[new_cols] = df_enriched[new_cols].fillna(0)
    df_enriched.drop(columns=['solar_farm_id'], inplace=True, errors='ignore')
    
    # --- Persist Final Artifact ---
    output_path = OUTPUT_ARTIFACT
    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()