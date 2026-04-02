import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
from shapely.geometry import LineString

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
LV_LINES_INPUT = r"[REDACTED_BY_SCRIPT]"
LV_POLES_INPUT = r"[REDACTED_BY_SCRIPT]"
L22_DATA_INPUT = '[REDACTED_BY_SCRIPT]'
SUBSTATION_L1_ARTIFACT = '[REDACTED_BY_SCRIPT]'

# Output Artifact
L23_DATA_OUTPUT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
WAYLEAVE_BUFFER_METERS_DLV = 5 # 10m total corridor for distribution LV
LOCAL_NETWORK_BUFFER_METERS = 2000


def load_and_prepare_dlv_assets(lines_path: str, poles_path: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads, re-projects, and prepares the raw LV datasets.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    dlv_lines = gpd.read_file(lines_path)
    dlv_poles = gpd.read_file(poles_path)

    # MANDATE: Unconditional CRS Unification.
    dlv_lines = dlv_lines.to_crs(TARGET_CRS)
    dlv_poles = dlv_poles.to_crs(TARGET_CRS)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # MANDATE: Schema Normalization.
    dlv_lines.columns = [col.lower().strip() for col in dlv_lines.columns]
    dlv_poles.columns = [col.lower().strip() for col in dlv_poles.columns]

    # MANDATE: Create D-LV Wayleave Corridors.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    dlv_corridors = dlv_lines.copy()
    dlv_corridors['geometry'] = dlv_lines.geometry.buffer(WAYLEAVE_BUFFER_METERS_DLV)
    
    return dlv_lines, dlv_poles, dlv_corridors


def generate_dlv_features(gdf_solar: gpd.GeoDataFrame, dlv_lines: gpd.GeoDataFrame, dlv_poles: gpd.GeoDataFrame, dlv_corridors: gpd.GeoDataFrame, gdf_substations: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Generates density and frictional features for the D-LV network.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Ambient Settlement & Density Features ---
    buffer_area_km2 = np.pi * ((LOCAL_NETWORK_BUFFER_METERS / 1000) ** 2)
    gdf_solar_buffers = gdf_solar.copy()
    gdf_solar_buffers['geometry'] = gdf_solar_buffers.geometry.buffer(LOCAL_NETWORK_BUFFER_METERS)
    
    clipped_lines = gpd.overlay(dlv_lines, gdf_solar_buffers, how='intersection')
    clipped_lines['length_km'] = clipped_lines.geometry.length / 1000
    line_density_summary = clipped_lines.groupby('solar_farm_id')['length_km'].sum().reset_index()
    line_density_summary['[REDACTED_BY_SCRIPT]'] = line_density_summary['length_km'] / buffer_area_km2

    poles_in_buffer = gpd.sjoin(dlv_poles, gdf_solar_buffers, how="inner", predicate="within")
    pole_counts = poles_in_buffer.groupby('solar_farm_id').size().reset_index(name='dlv_pole_count_2km')
    pole_counts['[REDACTED_BY_SCRIPT]'] = pole_counts['dlv_pole_count_2km'] / buffer_area_km2

    # --- Frictional Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_nearest_sub = gpd.sjoin_nearest(gdf_solar, gdf_substations, how='left', distance_col="dist_to_sub")
    gdf_nearest_sub = gdf_nearest_sub.drop_duplicates(subset=['solar_farm_id'], keep='first')
    
    substation_geoms = gdf_substations.geometry
    nearest_substation_geoms = gdf_nearest_sub['index_right'].map(substation_geoms)
    
    connection_paths = [
        LineString([site_geom, sub_geom]) if pd.notna(sub_geom) else None
        for site_geom, sub_geom in zip(gdf_nearest_sub.geometry, nearest_substation_geoms)
    ]
    gdf_paths = gpd.GeoDataFrame(gdf_nearest_sub[['solar_farm_id']], geometry=connection_paths, crs=TARGET_CRS).dropna(subset=['geometry'])

    if not gdf_paths.empty:
        path_intersections = gpd.sjoin(gdf_paths, dlv_corridors, how='inner', predicate='intersects')
        intersection_counts = path_intersections.groupby('solar_farm_id').size().reset_index(name='[REDACTED_BY_SCRIPT]')
    else:
        intersection_counts = pd.DataFrame(columns=['solar_farm_id', '[REDACTED_BY_SCRIPT]'])

    # --- Combine all features ---
    df_features = gdf_solar[['solar_farm_id']].copy()
    df_features = df_features.merge(line_density_summary[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(pole_counts[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(intersection_counts, on='solar_farm_id', how='left')
    
    return df_features


def main():
    """
    Main function to execute the D-LV network analysis and capstone synthesis.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest and Prepare D-LV Assets ---
    dlv_lines, dlv_poles, dlv_corridors = load_and_prepare_dlv_assets(LV_LINES_INPUT, LV_POLES_INPUT)
    
    # --- Load Prerequisite Artifacts ---
    try:
        df_l22 = pd.read_csv(L22_DATA_INPUT)
        gdf_solar = gpd.GeoDataFrame(
            df_l22, geometry=gpd.points_from_xy(df_l22.easting_x, df_l22.northing_x), crs=TARGET_CRS
        )
        gdf_solar['solar_farm_id'] = gdf_solar.index
        
        gdf_substations = gpd.read_file(SUBSTATION_L1_ARTIFACT)
        if gdf_substations.crs != TARGET_CRS:
            gdf_substations = gdf_substations.to_crs(TARGET_CRS)
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- Phase 2: D-LV Feature Generation ---
    dlv_features = generate_dlv_features(gdf_solar, dlv_lines, dlv_poles, dlv_corridors, gdf_substations)
    
    # --- Integration ---
    df_enriched = gdf_solar.merge(dlv_features, on='solar_farm_id', how='left')

    # --- Phase 3: The Capstone Synthesis (Amaryllis Alpha) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Fill NaNs with 0 before synthesis to ensure correct summation.
    dlv_cols = [col for col in df_enriched.columns if 'dlv_' in col]
    df_enriched[dlv_cols] = df_enriched[dlv_cols].fillna(0)
    
    # 1. Total Connection Friction Index
    df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] + df_enriched['[REDACTED_BY_SCRIPT]']
    
    # ARCHITECTURAL NOTE: Total sterilization is omitted as solar sites are points, making area-based calculations invalid.
    
    # --- Finalization ---
    final_cols = list(df_l22.columns) + dlv_cols + ['[REDACTED_BY_SCRIPT]']
    df_final = df_enriched[final_cols]
    
    df_final.drop(columns=['geometry'], inplace=True, errors='ignore')
    
    # --- Persist Final Artifact ---
    df_final.to_csv(L23_DATA_OUTPUT, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")


if __name__ == "__main__":
    main()