import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Configuration variables
LCT_DATA_INPUT = r"[REDACTED_BY_SCRIPT]' Secondary Sites.geojson"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'
# Define data directories for clarity
RAW_DATA_DIR = 'raw_data'
PROCESSED_DATA_DIR = 'processed_data'

BUFFER_RADIUS_METERS = 5000

def load_and_prepare_lct_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Loads, re-projects, and sanitizes the LCT GeoJSON data.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_lct = gpd.read_file(filepath)

    # MANDATE: Unconditional CRS Unification to mitigate Pattern 1.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_lct = gdf_lct.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Proactive Schema Normalization.
    gdf_lct.columns = [col.lower().strip() for col in gdf_lct.columns]
    
    # Rename for clarity and consistency
    gdf_lct.rename(columns={'lct_connections': 'lct_connections', '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]'}, inplace=True)

    # MANDATE: Paranoid Type Casting.
    numeric_cols = ['lct_connections', 'import', 'export']
    for col in numeric_cols:
        gdf_lct[col] = pd.to_numeric(gdf_lct[col], errors='coerce')
    
    # Data cleaning and validation
    initial_rows = len(gdf_lct)
    gdf_lct.dropna(subset=['lct_connections', '[REDACTED_BY_SCRIPT]'], inplace=True)
    if len(gdf_lct) < initial_rows:
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    # Clean categorical columns to prevent aggregation errors
    for col in ['category', 'type']:
        if col in gdf_lct.columns:
            gdf_lct[col] = gdf_lct[col].str.strip().str.title()

    return gdf_lct

def calculate_buffer_features(gdf_solar: gpd.GeoDataFrame, gdf_lct: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculates ambient LCT density and character features within a buffer around each solar application.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_solar_buffers = gdf_solar.copy()
    gdf_solar_buffers['geometry'] = gdf_solar_buffers.geometry.buffer(BUFFER_RADIUS_METERS)

    logging.info("[REDACTED_BY_SCRIPT]")
    sjoined = gpd.sjoin(gdf_lct, gdf_solar_buffers, how="inner", predicate="within")

    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Stratified aggregations using the correct, stable key
    grouped = sjoined.groupby('solar_farm_id')
    
    # Base aggregations
    agg_df = grouped.agg(
        lct_secondary_sub_count_in_5km=('solar_farm_id', 'size'),
        lct_total_connections_in_5km=('lct_connections', 'sum'),
        lct_total_import_kw_in_5km=('import', 'sum'),
        lct_total_export_kw_in_5km=('export', 'sum')
    )

    # MANDATE: Stratified Sentiment & Saturation Features
    agg_df['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['category'] == 'Demand']['lct_connections'].sum())
    agg_df['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['category'] == 'Generation']['lct_connections'].sum())
    agg_df['lct_ev_connections_in_5km'] = grouped.apply(lambda x: x[x['type'] == 'Ev Charging Point']['lct_connections'].sum())
    agg_df['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['type'] == 'Solar']['lct_connections'].sum())

    # MANDATE: The Critical Ratio Feature with division-by-zero guard
    agg_df['[REDACTED_BY_SCRIPT]'] = agg_df['[REDACTED_BY_SCRIPT]'] / (agg_df['[REDACTED_BY_SCRIPT]'] + 1)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    return agg_df


def calculate_primary_substation_features(gdf_lct: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Aggregates LCT data to the primary substation level.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    grouped_primary = gdf_lct.groupby('[REDACTED_BY_SCRIPT]')
    
    primary_summary = grouped_primary.agg(
        primary_sub_total_lct_connections=('lct_connections', 'sum'),
        primary_sub_total_lct_import_kw=('import', 'sum'),
        primary_sub_total_lct_export_kw=('export', 'sum')
    )
    
    primary_summary['[REDACTED_BY_SCRIPT]'] = grouped_primary.apply(lambda x: x[x['category'] == 'Demand']['lct_connections'].sum())
    primary_summary['[REDACTED_BY_SCRIPT]'] = grouped_primary.apply(lambda x: x[x['category'] == 'Generation']['lct_connections'].sum())
    
    # Critical ratio at primary substation level
    primary_summary['[REDACTED_BY_SCRIPT]'] = primary_summary['[REDACTED_BY_SCRIPT]'] / (primary_summary['[REDACTED_BY_SCRIPT]'] + 1)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    return primary_summary


def main():
    """
    Main function to execute the LCT feature enrichment pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    gdf_lct_unified = load_and_prepare_lct_data(LCT_DATA_INPUT)
    
    df_solar = pd.read_csv(SOLAR_DATA_INPUT)
    
    gdf_solar = gpd.GeoDataFrame(
        df_solar, 
        geometry=gpd.points_from_xy(df_solar['easting'], df_solar['northing']),
        crs="EPSG:27700"
    )
    # Create a stable, unique identifier for joining, as one does not exist in the L12 artifact.
    gdf_solar['solar_farm_id'] = gdf_solar.index
    df_solar['solar_farm_id'] = df_solar.index
    
    # --- Forge the missing link between solar farms and primary substations ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Create a unique GeoDataFrame of primary substations
    gdf_primary_subs = gdf_lct_unified.drop_duplicates(subset='[REDACTED_BY_SCRIPT]').set_index('[REDACTED_BY_SCRIPT]')
    
    # Find the nearest primary substation for each solar farm
    gdf_solar = gpd.sjoin_nearest(gdf_solar, gdf_primary_subs[['geometry']], how='left')
    # The join adds 'index_right' which is the primary_functionallocation. Rename for clarity.
    gdf_solar.rename(columns={'index_right': '[REDACTED_BY_SCRIPT]'}, inplace=True)
    # Update the original dataframe with this new, critical information
    df_solar = df_solar.merge(gdf_solar[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Phase 2: Calculate buffer features
    buffer_features = calculate_buffer_features(gdf_solar, gdf_lct_unified)
    
    # Phase 3: Calculate primary substation features
    primary_sub_features = calculate_primary_substation_features(gdf_lct_unified)
    
    # Final Integration
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Join buffer features using the correct key
    df_enriched = df_solar.merge(buffer_features, on='solar_farm_id', how='left')
    
    # Join primary substation features using the now-correct key
    df_enriched = df_enriched.merge(primary_sub_features, on='[REDACTED_BY_SCRIPT]', how='left')
    
    # Fill NaNs with -1 for applications with no LCTs in their buffer/area
    lct_feature_columns = [col for col in df_enriched.columns if 'lct_' in col or 'primary_sub_' in col]
    df_enriched[lct_feature_columns] = df_enriched[lct_feature_columns].fillna(-1.0)

    # Remove temporary ID column before saving
    df_enriched.drop(columns=['solar_farm_id'], inplace=True, errors='ignore')
    
    # Persist final artifact
    df_enriched.drop(columns=['geometry'], inplace=True, errors='ignore')
    df_enriched.to_csv(OUTPUT_ARTIFACT, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()