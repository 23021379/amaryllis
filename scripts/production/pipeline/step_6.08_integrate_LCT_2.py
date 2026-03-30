import pandas as pd
import geopandas as gpd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')


# Inputs from previous directives
GRANULAR_LCT_INPUT =  r"[REDACTED_BY_SCRIPT]' Secondary Sites.geojson"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# New input for this directive
AGGREGATE_LCT_INPUT = r"[REDACTED_BY_SCRIPT]"

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'


def load_and_prepare_primary_lct_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Loads, sanitizes, and aggregates the new primary substation LCT data.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(filepath)

    # MANDATE: Proactive Schema Normalization.
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]

    # MANDATE: Paranoid Type Casting.
    numeric_cols = ['lct_connections', 'importrating', 'exportrating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols + ['sitefunctionallocation'], inplace=True)

    # MANDATE: Geospatial Integrity. Create geometry and immediately unify CRS.
    df[['latitude', 'longitude']] = df['spatialcoordinates'].str.split(',', expand=True).astype(float)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    gdf = gdf.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # Aggregate to create stratified features by primary substation
    grouped = gdf.groupby('sitefunctionallocation')
    
    agg_base = grouped.agg(
        primary_direct_total_connections=('lct_connections', 'sum'),
        primary_direct_total_import_kw=('importrating', 'sum'),
        primary_direct_total_export_kw=('exportrating', 'sum')
    )
    
    agg_base['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['category'] == 'Demand']['lct_connections'].sum())
    agg_base['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['category'] == 'Generation']['lct_connections'].sum())
    agg_base['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['type'] == 'Solar']['lct_connections'].sum())
    agg_base['[REDACTED_BY_SCRIPT]'] = agg_base['[REDACTED_BY_SCRIPT]'] / (agg_base['[REDACTED_BY_SCRIPT]'] + 1)
    
    return agg_base.reset_index()

def calculate_max_hotspot_per_primary(filepath: str) -> pd.DataFrame:
    """
    Calculates the size of the largest secondary substation LCT hotspot for each primary substation.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_granular = gpd.read_file(filepath)
    gdf_granular.columns = [col.lower().strip() for col in gdf_granular.columns]
    
    # Ensure numeric type
    gdf_granular['lct_connections'] = pd.to_numeric(gdf_granular['lct_connections'], errors='coerce').fillna(-1.0)

    # Group by primary and find the max lct_connections from any single secondary sub
    concentration_lookup = gdf_granular.groupby('[REDACTED_BY_SCRIPT]')['lct_connections'].max().reset_index()
    concentration_lookup.rename(columns={'lct_connections': '[REDACTED_BY_SCRIPT]'}, inplace=True)
    
    return concentration_lookup

def main():
    """
    Main function to execute the LCT synthesis pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Load all necessary data artifacts ---
    df_solar_l13 = pd.read_csv(SOLAR_DATA_INPUT)
    
    primary_lct_agg = load_and_prepare_primary_lct_data(
        AGGREGATE_LCT_INPUT
    )
    
    concentration_lookup = calculate_max_hotspot_per_primary(
        GRANULAR_LCT_INPUT
    )

    # --- Phase 1: Join Standalone Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_enriched = df_solar_l13.merge(
        primary_lct_agg,
        left_on='[REDACTED_BY_SCRIPT]',
        right_on='sitefunctionallocation',
        how='left'
    )
    # Drop redundant join key
    df_enriched.drop(columns=['sitefunctionallocation'], inplace=True, errors='ignore')

    # --- Phase 2: Synthesize Cross-Dataset Features (The Alpha) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Feature 2.1: The Reconciliation Delta
    df_enriched['lct_reconciliation_delta_connections'] = df_enriched['[REDACTED_BY_SCRIPT]'] - df_enriched['[REDACTED_BY_SCRIPT]']
    df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] - df_enriched['[REDACTED_BY_SCRIPT]']

    # Feature 2.2: The LCT Concentration Index
    df_enriched = df_enriched.merge(concentration_lookup, on='[REDACTED_BY_SCRIPT]', how='left')
    df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] / (df_enriched['[REDACTED_BY_SCRIPT]'] + 1)

    # Feature 2.3: The Aggregation Completeness Ratio
    df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] / (df_enriched['[REDACTED_BY_SCRIPT]'] + 1)
    
    # --- Finalization ---
    # Fill all new columns with -1 where no data was available
    new_cols = [col for col in df_enriched.columns if col not in df_solar_l13.columns]
    df_enriched[new_cols] = df_enriched[new_cols].fillna(-1.0)

    output_path = OUTPUT_ARTIFACT
    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()