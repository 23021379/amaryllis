import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input for this directive
TRANSFORMER_ASSETS_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'


def load_and_aggregate_transformer_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Loads, sanitizes, and aggregates the raw transformer asset data into a stable, 
    one-row-per-site L1 artifact.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_tx = gpd.read_file(filepath)

    # MANDATE: Unconditional CRS Unification.
    gdf_tx = gdf_tx.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Schema Normalization.
    gdf_tx.columns = [col.lower().strip().replace(' ', '_') for col in gdf_tx.columns]
    
    # MANDATE: Paranoid Type Casting and Unit Verification.
    gdf_tx['onanrating_kva'] = pd.to_numeric(gdf_tx['onanrating_kva'], errors='coerce')
    gdf_tx.dropna(subset=['onanrating_kva', 'sitefunctionallocation'], inplace=True)
    
    # The values are in MVA, not kVA. Convert to kVA to standardize units.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_tx['onan_rating_kva'] = gdf_tx['onanrating_kva'] * 1000

    # MANDATE: Site-Level Aggregation to prevent granularity mismatch.
    logging.info("[REDACTED_BY_SCRIPT]")
    agg_rules = {
        'onan_rating_kva': ['sum', 'count', 'mean', 'max', 'min', 'var']
    }
    aggregated_data = gdf_tx.groupby('sitefunctionallocation').agg(agg_rules)
    
    # Flatten the multi-index columns and rename for clarity.
    aggregated_data.columns = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    aggregated_data['[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True) # Variance is NaN for single-transformer sites.

    # Re-join with geometry to create the final GeoDataFrame L1 artifact.
    sites_geometry = gdf_tx[['sitefunctionallocation', 'geometry']].drop_duplicates(subset='sitefunctionallocation').set_index('sitefunctionallocation')
    gdf_tx_l1 = sites_geometry.join(aggregated_data).reset_index()

    logging.info("[REDACTED_BY_SCRIPT]")
    return gdf_tx_l1

def main():
    """
    Main function to execute the transformer asset feature enrichment pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest and Aggregate ---
    gdf_tx_l1 = load_and_aggregate_transformer_data(TRANSFORMER_ASSETS_INPUT)
    
    # --- Load Solar Data ---
    df_solar_l16 = pd.read_csv(SOLAR_DATA_INPUT)

    # --- Phase 2: Integration and Feature Engineering ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_enriched = df_solar_l16.merge(
        gdf_tx_l1.drop(columns='geometry'), # Drop geometry before non-spatial merge
        left_on='[REDACTED_BY_SCRIPT]',
        right_on='sitefunctionallocation',
        how='left'
    )
    df_enriched.drop(columns=['sitefunctionallocation'], inplace=True, errors='ignore')

    # MANDATE: Final Feature Interaction.
    # Convert solar farm MW to kVA for a unit-consistent ratio.
    df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] / (df_enriched['[REDACTED_BY_SCRIPT]'] * 1000)
    # Handle division by zero for sites with no transformer data or zero capacity farms.
    df_enriched['[REDACTED_BY_SCRIPT]'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Finalization ---
    new_cols = [col for col in df_enriched.columns if 'primary_sub_' in col or '[REDACTED_BY_SCRIPT]' in col]
    # Filter to only new columns that were actually in L16 to avoid re-filling old ones
    new_cols_to_fill = [col for col in new_cols if col not in df_solar_l16.columns]
    df_enriched[new_cols_to_fill] = df_enriched[new_cols_to_fill].fillna(0)
    
    # --- Persist Final Artifact ---
    output_path =  OUTPUT_ARTIFACT
    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()