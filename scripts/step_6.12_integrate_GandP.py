import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')


# Input for this directive
SUBSTATION_BIO_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'


def load_and_prepare_substation_bio_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Loads, robustly parses, and sanitizes the substation biography data into a stable L1 artifact.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_bio = gpd.read_file(filepath)

    # MANDATE: Unconditional CRS Unification.
    gdf_bio = gdf_bio.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Schema Normalization.
    gdf_bio.columns = [col.lower().strip().replace('.', '_') for col in gdf_bio.columns]

    # MANDATE: Robust Parsing for inconsistent and sparse columns.
    # Dates
    for col in ['datecommissioned', 'assessmentdate']:
        gdf_bio[col] = pd.to_datetime(gdf_bio[col], errors='coerce')

    # Resistance (fallback logic)
    gdf_bio['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(gdf_bio['[REDACTED_BY_SCRIPT]'], errors='coerce')
    
    def parse_calc_resistance(value):
        if pd.isna(value):
            return np.nan
        match = re.search(r'(\d+\.?\d*)', str(value))
        return float(match.group(1)) if match else np.nan
        
    gdf_bio['[REDACTED_BY_SCRIPT]'] = gdf_bio['calculatedresistance'].apply(parse_calc_resistance)
    gdf_bio['resistance_ohm'] = gdf_bio['[REDACTED_BY_SCRIPT]'].fillna(gdf_bio['[REDACTED_BY_SCRIPT]'])

    # Demand, Ratings, and Counts
    numeric_cols = [
        'powertransformercount', 'transratingwinter', 'transratingsummer',
        'maxdemandsummer', 'maxdemandwinter'
    ]
    for col in numeric_cols:
        gdf_bio[col] = pd.to_numeric(gdf_bio[col], errors='coerce')

    # Clean up intermediate columns
    gdf_bio.drop(columns=['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'calculatedresistance'], inplace=True)
    
    # MANDATE: Encode 'reversepower'
    def encode_reverse_power(value):
        if pd.isna(value):
            return np.nan
        if value == '100%':
            return 1
        elif value == '<100%':
            return 0
        return np.nan
    
    if 'reversepower' in gdf_bio.columns:
        gdf_bio['reversepower_encoded'] = gdf_bio['reversepower'].apply(encode_reverse_power)
        gdf_bio.drop(columns=['reversepower'], inplace=True)

    logging.info("[REDACTED_BY_SCRIPT]")
    return gdf_bio


def main():
    """
    Main function to execute the substation biography feature enrichment pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest and Parse ---
    gdf_bio_l1 = load_and_prepare_substation_bio_data(SUBSTATION_BIO_INPUT)
    
    # --- Load Solar Data ---
    df_solar_l17 = pd.read_csv(SOLAR_DATA_INPUT)
    # Ensure submission_date is datetime for temporal calculations
    df_solar_l17['submission_date'] = pd.to_datetime(df_solar_l17['submission_date'], errors='coerce')

    # --- Phase 2: Integration and Feature Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_enriched = df_solar_l17.merge(
        gdf_bio_l1.drop(columns='geometry'),
        left_on='[REDACTED_BY_SCRIPT]',
        right_on='sitefunctionallocation',
        how='left'
    )

    # Generate Standalone & Point-in-Time Features
    df_enriched['substation_age_at_submission'] = df_enriched['submission_year'] - df_enriched['datecommissioned'].dt.year
    df_enriched['time_since_last_assessment_days'] = (df_enriched['submission_date'] - df_enriched['assessmentdate']).dt.days
    df_enriched['is_hot_site'] = (df_enriched['siteclassification'] == 'HOT').astype(int)

    # Generate Synthetic & Interaction Features (The Alpha)
    df_enriched['[REDACTED_BY_SCRIPT]'] = (df_enriched['powertransformercount'] != df_enriched['[REDACTED_BY_SCRIPT]']).astype(int)
    df_enriched['has_demand_data'] = df_enriched['maxdemandwinter'].notna().astype(int)
    
    # Convert date columns to year for the final artifact, after calculations
    df_enriched['datecommissioned'] = df_enriched['datecommissioned'].dt.year
    df_enriched['assessmentdate'] = df_enriched['assessmentdate'].dt.year
    
    # Unit Conversion Mandate: maxdemand is MW, primary_sub_total_kva is kVA. Convert MW to kVA.
    df_enriched['[REDACTED_BY_SCRIPT]'] = (df_enriched['maxdemandwinter'] * 1000) / df_enriched['[REDACTED_BY_SCRIPT]']
    df_enriched['[REDACTED_BY_SCRIPT]'] = (df_enriched['maxdemandsummer'] * 1000) / df_enriched['[REDACTED_BY_SCRIPT]']
    
    df_enriched['kva_per_transformer'] = df_enriched['[REDACTED_BY_SCRIPT]'] / df_enriched['powertransformercount']
    
    # --- Finalization ---
    # Handle division by zero and infinite values from ratios
    ratio_cols = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'kva_per_transformer']
    df_enriched[ratio_cols] = df_enriched[ratio_cols].replace([np.inf, -np.inf], np.nan)

    # Fill all new columns with a neutral value
    new_cols = [col for col in df_enriched.columns if col not in df_solar_l17.columns]
    df_enriched[new_cols] = df_enriched[new_cols].fillna(0)
    
    # --- Persist Final Artifact ---
    output_path = OUTPUT_ARTIFACT
    # Drop geometry and other unneeded columns before saving
    df_enriched.drop(columns=[col for col in df_enriched.columns if 'geometry' in str(col) or 'sitefunctionallocation' in str(col)], inplace=True, errors='ignore')
    strings_need_dropping=["licencearea", "sitename", "sitetype", "gridref", "street", "suburb", "town", "county", "postcode", "last_report", "northing_y", "easting_y", "local_authority", "[REDACTED_BY_SCRIPT]", "what3words", "siteclassification", "towncity", "next_assessmentdate"]
    df_enriched.drop(columns=strings_need_dropping, inplace=True, errors='ignore')


    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()