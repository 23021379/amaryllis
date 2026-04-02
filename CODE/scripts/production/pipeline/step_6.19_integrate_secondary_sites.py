import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
SECONDARY_SUB_INPUT = r"[REDACTED_BY_SCRIPT]"
L24_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output Artifact
L25_DATA_OUTPUT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
K_NEIGHBORS = 50
REINFORCEMENT_HORIZON_YRS = 5
NULL_SENTINEL_YEAR = 9999


def parse_and_clean_secondary_subs(gdf_raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Executes the robust parsing and decontamination protocol for the raw secondary substation data.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_raw.columns = [col.lower().strip() for col in gdf_raw.columns]

    # --- NEW MANDATE: Geospatial Decontamination Gate ---
    initial_count = len(gdf_raw)
    invalid_geom_mask = gdf_raw.geometry.is_empty | gdf_raw.geometry.isna()
    if invalid_geom_mask.any():
        num_invalid = invalid_geom_mask.sum()
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf_raw = gdf_raw[~invalid_geom_mask].copy()
        logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- MANDATE: Robust Parsing of Hostile String Formats ---
    # Parse ONAN rating (e.g., "500 kVA" -> 500.0)
    gdf_raw['onan_rating_kva'] = pd.to_numeric(gdf_raw['onanrating'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

    # Parse utilisation band (e.g., "20-40%" -> 0.30)
    bands = gdf_raw['utilisation_band'].str.extract(r'(\d+)-(\d+)%')
    low = pd.to_numeric(bands[0], errors='coerce')
    high = pd.to_numeric(bands[1], errors='coerce')
    gdf_raw['utilisation_midpoint_pct'] = ((low + high) / 2) / 100.0

    # One-hot encode substation design (GMT vs PMT)
    gdf_raw['is_gmt'] = (gdf_raw['substationdesign'] == 'GMT').astype(int)

    # Parse and guard the reinforcement year
    gdf_raw['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(gdf_raw['[REDACTED_BY_SCRIPT]'], errors='coerce')
    gdf_raw['[REDACTED_BY_SCRIPT]'].fillna(NULL_SENTINEL_YEAR, inplace=True)

    # Schema Pruning
    essential_cols = [
        'geometry', 'functionallocation', '[REDACTED_BY_SCRIPT]',
        'onan_rating_kva', 'utilisation_midpoint_pct', 'is_gmt', 'customer_count',
        '[REDACTED_BY_SCRIPT]'
    ]
    return gdf_raw[essential_cols]


def main():
    """
    Main function to execute the granular synthesis protocol for secondary substations.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Phase 1: Ingest, Parse, and Prepare L1 Artifact ---
    try:
        gdf_sec_subs_raw = gpd.read_file(SECONDARY_SUB_INPUT)
        gdf_sec_subs_raw = gdf_sec_subs_raw.to_crs(TARGET_CRS)
        gdf_sec_subs_l1 = parse_and_clean_secondary_subs(gdf_sec_subs_raw)
        
        df_l24 = pd.read_csv(L24_DATA_INPUT)
        gdf_solar = gpd.GeoDataFrame(
            df_l24, geometry=gpd.points_from_xy(df_l24.easting_x, df_l24.northing_x), crs=TARGET_CRS
        )
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- Phase 2: Local Environment Feature Generation via k-NN ---
    logging.info("[REDACTED_BY_SCRIPT]")
    sec_sub_coords = np.array(list(zip(gdf_sec_subs_l1.geometry.x, gdf_sec_subs_l1.geometry.y)))
    sec_sub_tree = cKDTree(sec_sub_coords)

    knn_results = []
    for index, site in tqdm(gdf_solar.iterrows(), total=len(gdf_solar), desc="[REDACTED_BY_SCRIPT]"):
        distances, indices = sec_sub_tree.query([site.geometry.x, site.geometry.y], k=K_NEIGHBORS)
        neighbors = gdf_sec_subs_l1.iloc[indices]
        
        # Local Character Features
        gmt_ratio = neighbors['is_gmt'].mean()
        avg_customer_count = neighbors['customer_count'].mean()
        
        # Local Grid Stress Features
        avg_utilisation = neighbors['utilisation_midpoint_pct'].mean()
        high_utilisation_count = (neighbors['utilisation_midpoint_pct'] > 0.8).sum()
        
        # MANDATE: Temporal Guard Protocol for Future Constraints
        submission_year = site['submission_year']
        imminent_reinforcement_mask = (
            (neighbors['[REDACTED_BY_SCRIPT]'] >= submission_year) &
            (neighbors['[REDACTED_BY_SCRIPT]'] < submission_year + REINFORCEMENT_HORIZON_YRS)
        )
        reinforcement_needed_count = imminent_reinforcement_mask.sum()
        
        knn_results.append({
            'index': index,
            '[REDACTED_BY_SCRIPT]': gmt_ratio,
            '[REDACTED_BY_SCRIPT]': avg_customer_count,
            '[REDACTED_BY_SCRIPT]': avg_utilisation,
            '[REDACTED_BY_SCRIPT]': high_utilisation_count,
            '[REDACTED_BY_SCRIPT]': reinforcement_needed_count
        })
    
    df_knn_features = pd.DataFrame(knn_results).set_index('index')

    # --- Phase 3: Aggregation to Primary Substation ---
    logging.info("[REDACTED_BY_SCRIPT]")
    primary_sub_agg = gdf_sec_subs_l1.groupby('[REDACTED_BY_SCRIPT]').agg(
        primary_sub_downstream_sec_sub_count=('functionallocation', 'count'),
        primary_sub_downstream_total_customers=('customer_count', 'sum'),
        primary_sub_downstream_total_kva=('onan_rating_kva', 'sum'),
        primary_sub_downstream_avg_utilisation=('utilisation_midpoint_pct', 'mean'),
        primary_sub_downstream_gmt_ratio=('is_gmt', 'mean')
    ).reset_index()

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Merge k-NN features
    df_enriched = gdf_solar.merge(df_knn_features, left_index=True, right_index=True, how='left')
    
    # Merge aggregated primary sub features
    # The authoritative key for the governing primary substation in the L24 artifact is '[REDACTED_BY_SCRIPT]'.
    df_enriched = df_enriched.merge(
        primary_sub_agg,
        left_on='[REDACTED_BY_SCRIPT]',
        right_on='[REDACTED_BY_SCRIPT]',
        how='left'
    )

    # --- Finalization ---
    df_final = df_enriched.drop(columns=['geometry', '[REDACTED_BY_SCRIPT]'], errors='ignore')
    
    # Persist Final Artifact
    df_final.to_csv(L25_DATA_OUTPUT, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()