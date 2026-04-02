import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm
import logging
import sys

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input/Output Artifacts
L6_DATA_PATH = '[REDACTED_BY_SCRIPT]'
FAULT_LEVEL_CSV_PATH = r"[REDACTED_BY_SCRIPT]"
DFES_GEOJSON_PATH = r"[REDACTED_BY_SCRIPT]" # Required for the geospatial bridge
L7_DATA_PATH = '[REDACTED_BY_SCRIPT]'

# Architectural Hyperparameters
TARGET_CRS = "EPSG:27700"
K_NEIGHBORS = 5
NULL_SENTINEL = -1


import re

def find_column_by_pattern(df: pd.DataFrame, pattern: str) -> str:
    """
    Finds a single column by matching a regex pattern against a
    semantically normalized version of the column names.
    """
    matches = []
    for col in df.columns:
        # The existing normalized columns are already lowercase.
        if re.search(pattern, col):
            matches.append(col) # Append the ORIGINAL (but now lowercase) column name
            
    if len(matches) == 0:
        raise KeyError(f"[REDACTED_BY_SCRIPT]'{pattern}'.")
    if len(matches) > 1:
        raise ValueError(f"[REDACTED_BY_SCRIPT]'{pattern}': {matches}")
    return matches[0]


def main():
    """
    Executes Directive 009. Ingests non-spatial grid stability data, geolocates it
    using the DFES data as a 'Geospatial Bridge', and enriches the L6 solar dataset
    with features describing the engineering constraints of nearby substations.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Load Primary Solar Cohort ---
    try:
        df_l6 = pd.read_csv(L6_DATA_PATH)
        gdf_solar = gpd.GeoDataFrame(
            df_l6, geometry=gpd.points_from_xy(df_l6.easting, df_l6.northing), crs=TARGET_CRS
        )
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- MANDATE 10.1: NORMALIZE-THEN-JOIN PATTERN ---
    
    # 1. INGEST AND NORMALIZE SOURCE A (Fault Level CSV)
    try:
        df_fault = pd.read_csv(FAULT_LEVEL_CSV_PATH)
        # IMMEDIATELY NORMALIZE:
        df_fault.columns = [col.lower().strip() for col in df_fault.columns]
        logging.info("[REDACTED_BY_SCRIPT]")
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # 2. INGEST AND NORMALIZE SOURCE B (DFES GeoJSON for locations)
    try:
        gdf_dfes_locations = gpd.read_file(DFES_GEOJSON_PATH)
        # IMMEDIATELY NORMALIZE:
        gdf_dfes_locations.columns = [col.lower().strip() for col in gdf_dfes_locations.columns]
        logging.info(f"[REDACTED_BY_SCRIPT]")

        # --- ARCHITECTURAL MANDATE: GEOSPATIAL INTEGRITY VALIDATION ---
        # Purge any records from the bridge data that are missing geometry.
        # This prevents propagation of nulls during the join.
        initial_count = len(gdf_dfes_locations)
        gdf_dfes_locations.dropna(subset=['geometry'], inplace=True)
        final_count = len(gdf_dfes_locations)
        
        if initial_count > final_count:
            dropped_count = initial_count - final_count
            logging.warning(f"[REDACTED_BY_SCRIPT]")
        logging.info("[REDACTED_BY_SCRIPT]")
        # --- END MANDATE ---

    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- MANDATE 10.2: Semantic Key Harmonization ---
    # After syntactic normalization, we now harmonize the semantic keys.
    try:
        # Find the actual key name in each dataframe using a robust pattern
        fault_key_actual = find_column_by_pattern(df_fault, r'[REDACTED_BY_SCRIPT]')
        dfes_key_actual = find_column_by_pattern(gdf_dfes_locations, r'[REDACTED_BY_SCRIPT]')
        
        # Rename both to the single, canonical key
        canonical_key = 'sitefunctionalallocation'
        df_fault.rename(columns={fault_key_actual: canonical_key}, inplace=True)
        gdf_dfes_locations.rename(columns={dfes_key_actual: canonical_key}, inplace=True)
        logging.info(f"[REDACTED_BY_SCRIPT]'{canonical_key}'.")
    except (KeyError, ValueError) as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


    # --- MANDATE 9.1: The Geospatial Bridge (Now Guaranteed to Succeed) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_dfes_locations = gdf_dfes_locations.to_crs(TARGET_CRS)
    
    # Create a clean, unique lookup table from the HARMONIZED DFES data
    location_lookup = gdf_dfes_locations[['sitefunctionalallocation', 'geometry']].drop_duplicates(subset='sitefunctionalallocation')

    # Join the HARMONIZED fault data with the lookup on the HARMONIZED key
    gdf_fault_located = pd.merge(
        df_fault,
        location_lookup,
        on='sitefunctionalallocation',
        how='inner'
    )
    gdf_fault_located = gpd.GeoDataFrame(gdf_fault_located, geometry='geometry', crs=TARGET_CRS)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- MANDATE 9.2: Aggregation by Maximum Constraint ---
    logging.info("[REDACTED_BY_SCRIPT]")
    fault_cols_to_agg = ['threephasermsbreak', 'threephasepeakmake', 'earthfaultrmsbreak', 'earthfaultpeakmake']
    substation_fault_profile = gdf_fault_located.groupby('sitefunctionalallocation').agg(
        **{col: (col, 'max') for col in fault_cols_to_agg},
        geometry=('geometry', 'first')
    ).reset_index()
    
    substation_fault_profile = gpd.GeoDataFrame(substation_fault_profile, geometry='geometry', crs=TARGET_CRS)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- ARCHITECTURAL MANDATE: FINAL GUARDRAIL ASSERTION ---
    # This is a paranoid check to ensure no null geometries survived aggregation.
    # The script will halt here if the upstream sanitation fails for any reason.
    assert substation_fault_profile.geometry.notna().all(), "[REDACTED_BY_SCRIPT]"
    logging.info("[REDACTED_BY_SCRIPT]")
    # --- END MANDATE ---

    # --- Build Spatial Index ---
    substation_tree = cKDTree(np.array(list(zip(substation_fault_profile.geometry.x, substation_fault_profile.geometry.y))))

    # --- MANDATE 9.3: KNN Feature Synthesis ---
    results = []
    for index, solar_site in tqdm(gdf_solar.iterrows(), total=gdf_solar.shape[0], desc="[REDACTED_BY_SCRIPT]"):
        target_coord = [solar_site.geometry.x, solar_site.geometry.y]
        distances_m, indices = substation_tree.query(target_coord, k=K_NEIGHBORS)
        neighbors = substation_fault_profile.iloc[indices]
        
        features = {'original_index': index}
        
        # Nearest Substation (n=1) features
        nearest_neighbor = neighbors.iloc[0]
        features['[REDACTED_BY_SCRIPT]'] = nearest_neighbor['threephasermsbreak']
        features['[REDACTED_BY_SCRIPT]'] = nearest_neighbor['earthfaultrmsbreak']
        
        # Substation Cluster (n=k) features
        features['[REDACTED_BY_SCRIPT]'] = neighbors['threephasermsbreak'].mean()
        features['avg_max_ef_fault_5nn_ka'] = neighbors['earthfaultrmsbreak'].mean()
        
        results.append(features)

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_grid_stability_features = pd.DataFrame(results).set_index('original_index')
    df_l7 = gdf_solar.merge(df_grid_stability_features, left_index=True, right_index=True, how='left')
    
    df_l7 = pd.DataFrame(df_l7.drop(columns='geometry'))
    df_l7.fillna(NULL_SENTINEL, inplace=True)

    df_l7.to_csv(L7_DATA_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")


if __name__ == '__main__':
    main()