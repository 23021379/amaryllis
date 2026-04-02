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
L5_DATA_PATH = '[REDACTED_BY_SCRIPT]'
DFES_GEOJSON_PATH = r"[REDACTED_BY_SCRIPT]"
L6_DATA_PATH = '[REDACTED_BY_SCRIPT]'
SUBSTATION_L1_ARTIFACT_PATH = '[REDACTED_BY_SCRIPT]' # Mandated L1 artifact

# Architectural Hyperparameters
TARGET_CRS = "EPSG:27700"
K_NEIGHBORS = 5
TARGET_SCENARIO = 'Holistic Transition'
NULL_SENTINEL = -1

def main():
    """
    Executes the revised Directive 008. Ingests DFES grid capacity data from GeoJSON,
    performs a spatio-temporally aware KNN query, and enriches the L5 solar dataset
    with features describing the technical feasibility of grid connection.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    try:
        df_l5 = pd.read_csv(L5_DATA_PATH)
        gdf_solar = gpd.GeoDataFrame(
            df_l5, geometry=gpd.points_from_xy(df_l5.easting, df_l5.northing), crs=TARGET_CRS
        )
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- MANDATE 8.1: Ingest GeoJSON and Unify CRS ---
    try:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf_dfes = gpd.read_file(DFES_GEOJSON_PATH)
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf_dfes = gdf_dfes.to_crs(TARGET_CRS) # MISSION-CRITICAL STEP
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- MANDATE 8.2: Restructure via Pivoting (with confirmed column names) ---
    logging.info(f"[REDACTED_BY_SCRIPT]'{TARGET_SCENARIO}'...")
    dfes_filtered = gdf_dfes[gdf_dfes['scenario'] == TARGET_SCENARIO].copy()

    # The GeoJSON properties are already clean; we directly use the confirmed keys.
    substation_profile = dfes_filtered.pivot_table(
        index=['sitefunctionallocation', 'substation_name', 'voltage_kv', 'geometry'], # REVISED: Matched to GeoJSON keys
        columns='year', # REVISED
        values='headroom_mw' # REVISED
    ).reset_index()

    # Flatten column names after pivot
    substation_profile.columns = [f'headroom_mw_{col}' if str(col).isdigit() else col for col in substation_profile.columns]
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- MANDATE 8.6: EXPLICIT GEOSPATIAL STATE RESTORATION ---
    logging.info("[REDACTED_BY_SCRIPT]")
    substation_profile = gpd.GeoDataFrame(
        substation_profile, 
        geometry='geometry', 
        crs=TARGET_CRS  # Re-assert the CRS
    )

    # --- NEW MANDATE: Persist Authoritative Substation L1 Artifact ---
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        # Isolate core identifying and geospatial columns, excluding transient headroom data.
        authoritative_substations = substation_profile[[
            'sitefunctionallocation', 'substation_name', 'voltage_kv', 'geometry'
        ]].copy()

        # Harmonize schema to the project-wide standard for this entity.
        authoritative_substations.rename(columns={
            'sitefunctionallocation': 'substation_id',
            'substation_name': 'name',
            'voltage_kv': 'voltage'
        }, inplace=True)

        # Persist the artifact as a GeoPackage for maximum fidelity.
        authoritative_substations.to_file(SUBSTATION_L1_ARTIFACT_PATH, driver='GPKG')
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        # This constitutes a pipeline failure in a production environment.

    # --- MANDATE 8.3: Temporally-Aware Feature Selection Logic (Unchanged) ---
    forecast_years = sorted([int(str(col).split('_')[-1]) for col in substation_profile.columns if 'headroom_mw_' in str(col)])
    
    def get_relevant_headroom_col(submission_year):
        for forecast_year in forecast_years:
            if submission_year < forecast_year:
                return f'[REDACTED_BY_SCRIPT]'
        return f'[REDACTED_BY_SCRIPT]' # Fallback to latest forecast

    # Build spatial index on the now-guaranteed GeoDataFrame
    logging.info("[REDACTED_BY_SCRIPT]")
    substation_coords = np.array(list(zip(substation_profile.geometry.x, substation_profile.geometry.y)))
    substation_tree = cKDTree(substation_coords)

    # --- MANDATE 8.4: Synthesis of KNN Grid Capacity Features ---
    results = []
    for index, solar_site in tqdm(gdf_solar.iterrows(), total=gdf_solar.shape[0], desc="[REDACTED_BY_SCRIPT]"):
        target_coord = [solar_site.geometry.x, solar_site.geometry.y]
        distances_m, indices = substation_tree.query(target_coord, k=K_NEIGHBORS)
        neighbors = substation_profile.iloc[indices]
        
        relevant_headroom_col = get_relevant_headroom_col(solar_site['submission_year'])
        
        features = {'original_index': index}
        
        # Nearest Substation (n=1) features
        nearest_neighbor = neighbors.iloc[0]
        features['[REDACTED_BY_SCRIPT]'] = distances_m[0] / 1000.0
        features['[REDACTED_BY_SCRIPT]'] = nearest_neighbor['voltage_kv'] # REVISED
        features['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get(relevant_headroom_col, NULL_SENTINEL)
        
        # Substation Cluster (n=k) features
        features['[REDACTED_BY_SCRIPT]'] = np.mean(distances_m) / 1000.0
        if relevant_headroom_col in neighbors.columns:
            headroom_values = neighbors[relevant_headroom_col].dropna()
            if not headroom_values.empty:
                features['[REDACTED_BY_SCRIPT]'] = headroom_values.mean()
                features['[REDACTED_BY_SCRIPT]'] = headroom_values.std() if len(headroom_values) > 1 else 0
            else:
                features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
                features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        
        results.append(features)

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_grid_features = pd.DataFrame(results).set_index('original_index')
    df_l6 = gdf_solar.merge(df_grid_features, left_index=True, right_index=True, how='left')
    
    df_l6 = pd.DataFrame(df_l6.drop(columns='geometry'))
    df_l6.fillna(NULL_SENTINEL, inplace=True)

    df_l6.to_csv(L6_DATA_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()
