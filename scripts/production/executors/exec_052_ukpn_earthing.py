import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
import os
import re
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input/Output Artifacts
FAULT_LEVEL_CSV_PATH = r"[REDACTED_BY_SCRIPT]"
DFES_GEOJSON_PATH = r"[REDACTED_BY_SCRIPT]"
UKPN_FAULT_PROFILE_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Architectural Hyperparameters
TARGET_CRS = "EPSG:27700"
K_NEIGHBORS = 5
NULL_SENTINEL = -1

def find_column_by_pattern(df: pd.DataFrame, pattern: str) -> str:
    """[REDACTED_BY_SCRIPT]"""
    matches = [col for col in df.columns if re.search(pattern, col)]
    if not matches:
        raise KeyError(f"[REDACTED_BY_SCRIPT]'{pattern}'.")
    if len(matches) > 1:
        raise ValueError(f"[REDACTED_BY_SCRIPT]'{pattern}': {matches}")
    return matches[0]

def get_or_create_l1_fault_profile_artifact(fault_csv_path, dfes_geojson_path, cache_path):
    """
    Loads the geolocated and aggregated substation fault profile artifact from cache, or creates it.
    """
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    
    # 1. Ingest and Normalize Fault Data
    df_fault = pd.read_csv(fault_csv_path)
    # Strict normalization: remove spaces and non-alphanumeric chars to handle "[REDACTED_BY_SCRIPT]"
    df_fault.columns = [re.sub(r'[^a-z0-9]', '', col.lower()) for col in df_fault.columns]

    # 2. Ingest and Normalize DFES Location Data
    gdf_dfes_locations = gpd.read_file(dfes_geojson_path).to_crs(TARGET_CRS)
    gdf_dfes_locations.columns = [re.sub(r'[^a-z0-9]', '', col.lower()) for col in gdf_dfes_locations.columns]
    gdf_dfes_locations.dropna(subset=['geometry'], inplace=True)

    # 3. Harmonize Keys
    try:
        fault_key = find_column_by_pattern(df_fault, r'[REDACTED_BY_SCRIPT]')
        dfes_key = find_column_by_pattern(gdf_dfes_locations, r'[REDACTED_BY_SCRIPT]')
        canonical_key = 'sitefunctionalallocation'
        df_fault.rename(columns={fault_key: canonical_key}, inplace=True)
        gdf_dfes_locations.rename(columns={dfes_key: canonical_key}, inplace=True)
    except (KeyError, ValueError) as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        raise

    # 4. Geospatial Bridge
    location_lookup = gdf_dfes_locations[[canonical_key, 'geometry']].drop_duplicates(subset=canonical_key)
    gdf_fault_located = pd.merge(df_fault, location_lookup, on=canonical_key, how='inner')
    gdf_fault_located = gpd.GeoDataFrame(gdf_fault_located, geometry='geometry', crs=TARGET_CRS)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # 5. Aggregate by Maximum Constraint
    fault_cols_to_agg = ['threephasermsbreak', 'threephasepeakmake', 'earthfaultrmsbreak', 'earthfaultpeakmake']
    
    # Coerce to numeric, handling non-numeric contaminants
    for col in fault_cols_to_agg:
        if col in gdf_fault_located.columns:
            gdf_fault_located[col] = pd.to_numeric(gdf_fault_located[col], errors='coerce')

    # Ensure columns exist before trying to aggregate
    existing_fault_cols = [col for col in fault_cols_to_agg if col in gdf_fault_located.columns]
    
    substation_fault_profile = gdf_fault_located.groupby(canonical_key).agg(
        **{col: (col, 'max') for col in existing_fault_cols},
        geometry=('geometry', 'first')
    ).reset_index()
    
    substation_fault_profile = gpd.GeoDataFrame(substation_fault_profile, geometry='geometry', crs=TARGET_CRS)
    assert substation_fault_profile.geometry.notna().all(), "[REDACTED_BY_SCRIPT]"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    substation_fault_profile.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return substation_fault_profile

def execute(master_gdf):
    """
    Executor entry point for calculating UKPN grid stability (earthing) features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Critical CRS Check (Pattern 1 Defense)
    if master_gdf.crs is None:
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf.set_crs(TARGET_CRS, inplace=True)
    elif master_gdf.crs.to_string() != TARGET_CRS:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        master_gdf = master_gdf.to_crs(TARGET_CRS)

    try:
        substation_fault_profile = get_or_create_l1_fault_profile_artifact(
            FAULT_LEVEL_CSV_PATH, DFES_GEOJSON_PATH, UKPN_FAULT_PROFILE_CACHE_PATH
        )
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")
    substation_coords = np.array(list(zip(substation_fault_profile.geometry.x, substation_fault_profile.geometry.y)))
    substation_tree = cKDTree(substation_coords)

    results = []
    for index, solar_site in tqdm(master_gdf.iterrows(), total=master_gdf.shape[0], desc="[REDACTED_BY_SCRIPT]"):
        target_coord = [solar_site.geometry.x, solar_site.geometry.y]
        distances_m, indices = substation_tree.query(target_coord, k=K_NEIGHBORS)
        neighbors = substation_fault_profile.iloc[indices]
        
        features = {'hex_id': solar_site['hex_id']}
        
        # Nearest Substation (n=1) features
        nearest_neighbor = neighbors.iloc[0]
        features['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get('threephasermsbreak', NULL_SENTINEL)
        features['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get('earthfaultrmsbreak', NULL_SENTINEL)
        
        # Substation Cluster (n=k) features
        features[f'[REDACTED_BY_SCRIPT]'] = neighbors['threephasermsbreak'].mean()
        features[f'[REDACTED_BY_SCRIPT]'] = neighbors['earthfaultrmsbreak'].mean()
        
        results.append(features)

    logging.info("[REDACTED_BY_SCRIPT]")
    df_stability_features = pd.DataFrame(results)
    
    master_gdf = master_gdf.merge(df_stability_features, on='hex_id', how='left')
    
    new_cols = [col for col in df_stability_features.columns if col != 'hex_id']
    master_gdf[new_cols] = master_gdf[new_cols].fillna(NULL_SENTINEL)

    # Artifact Validation: Ensure output remains a valid GeoDataFrame
    if not isinstance(master_gdf, gpd.GeoDataFrame):
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf = gpd.GeoDataFrame(master_gdf, geometry='geometry', crs=TARGET_CRS)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf
