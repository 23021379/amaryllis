import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
import os
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input/Output Artifacts
DFES_GEOJSON_PATH = r"[REDACTED_BY_SCRIPT]"
UKPN_SUBSTATION_L1_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Architectural Hyperparameters
TARGET_CRS = "EPSG:27700"
K_NEIGHBORS = 5
TARGET_SCENARIO = 'Holistic Transition'
NULL_SENTINEL = -1

def get_or_create_l1_ukpn_substation_artifact(geojson_path, cache_path):
    """
    Loads the L1 UKPN substation artifact from cache if it exists, otherwise creates it.
    This artifact contains the pivoted, geospatially-aware substation profiles.
    """
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_dfes = gpd.read_file(geojson_path).to_crs(TARGET_CRS)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        raise

    dfes_filtered = gdf_dfes[gdf_dfes['scenario'] == TARGET_SCENARIO].copy()

    # Pivot on stable identifiers only, avoiding geometry object keys
    pivot_idx = ['sitefunctionallocation', 'substation_name', 'voltage_kv']
    substation_profile = dfes_filtered.pivot_table(
        index=pivot_idx,
        columns='year',
        values='headroom_mw'
    ).reset_index()

    # Re-attach geometry securely by merging back on identifiers
    geometries = dfes_filtered[pivot_idx + ['geometry']].drop_duplicates(subset=pivot_idx)
    substation_profile = substation_profile.merge(geometries, on=pivot_idx, how='left')

    substation_profile.columns = [f'headroom_mw_{col}' if str(col).isdigit() else col for col in substation_profile.columns]
    
    substation_profile = gpd.GeoDataFrame(substation_profile, geometry='geometry', crs=TARGET_CRS)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    substation_profile.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return substation_profile

def execute(master_gdf):
    """
    Executor entry point for calculating UKPN DFES substation features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Critical CRS Check (Pattern 1 Defense)
    if master_gdf.crs is None:
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf.set_crs(TARGET_CRS, inplace=True)
    elif master_gdf.crs.to_string() != TARGET_CRS:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        master_gdf = master_gdf.to_crs(TARGET_CRS)

    if 'submission_year' not in master_gdf.columns:
        logging.error("FATAL: 'submission_year'[REDACTED_BY_SCRIPT]")
        return master_gdf

    try:
        substation_profile = get_or_create_l1_ukpn_substation_artifact(DFES_GEOJSON_PATH, UKPN_SUBSTATION_L1_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    forecast_years = sorted([int(str(col).split('_')[-1]) for col in substation_profile.columns if 'headroom_mw_' in str(col)])
    
    def get_relevant_headroom_col(submission_year):
        if pd.isna(submission_year):
            return f'[REDACTED_BY_SCRIPT]' # Fallback for safety
        for forecast_year in forecast_years:
            if submission_year < forecast_year:
                return f'[REDACTED_BY_SCRIPT]'
        return f'[REDACTED_BY_SCRIPT]'

    logging.info("[REDACTED_BY_SCRIPT]")
    substation_coords = np.array(list(zip(substation_profile.geometry.x, substation_profile.geometry.y)))
    substation_tree = cKDTree(substation_coords)

    results = []
    for index, solar_site in tqdm(master_gdf.iterrows(), total=master_gdf.shape[0], desc="[REDACTED_BY_SCRIPT]"):
        target_coord = [solar_site.geometry.x, solar_site.geometry.y]
        distances_m, indices = substation_tree.query(target_coord, k=K_NEIGHBORS)
        neighbors = substation_profile.iloc[indices]
        
        relevant_headroom_col = get_relevant_headroom_col(solar_site['submission_year'])
        
        features = {'hex_id': solar_site['hex_id']}
        
        # Nearest Substation (n=1) features
        nearest_neighbor = neighbors.iloc[0]
        features['[REDACTED_BY_SCRIPT]'] = distances_m[0] / 1000.0
        features['[REDACTED_BY_SCRIPT]'] = nearest_neighbor['voltage_kv']
        features['[REDACTED_BY_SCRIPT]'] = nearest_neighbor.get(relevant_headroom_col, NULL_SENTINEL)
        
        # Substation Cluster (n=k) features
        features[f'[REDACTED_BY_SCRIPT]'] = np.mean(distances_m) / 1000.0
        if relevant_headroom_col in neighbors.columns:
            headroom_values = neighbors[relevant_headroom_col].dropna()
            if not headroom_values.empty:
                features[f'[REDACTED_BY_SCRIPT]'] = headroom_values.mean()
                features[f'[REDACTED_BY_SCRIPT]'] = headroom_values.std() if len(headroom_values) > 1 else 0
            else:
                features[f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
                features[f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        
        results.append(features)

    logging.info("[REDACTED_BY_SCRIPT]")
    df_grid_features = pd.DataFrame(results)
    
    master_gdf = master_gdf.merge(df_grid_features, on='hex_id', how='left')
    
    new_cols = [col for col in df_grid_features.columns if col != 'hex_id']
    master_gdf[new_cols] = master_gdf[new_cols].fillna(NULL_SENTINEL)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf
