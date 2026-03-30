import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys
from sklearn.neighbors import BallTree
from tqdm import tqdm

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input file
PQ_INPUT = r"[REDACTED_BY_SCRIPT]"

# Artifact path
PQ_L1_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = 'EPSG:27700'

# Analysis parameters
K_NEIGHBORS = 5
NULL_SENTINEL = 0

def get_or_create_l1_pq_artifact(filepath, cache_path):
    """
    Loads, sanitizes, and aggregates the long-form PQ data into a stable L1 artifact, caching the result.
    This function replicates the "anti-pivot" logic from the original script.
    """
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_pq = gpd.read_file(filepath).to_crs(TARGET_CRS)

    gdf_pq.columns = [col.lower().strip().replace(' ', '_') for col in gdf_pq.columns]
    gdf_pq['highest'] = pd.to_numeric(gdf_pq['highest'], errors='coerce')
    gdf_pq.dropna(subset=['highest', 'sitefunctionallocation'], inplace=True)

    gdf_pq['harmonic_order'] = gdf_pq['harmonic'].replace('THD', '-1').str.replace('H', '').astype(int)

    def aggregate_site(group):
        pivot = group.pivot_table(index='sitefunctionallocation', columns='harmonic_order', values='highest', aggfunc='first')
        
        res = {
            'ukpn_pq_thd_highest': pivot.get(-1, pd.Series(0)).iloc[0],
            'ukpn_pq_h5_highest': pivot.get(5, pd.Series(0)).iloc[0],
        }
        
        odd_harmonics = group[group['harmonic_order'] > 0 & (group['harmonic_order'] % 2 != 0)]
        res['[REDACTED_BY_SCRIPT]'] = odd_harmonics['highest'].mean() if not odd_harmonics.empty else 0
        
        return pd.Series(res)

    logging.info("[REDACTED_BY_SCRIPT]")
    aggregated_data = gdf_pq.groupby('sitefunctionallocation').apply(aggregate_site)
    
    sites_geometry = gdf_pq[['sitefunctionallocation', 'geometry']].drop_duplicates(subset='sitefunctionallocation').set_index('sitefunctionallocation')
    gdf_pq_l1 = sites_geometry.join(aggregated_data).reset_index()
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_pq_l1.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_pq_l1

def calculate_spatial_pq_features(master_gdf, gdf_pq_l1, pq_tree):
    """
    Generates spatially interpolated PQ features for each site using IDW k-NN.
    """
    if gdf_pq_l1.empty:
        logging.warning("[REDACTED_BY_SCRIPT]")
        return pd.DataFrame()

    app_coords = np.array(list(zip(master_gdf.geometry.x, master_gdf.geometry.y)))
    distances, indices = pq_tree.query(app_coords, k=min(K_NEIGHBORS, len(gdf_pq_l1)))

    results = []
    for i in tqdm(range(len(master_gdf)), desc="[REDACTED_BY_SCRIPT]", leave=False):
        cohort_indices = indices[i]
        cohort_distances = distances[i]
        
        cohort_distances[cohort_distances == 0] = 1e-6
        
        cohort = gdf_pq_l1.iloc[cohort_indices]
        
        weights = 1 / (cohort_distances ** 2)
        
        res = {
            'ukpn_pq_idw_thd_knn5': np.average(cohort['ukpn_pq_thd_highest'], weights=weights),
            '[REDACTED_BY_SCRIPT]': np.average(cohort['ukpn_pq_h5_highest'], weights=weights),
            '[REDACTED_BY_SCRIPT]': np.average(cohort['[REDACTED_BY_SCRIPT]'], weights=weights),
            '[REDACTED_BY_SCRIPT]': cohort_distances[0],
            'ukpn_pq_max_thd_in_knn5': cohort['ukpn_pq_thd_highest'].max(),
            'ukpn_pq_std_thd_in_knn5': cohort['ukpn_pq_thd_highest'].std()
        }
        results.append(res)
        
    return pd.DataFrame(results, index=master_gdf.index)

def execute(master_gdf):
    """
    Executor entry point for integrating UKPN Power Quality (PQ) features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # V14 ID Management: Ensure hex_id is the index for joins.
    id_col = 'hex_id'
    if master_gdf.index.name != id_col:
        if id_col in master_gdf.columns:
            master_gdf.set_index(id_col, inplace=True)
            logging.info(f"Set '{id_col}' as index.")
        else:
            logging.error(f"FATAL: '{id_col}'[REDACTED_BY_SCRIPT]")
            return master_gdf

    try:
        gdf_pq_l1 = get_or_create_l1_pq_artifact(PQ_INPUT, PQ_L1_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    # If the artifact is empty, there's nothing to calculate.
    if gdf_pq_l1.empty:
        logging.warning("[REDACTED_BY_SCRIPT]")
        # Define expected columns to add with null values to maintain schema consistency
        expected_cols = [
            'ukpn_pq_idw_thd_knn5', '[REDACTED_BY_SCRIPT]', 
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            'ukpn_pq_max_thd_in_knn5', 'ukpn_pq_std_thd_in_knn5'
        ]
        for col in expected_cols:
            master_gdf[col] = NULL_SENTINEL
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")
    pq_coords = np.array(list(zip(gdf_pq_l1.geometry.x, gdf_pq_l1.geometry.y)))
    pq_tree = BallTree(pq_coords, metric='euclidean')

    logging.info(f"[REDACTED_BY_SCRIPT]")
    # The calculation function correctly uses the master_gdf's index.
    pq_features = calculate_spatial_pq_features(master_gdf, gdf_pq_l1, pq_tree)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    # The join is on the index, which is hex_id, so this is correct.
    final_gdf = master_gdf.join(pq_features)
    
    if not pq_features.empty:
        final_gdf[pq_features.columns] = final_gdf[pq_features.columns].fillna(NULL_SENTINEL)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf
