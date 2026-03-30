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

# Input files
DNOA_LV_INPUT = r"[REDACTED_BY_SCRIPT]"
LCT_L1_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Artifact paths
DNOA_L1_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"
PRIMARY_SUB_DNOA_L2_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = 'EPSG:27700'

# Analysis parameters
K_NEIGHBORS = 20
TEMPORAL_WINDOW_YEARS = 3
NULL_SENTINEL = 0

def find_column_by_pattern(df, pattern):
    """[REDACTED_BY_SCRIPT]"""
    for col in df.columns:
        if pattern in col:
            return col
    return None

def get_or_create_l1_dnoa_artifact(filepath, cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_dnoa = gpd.read_file(filepath).to_crs(TARGET_CRS)
    
    gdf_dnoa.columns = [col.lower().strip().replace(' ', '_') for col in gdf_dnoa.columns]
    
    numeric_cols = ['constraint_year', '[REDACTED_BY_SCRIPT]']
    for col in numeric_cols:
        gdf_dnoa[col] = pd.to_numeric(gdf_dnoa[col], errors='coerce')
    
    gdf_dnoa.dropna(subset=numeric_cols, inplace=True)
    gdf_dnoa['constraint_year'] = gdf_dnoa['constraint_year'].astype(int)

    gdf_dnoa['dnoa_result'] = gdf_dnoa['dnoa_result'].str.strip()
    dnoa_dummies = pd.get_dummies(gdf_dnoa['dnoa_result'], prefix='dnoa_result_is', dtype=int)
    gdf_dnoa = pd.concat([gdf_dnoa, dnoa_dummies], axis=1)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_dnoa.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_dnoa

def get_or_create_l2_primary_sub_dnoa_features(gdf_dnoa, lct_cache_path, dnoa_cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(dnoa_cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return pd.read_parquet(dnoa_cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_lct = gpd.read_parquet(lct_cache_path)
    gdf_primary_subs = gdf_lct[['[REDACTED_BY_SCRIPT]', 'geometry']].drop_duplicates(subset='[REDACTED_BY_SCRIPT]')

    gdf_dnoa_linked = gpd.sjoin_nearest(gdf_dnoa, gdf_primary_subs, how='left')

    primary_summary = gdf_dnoa_linked.groupby('[REDACTED_BY_SCRIPT]').agg(
        ukpn_psub_downstream_constrained_tx_count=('geometry', 'count'),
        ukpn_psub_downstream_total_deferred_kva=('[REDACTED_BY_SCRIPT]', 'sum'),
        ukpn_psub_downstream_earliest_constraint_year=('constraint_year', 'min')
    ).reset_index()
    
    primary_summary.to_parquet(dnoa_cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return primary_summary

def calculate_knn_features(site_row, gdf_dnoa, dnoa_tree):
    """[REDACTED_BY_SCRIPT]"""
    feature_names = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'ukpn_dnoa_avg_deferred_kva_knn', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    features = {name: NULL_SENTINEL for name in feature_names}

    submission_year = site_row['submission_year']
    if pd.isna(submission_year):
        return pd.Series(features)

    # Temporal Guard: Filter DNOA data for the relevant future window
    dnoa_temporally_relevant_indices = gdf_dnoa.index[
        (gdf_dnoa['constraint_year'] >= submission_year) &
        (gdf_dnoa['constraint_year'] <= submission_year + TEMPORAL_WINDOW_YEARS)
    ]
    if dnoa_temporally_relevant_indices.empty:
        return pd.Series(features)

    # Spatial Query: Find k-NN in the full dataset first
    site_coords = [[site_row.geometry.x, site_row.geometry.y]]
    distances, indices = dnoa_tree.query(site_coords, k=min(K_NEIGHBORS, len(gdf_dnoa)))
    
    # Intersect spatial and temporal results to get the true cohort
    potential_spatial_indices = gdf_dnoa.index[indices[0]]
    final_cohort_indices = potential_spatial_indices.intersection(dnoa_temporally_relevant_indices)

    if final_cohort_indices.empty:
        return pd.Series(features)
        
    knn_cohort = gdf_dnoa.loc[final_cohort_indices].copy()
    
    # Recalculate distances for the actual cohort
    cohort_coords = np.array(list(zip(knn_cohort.geometry.x, knn_cohort.geometry.y)))
    distances_to_cohort = np.linalg.norm(cohort_coords - site_coords, axis=1)
    knn_cohort['distance_m'] = distances_to_cohort
    knn_cohort = knn_cohort.sort_values('distance_m')

    # Feature Synthesis
    nearest = knn_cohort.iloc[0]
    features['[REDACTED_BY_SCRIPT]'] = nearest['distance_m']
    features['[REDACTED_BY_SCRIPT]'] = nearest['constraint_year'] - submission_year
    features['[REDACTED_BY_SCRIPT]'] = len(knn_cohort)
    features['[REDACTED_BY_SCRIPT]'] = knn_cohort['distance_m'].mean()
    features['ukpn_dnoa_avg_deferred_kva_knn'] = knn_cohort['[REDACTED_BY_SCRIPT]'].mean()
    features['[REDACTED_BY_SCRIPT]'] = knn_cohort['constraint_year'].min()
    
    flex_col = '[REDACTED_BY_SCRIPT]'
    if flex_col in knn_cohort:
        features['[REDACTED_BY_SCRIPT]'] = knn_cohort[flex_col].mean()

    return pd.Series(features)

def execute(master_gdf):
    """
    Executor entry point for integrating UKPN DNOA LV constraint features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Critical CRS Check (Pattern 1 Defense)
    if master_gdf.crs is None:
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf.set_crs(TARGET_CRS, inplace=True)
    elif master_gdf.crs.to_string() != TARGET_CRS:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        master_gdf = master_gdf.to_crs(TARGET_CRS)

    # V14 ID Management
    id_col = 'hex_id'
    if master_gdf.index.name != id_col:
        if id_col in master_gdf.columns:
            master_gdf.set_index(id_col, inplace=True)
        else:
            logging.error(f"FATAL: '{id_col}'[REDACTED_BY_SCRIPT]")
            return master_gdf
    
    master_gdf_processed = master_gdf.reset_index()

    # Prerequisite checks
    if 'submission_date' not in master_gdf_processed.columns:
        logging.error("FATAL: 'submission_date'[REDACTED_BY_SCRIPT]")
        return master_gdf
    
    primary_sub_col = find_column_by_pattern(master_gdf_processed, '[REDACTED_BY_SCRIPT]')
    if not primary_sub_col:
        primary_sub_col = find_column_by_pattern(master_gdf_processed, '[REDACTED_BY_SCRIPT]')

    if not primary_sub_col:
        logging.error("FATAL: 'primary_functionallocation'[REDACTED_BY_SCRIPT]")
        return master_gdf
        
    master_gdf_processed['submission_date'] = pd.to_datetime(master_gdf_processed['submission_date'], errors='coerce')
    master_gdf_processed['submission_year'] = master_gdf_processed['submission_date'].dt.year

    try:
        gdf_dnoa = get_or_create_l1_dnoa_artifact(DNOA_LV_INPUT, DNOA_L1_CACHE_PATH)
        primary_sub_features = get_or_create_l2_primary_sub_dnoa_features(gdf_dnoa, LCT_L1_CACHE_PATH, PRIMARY_SUB_DNOA_L2_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    # --- k-NN Feature Calculation ---
    logging.info("[REDACTED_BY_SCRIPT]")
    dnoa_coords = np.array(list(zip(gdf_dnoa.geometry.x, gdf_dnoa.geometry.y)))
    dnoa_tree = BallTree(dnoa_coords, metric='euclidean')

    logging.info(f"[REDACTED_BY_SCRIPT]")
    tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
    knn_features = master_gdf_processed.progress_apply(
        calculate_knn_features, 
        axis=1, 
        gdf_dnoa=gdf_dnoa, 
        dnoa_tree=dnoa_tree
    )

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # V14 FIX: Join features back to the processed (non-indexed) dataframe first
    final_gdf_processed = master_gdf_processed.join(knn_features)
    final_gdf_processed = final_gdf_processed.merge(primary_sub_features, on=primary_sub_col, how='left')
    
    # Clean up and fill NaNs
    new_cols = list(knn_features.columns) + list(primary_sub_features.columns.drop(primary_sub_col, errors='ignore'))
    final_gdf_processed[new_cols] = final_gdf_processed[new_cols].fillna(NULL_SENTINEL)
    final_gdf_processed.drop(columns=['submission_year'], inplace=True, errors='ignore')

    # Restore index
    if id_col in final_gdf_processed.columns:
        final_gdf_processed.set_index(id_col, inplace=True)

    # Ensure columns from original master_gdf are not lost
    for col in master_gdf.columns:
        if col not in final_gdf_processed.columns:
            final_gdf_processed[col] = master_gdf[col]

    # Artifact Validation: Ensure output remains a valid GeoDataFrame
    if not isinstance(final_gdf_processed, gpd.GeoDataFrame):
        logging.warning("[REDACTED_BY_SCRIPT]")
        final_gdf_processed = gpd.GeoDataFrame(final_gdf_processed, geometry='geometry', crs=TARGET_CRS)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf_processed
