import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input files
TRANSFORMERS_PATH = r"[REDACTED_BY_SCRIPT]"
TRANSFORMERS_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = 'EPSG:27700'

# Analysis parameters
K_NEIGHBORS = 5
RADIUS_METERS = 10000
NULL_SENTINEL = 0

def clean_col_names(df, prefix):
    """[REDACTED_BY_SCRIPT]"""
    new_cols = []
    for col in df.columns:
        new_col = col.lower().strip().replace(' ', '_')
        new_col = ''.join(c if c.isalnum() else '_' for c in new_col)
        while '__' in new_col: new_col = new_col.replace('__', '_')
        new_col = new_col.strip('_')
        new_cols.append(f"{prefix}_{new_col}")
    df.columns = new_cols
    return df

def find_column_by_pattern(df, pattern):
    """[REDACTED_BY_SCRIPT]"""
    for col in df.columns:
        if pattern in col:
            return col
    raise KeyError(f"Pattern '{pattern}'[REDACTED_BY_SCRIPT]")

def get_or_create_l1_transformer_artifact(transformers_path, cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_transformers = gpd.read_file(transformers_path)
    if gdf_transformers.crs != TARGET_CRS:
        gdf_transformers = gdf_transformers.to_crs(TARGET_CRS)
    
    gdf_transformers = clean_col_names(gdf_transformers, 'ukpn_tx')
    initial_count = len(gdf_transformers)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    gdf_transformers['ukpn_tx_onanrating_kva'] = pd.to_numeric(gdf_transformers['ukpn_tx_onanrating_kva'], errors='coerce')
    gdf_transformers.dropna(subset=['ukpn_tx_onanrating_kva'], inplace=True)
    valid_kva_count = len(gdf_transformers)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    substation_key = find_column_by_pattern(gdf_transformers, 'sitefunctional')
    geometry_col_name = find_column_by_pattern(gdf_transformers, 'geometry')
    
    agg_data = gdf_transformers.groupby(substation_key).agg(
        total_onan_rating_kva=('ukpn_tx_onanrating_kva', 'sum'),
        geometry=(geometry_col_name, 'first')
    ).reset_index()

    gdf_substation_capacity = gpd.GeoDataFrame(agg_data, geometry='geometry', crs=TARGET_CRS)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_substation_capacity.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return gdf_substation_capacity

def calculate_transformer_features(site_geom, tree, substation_data):
    """
    For a single solar site, calculates KNN and radius-based transformer capacity features.
    """
    features = {}
    site_coords = (site_geom.x, site_geom.y)
    
    # KNN Query (k=5)
    distances, indices = tree.query(site_coords, k=K_NEIGHBORS, distance_upper_bound=np.inf)
    
    valid_indices = indices[np.isfinite(distances)]
    
    if len(valid_indices) > 0:
        nearest_substations = substation_data.iloc[valid_indices]
        features['[REDACTED_BY_SCRIPT]'] = nearest_substations.iloc[0]['total_onan_rating_kva']
        features['ukpn_tx_avg_total_kva_5nn'] = nearest_substations['total_onan_rating_kva'].mean()
    else:
        features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        features['ukpn_tx_avg_total_kva_5nn'] = NULL_SENTINEL

    # Radius Query (10km)
    radius_indices = tree.query_ball_point(site_coords, r=RADIUS_METERS)
    
    if len(radius_indices) > 0:
        substations_in_radius = substation_data.iloc[radius_indices]
        features['[REDACTED_BY_SCRIPT]'] = substations_in_radius['total_onan_rating_kva'].sum()
    else:
        features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        
    return pd.Series(features)

def execute(master_gdf):
    """
    Executor entry point for integrating UKPN grid transformer capacity metrics.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    try:
        gdf_substation_capacity = get_or_create_l1_transformer_artifact(TRANSFORMERS_PATH, TRANSFORMERS_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    if gdf_substation_capacity.empty:
        logging.warning("[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")
    substation_coords = np.array(list(zip(gdf_substation_capacity.geometry.x, gdf_substation_capacity.geometry.y)))
    tree = cKDTree(substation_coords)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
    transformer_features = master_gdf['geometry'].progress_apply(
        calculate_transformer_features, 
        tree=tree, 
        substation_data=gdf_substation_capacity
    )

    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf_with_features = master_gdf.join(transformer_features)
    
    new_cols = transformer_features.columns
    master_gdf_with_features[new_cols] = master_gdf_with_features[new_cols].fillna(NULL_SENTINEL)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf_with_features
