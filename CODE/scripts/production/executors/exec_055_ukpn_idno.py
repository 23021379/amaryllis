import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys
from shapely.geometry import Point, LineString
from tqdm import tqdm

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input files
IDNO_RAW_PATH = r"[REDACTED_BY_SCRIPT]"
LPA_BOUNDARIES_PATH = r"[REDACTED_BY_SCRIPT]"
SUBSTATION_CAPACITY_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Artifact paths
IDNO_L1_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"
LPA_IDNO_FEATURES_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = 'EPSG:27700'

# Analysis parameters
NULL_SENTINEL = 0

def clean_col_names(df, prefix=None):
    """[REDACTED_BY_SCRIPT]"""
    new_cols = []
    for col in df.columns:
        new_col = col.lower().strip().replace(' ', '_')
        new_col = ''.join(c if c.isalnum() else '_' for c in new_col)
        while '__' in new_col: new_col = new_col.replace('__', '_')
        new_col = new_col.strip('_')
        if prefix:
            new_cols.append(f"{prefix}_{new_col}")
        else:
            new_cols.append(new_col)
    df.columns = new_cols
    return df

def get_or_create_l1_idno_artifact(raw_path, cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_idno_raw = gpd.read_file(raw_path)
    
    if gdf_idno_raw.crs.to_string() != TARGET_CRS:
        gdf_idno_unified = gdf_idno_raw.to_crs(TARGET_CRS)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    else:
        gdf_idno_unified = gdf_idno_raw

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_idno_unified.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_idno_unified

def get_or_create_lpa_idno_features_artifact(lpa_path, idno_gdf, cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return pd.read_parquet(cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_lpa = gpd.read_file(lpa_path).to_crs(TARGET_CRS)
    gdf_lpa = clean_col_names(gdf_lpa)
    lpa_key = 'lpa23nm'

    # Feature: lpa_idno_area_as_percent_of_total_area
    lpa_idno_intersection = gpd.overlay(gdf_lpa, idno_gdf, how='intersection')
    lpa_idno_area = lpa_idno_intersection.groupby(lpa_key)['geometry'].apply(lambda g: g.area.sum())
    
    gdf_lpa['total_area'] = gdf_lpa.area
    gdf_lpa = gdf_lpa.set_index(lpa_key)
    gdf_lpa['[REDACTED_BY_SCRIPT]'] = (lpa_idno_area / gdf_lpa['total_area']) * 100

    # Feature: lpa_idno_count_per_100km2
    idno_in_lpa = gpd.sjoin(idno_gdf, gdf_lpa, how='inner', predicate='within')
    lpa_idno_count = idno_in_lpa.groupby(lpa_key).size()
    gdf_lpa['ukpn_lpa_idno_count'] = lpa_idno_count
    gdf_lpa['[REDACTED_BY_SCRIPT]'] = gdf_lpa['ukpn_lpa_idno_count'] / (gdf_lpa['total_area'] / 1e8) # 100km2 = 1e8 m2

    lpa_features = gdf_lpa.reset_index()[[lpa_key, '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]
    lpa_features.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return lpa_features

def execute(master_gdf):
    """
    Executor entry point for integrating UKPN Independent Distribution Network Operator (IDNO) features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    try:
        gdf_idno = get_or_create_l1_idno_artifact(IDNO_RAW_PATH, IDNO_L1_CACHE_PATH)
        lpa_features = get_or_create_lpa_idno_features_artifact(LPA_BOUNDARIES_PATH, gdf_idno, LPA_IDNO_FEATURES_CACHE_PATH)
        gdf_substations = gpd.read_parquet(SUBSTATION_CAPACITY_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    # --- Phase 2: Core IDNO Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    within_join = gpd.sjoin(master_gdf, gdf_idno, how='left', predicate='within')
    master_gdf['ukpn_idno_is_within'] = (~within_join.loc[~within_join.index.duplicated(keep='first'), 'index_right'].isnull()).astype(int)

    sites_outside = master_gdf[master_gdf['ukpn_idno_is_within'] == 0]
    if not sites_outside.empty:
        distances = sites_outside.geometry.apply(lambda g: gdf_idno.distance(g).min())
        master_gdf['[REDACTED_BY_SCRIPT]'] = distances
    else:
        master_gdf['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
    master_gdf['[REDACTED_BY_SCRIPT]'].fillna(NULL_SENTINEL, inplace=True)

    # --- Phase 3: Grid Interaction Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    substation_sindex = gdf_substations.sindex
    nearest_indices = [list(substation_sindex.nearest(g, return_all=False))[1][0] for g in master_gdf.geometry]
    nearest_substations = gdf_substations.iloc[nearest_indices].reset_index(drop=True)
    
    sub_within_join = gpd.sjoin(nearest_substations, gdf_idno, how='left', predicate='within')
    master_gdf['[REDACTED_BY_SCRIPT]'] = (~sub_within_join.loc[~sub_within_join.index.duplicated(keep='first'), 'index_right'].isnull()).astype(int)

    connection_paths = [LineString([solar_geom, sub_geom]) for solar_geom, sub_geom in zip(master_gdf.geometry, nearest_substations.geometry)]
    gdf_paths = gpd.GeoDataFrame(geometry=connection_paths, crs=TARGET_CRS, index=master_gdf.index)
    path_crosses_join = gpd.sjoin(gdf_paths, gdf_idno, how='inner', predicate='intersects')
    crossing_path_indices = path_crosses_join.index.unique()
    master_gdf['[REDACTED_BY_SCRIPT]'] = master_gdf.index.isin(crossing_path_indices).astype(int)

    # --- Final Integration: LPA Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_lpa_boundaries = gpd.read_file(LPA_BOUNDARIES_PATH).to_crs(TARGET_CRS)
    gdf_lpa_boundaries = clean_col_names(gdf_lpa_boundaries)
    
    master_with_lpa_key = gpd.sjoin(master_gdf, gdf_lpa_boundaries[['lpa23nm', 'geometry']], how='left', predicate='within')
    master_with_lpa_key = master_with_lpa_key.drop_duplicates(subset=['master_grid_id'] if 'master_grid_id' in master_with_lpa_key.columns else master_with_lpa_key.index.name, keep='first')

    final_gdf = pd.merge(master_with_lpa_key, lpa_features, on='lpa23nm', how='left')
    final_gdf.drop(columns=['index_right', 'lpa23nm'], inplace=True, errors='ignore')
    
    # Fill NaNs for the newly merged columns
    new_feature_cols = lpa_features.columns.drop('lpa23nm')
    final_gdf[new_feature_cols] = final_gdf[new_feature_cols].fillna(NULL_SENTINEL)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf
