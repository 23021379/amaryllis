import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys
from tqdm import tqdm

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input files
LCT_DATA_INPUT = r"[REDACTED_BY_SCRIPT]' Secondary Sites.geojson"

# Artifact paths
LCT_L1_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"
PRIMARY_SUB_FEATURES_L2_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = 'EPSG:27700'

# Analysis parameters
BUFFER_RADIUS_METERS = 5000
NULL_SENTINEL = 0

def get_or_create_l1_lct_artifact(filepath, cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_lct = gpd.read_file(filepath).to_crs(TARGET_CRS)
    
    gdf_lct.columns = [col.lower().strip().replace(' ', '_') for col in gdf_lct.columns]
    
    numeric_cols = ['lct_connections', 'import', 'export']
    for col in numeric_cols:
        gdf_lct[col] = pd.to_numeric(gdf_lct[col], errors='coerce')
    
    initial_rows = len(gdf_lct)
    gdf_lct.dropna(subset=['lct_connections', '[REDACTED_BY_SCRIPT]'], inplace=True)
    if len(gdf_lct) < initial_rows:
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    for col in ['category', 'type']:
        if col in gdf_lct.columns:
            gdf_lct[col] = gdf_lct[col].str.strip().str.title()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_lct.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_lct

def get_or_create_l2_primary_sub_features(gdf_lct, cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return pd.read_parquet(cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    grouped_primary = gdf_lct.groupby('[REDACTED_BY_SCRIPT]')
    
    primary_summary = grouped_primary.agg(
        ukpn_psub_total_lct_connections=('lct_connections', 'sum'),
        ukpn_psub_total_lct_import_kw=('import', 'sum'),
        ukpn_psub_total_lct_export_kw=('export', 'sum')
    )
    
    primary_summary['[REDACTED_BY_SCRIPT]'] = grouped_primary.apply(lambda x: x[x['category'] == 'Demand']['lct_connections'].sum())
    primary_summary['[REDACTED_BY_SCRIPT]'] = grouped_primary.apply(lambda x: x[x['category'] == 'Generation']['lct_connections'].sum())
    
    primary_summary['[REDACTED_BY_SCRIPT]'] = primary_summary['[REDACTED_BY_SCRIPT]'] / (primary_summary['[REDACTED_BY_SCRIPT]'] + 1)
    
    primary_summary.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return primary_summary

def calculate_buffer_features(master_gdf, gdf_lct, id_col):
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    master_buffers = master_gdf.copy()
    master_buffers['geometry'] = master_buffers.geometry.buffer(BUFFER_RADIUS_METERS)

    logging.info("[REDACTED_BY_SCRIPT]")
    # Use the provided id_col for grouping. This is the key fix.
    if id_col not in master_buffers.columns:
        raise KeyError(f"[REDACTED_BY_SCRIPT]'{id_col}'[REDACTED_BY_SCRIPT]")

    sjoined = gpd.sjoin(gdf_lct, master_buffers[[id_col, 'geometry']], how="inner", predicate="within")

    logging.info("[REDACTED_BY_SCRIPT]")
    grouped = sjoined.groupby(id_col)
    
    agg_df = grouped.agg(
        ukpn_lct_secondary_sub_count_5km=(id_col, 'size'),
        ukpn_lct_total_connections_5km=('lct_connections', 'sum'),
        ukpn_lct_total_import_kw_5km=('import', 'sum'),
        ukpn_lct_total_export_kw_5km=('export', 'sum')
    )

    agg_df['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['category'] == 'Demand']['lct_connections'].sum())
    agg_df['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['category'] == 'Generation']['lct_connections'].sum())
    agg_df['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['type'] == 'Ev Charging Point']['lct_connections'].sum())
    agg_df['[REDACTED_BY_SCRIPT]'] = grouped.apply(lambda x: x[x['type'] == 'Solar']['lct_connections'].sum())
    agg_df['[REDACTED_BY_SCRIPT]'] = agg_df['[REDACTED_BY_SCRIPT]'] / (agg_df['[REDACTED_BY_SCRIPT]'] + 1)
    
    return agg_df

def execute(master_gdf):
    """
    Orchestrates the LCT feature engineering process.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Critical CRS Check (Pattern 1 Defense)
    if master_gdf.crs is None:
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf.set_crs(TARGET_CRS, inplace=True)
    elif master_gdf.crs.to_string() != TARGET_CRS:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        master_gdf = master_gdf.to_crs(TARGET_CRS)

    # V14 FIX: The orchestrator now passes a GDF with 'hex_id' as the index.
    # This executor'[REDACTED_BY_SCRIPT]'hex_id' to be a column for joins.
    # We will reset the index at the beginning and set it back at the end.
    id_col = 'hex_id'
    if master_gdf.index.name != id_col:
        if id_col in master_gdf.columns:
            master_gdf.set_index(id_col, inplace=True)
            logging.info(f"Set '{id_col}' as index.")
        else:
            logging.error(f"FATAL: '{id_col}'[REDACTED_BY_SCRIPT]")
            return master_gdf
            
    master_gdf_processed = master_gdf.reset_index()

    try:
        gdf_lct = get_or_create_l1_lct_artifact(LCT_DATA_INPUT, LCT_L1_CACHE_PATH)
        primary_sub_features = get_or_create_l2_primary_sub_features(gdf_lct, PRIMARY_SUB_FEATURES_L2_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf # Return original gdf

    # --- Link master_gdf to nearest primary substation ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    gdf_primary_subs = gdf_lct[['[REDACTED_BY_SCRIPT]', 'geometry']].drop_duplicates(subset='[REDACTED_BY_SCRIPT]').reset_index(drop=True)

    master_with_psub_link = gpd.sjoin_nearest(
        master_gdf_processed[[id_col, 'geometry']], 
        gdf_primary_subs, 
        how='left'
    )
    master_with_psub_link.drop_duplicates(subset=[id_col], keep='first', inplace=True)

    master_with_psub = master_gdf_processed.merge(
        master_with_psub_link[[id_col, '[REDACTED_BY_SCRIPT]']],
        on=id_col,
        how='left'
    )
    
    # --- Calculate Buffer Features ---
    buffer_features = calculate_buffer_features(master_gdf_processed, gdf_lct, id_col)
    
    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    final_gdf = master_with_psub.merge(buffer_features, left_on=id_col, right_index=True, how='left')
    final_gdf = final_gdf.merge(primary_sub_features, on='[REDACTED_BY_SCRIPT]', how='left')
    
    new_cols = list(buffer_features.columns) + list(primary_sub_features.columns)
    final_gdf[new_cols] = final_gdf[new_cols].fillna(NULL_SENTINEL)
    
    # --- V14 Finalization ---
    # Restore the original index to be compliant with the orchestrator's expectations.
    if id_col in final_gdf.columns:
        final_gdf.set_index(id_col, inplace=True)

    # Ensure all original columns are preserved
    for col in master_gdf.columns:
        if col not in final_gdf.columns:
            final_gdf[col] = master_gdf[col]

    # Artifact Validation: Ensure output remains a valid GeoDataFrame
    if not isinstance(final_gdf, gpd.GeoDataFrame):
        logging.warning("[REDACTED_BY_SCRIPT]")
        final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry', crs=TARGET_CRS)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf
