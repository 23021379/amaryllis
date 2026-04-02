import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input file
TRANSFORMER_ASSETS_INPUT = r"[REDACTED_BY_SCRIPT]"

# Artifact path
PRIMARY_TX_L1_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = 'EPSG:27700'

# Analysis parameters
NULL_SENTINEL = 0

def find_column_by_pattern(df, pattern):
    """[REDACTED_BY_SCRIPT]"""
    for col in df.columns:
        if pattern in col:
            return col
    return None

def get_or_create_l1_primary_tx_artifact(filepath, cache_path):
    """
    Loads, sanitizes, and aggregates the raw primary transformer asset data into a stable,
    one-row-per-site L1 artifact, caching the result.
    """
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_tx = gpd.read_file(filepath).to_crs(TARGET_CRS)

    gdf_tx.columns = [col.lower().strip().replace(' ', '_') for col in gdf_tx.columns]
    
    gdf_tx['onanrating_kva'] = pd.to_numeric(gdf_tx['onanrating_kva'], errors='coerce')
    gdf_tx.dropna(subset=['onanrating_kva', 'sitefunctionallocation'], inplace=True)
    
    # The original script notes values are in MVA; convert to kVA.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_tx['onan_rating_kva'] = gdf_tx['onanrating_kva'] * 1000

    logging.info("[REDACTED_BY_SCRIPT]")
    agg_rules = {
        'onan_rating_kva': ['sum', 'count', 'mean', 'max', 'min', 'var']
    }
    aggregated_data = gdf_tx.groupby('sitefunctionallocation').agg(agg_rules)
    
    aggregated_data.columns = [
        'ukpn_psub_total_kva', 'ukpn_psub_tx_count', 'ukpn_psub_avg_tx_kva',
        'ukpn_psub_max_tx_kva', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    aggregated_data['[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)

    sites_geometry = gdf_tx[['sitefunctionallocation', 'geometry']].drop_duplicates(subset='sitefunctionallocation').set_index('sitefunctionallocation')
    gdf_tx_l1 = sites_geometry.join(aggregated_data).reset_index()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_tx_l1.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_tx_l1

def execute(master_gdf):
    """
    Executor entry point for integrating UKPN Primary Transformer asset features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # V14 ID Management
    id_col = 'hex_id'
    if master_gdf.index.name != id_col:
        if id_col in master_gdf.columns:
            master_gdf.set_index(id_col, inplace=True)
        else:
            logging.error(f"FATAL: '{id_col}'[REDACTED_BY_SCRIPT]")
            return master_gdf
    
    master_gdf_processed = master_gdf.reset_index()

    # Gracefully skip if no UKPN data is present or if dno_source column is missing
    if 'dno_source' not in master_gdf_processed.columns or 'ukpn' not in master_gdf_processed['dno_source'].str.lower().unique():
        logging.info("[REDACTED_BY_SCRIPT]'dno_source'[REDACTED_BY_SCRIPT]")
        return master_gdf

    # Search for the primary substation column using multiple patterns
    primary_sub_col = find_column_by_pattern(master_gdf_processed, '[REDACTED_BY_SCRIPT]')
    if not primary_sub_col:
        primary_sub_col = find_column_by_pattern(master_gdf_processed, '[REDACTED_BY_SCRIPT]')
    
    if not primary_sub_col:
        logging.error("FATAL: 'primary_functionallocation'[REDACTED_BY_SCRIPT]")
        return master_gdf
    
    installed_cap_col = find_column_by_pattern(master_gdf_processed, '[REDACTED_BY_SCRIPT]')
    if not installed_cap_col:
        logging.warning("'installed_capacity_mwelec'[REDACTED_BY_SCRIPT]")

    try:
        gdf_tx_l1 = get_or_create_l1_primary_tx_artifact(TRANSFORMER_ASSETS_INPUT, PRIMARY_TX_L1_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")
    # V14 FIX: Use the processed dataframe for the merge
    final_gdf_processed = master_gdf_processed.merge(
        gdf_tx_l1.drop(columns='geometry', errors='ignore'),
        left_on=primary_sub_col,
        right_on='sitefunctionallocation',
        how='left'
    )
    final_gdf_processed.drop(columns=['sitefunctionallocation'], inplace=True, errors='ignore')

    # Feature Interaction: kVA per MW ratio
    if installed_cap_col and 'ukpn_psub_total_kva' in final_gdf_processed.columns:
        solar_capacity_kva = final_gdf_processed[installed_cap_col] * 1000
        final_gdf_processed['[REDACTED_BY_SCRIPT]'] = final_gdf_processed['ukpn_psub_total_kva'] / solar_capacity_kva
        final_gdf_processed['[REDACTED_BY_SCRIPT]'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Finalization
    new_cols = [col for col in gdf_tx_l1.columns if 'ukpn_psub_' in col]
    new_cols_to_fill = [col for col in new_cols if col in final_gdf_processed.columns]
    final_gdf_processed[new_cols_to_fill] = final_gdf_processed[new_cols_to_fill].fillna(NULL_SENTINEL)
    
    # Restore index
    if id_col in final_gdf_processed.columns:
        final_gdf_processed.set_index(id_col, inplace=True)

    # Ensure columns from original master_gdf are not lost
    for col in master_gdf.columns:
        if col not in final_gdf_processed.columns:
            final_gdf_processed[col] = master_gdf[col]

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf_processed
