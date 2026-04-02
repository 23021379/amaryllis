import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys
import re
from pathlib import Path
from scipy import stats
import warnings
from tqdm import tqdm

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

# Input paths
OA_DATA_DIR = Path(r"[REDACTED_BY_SCRIPT]")
OA_BOUNDARIES_PATH = OA_DATA_DIR.parent / 'boundaries' / '[REDACTED_BY_SCRIPT]'

# Artifact paths
ARTIFACT_DIR = OA_DATA_DIR / 'artifacts'
L1_NON_SPATIAL_OA_CACHE = ARTIFACT_DIR / '[REDACTED_BY_SCRIPT]'
L2_GEOSPATIAL_OA_CACHE = ARTIFACT_DIR / '[REDACTED_BY_SCRIPT]'
L3_ENRICHED_OA_CACHE = ARTIFACT_DIR / '[REDACTED_BY_SCRIPT]'

# Geospatial constants
TARGET_CRS = "EPSG:27700"
NULL_SENTINEL = 0

# --- HELPER & ARTIFACT FUNCTIONS ---

def sanitize_col_name(col_name):
    s = str(col_name).strip().lower()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^0-9a-zA-Z_]', '', s)
    s = re.sub(r'^(\d)', r'_\1', s)
    return s

def get_or_create_l1_oa_non_spatial_artifact():
    if os.path.exists(L1_NON_SPATIAL_OA_CACHE):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return pd.read_parquet(L1_NON_SPATIAL_OA_CACHE)

    logging.info("[REDACTED_BY_SCRIPT]")
    pop_df = pd.read_csv(OA_DATA_DIR / 'OA - population.csv', index_col='2021 output area')
    pop_df.columns = [f"pop_{c}" for c in pop_df.columns]
    pop_df.index.name = 'oa21cd'
    master_df = pop_df

    categorical_files = {
        'dep': ('ons-deprivation.csv', '[REDACTED_BY_SCRIPT]'),
        'edu': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'emp': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'hhd_comp': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'car': ('ons-vehicles.csv', '[REDACTED_BY_SCRIPT]'),
    }

    for prefix, (filename, desc_col) in categorical_files.items():
        df = pd.read_csv(OA_DATA_DIR / filename)
        code_col = [c for c in df.columns if 'Code' in c and 'Area' not in c][0]
        df = df[df[code_col] != -8]
        pivot_df = df.pivot_table(index='Output Areas Code', columns=desc_col, values='Observation', fill_value=0)
        pivot_df.columns = [f"[REDACTED_BY_SCRIPT]" for c in pivot_df.columns]
        pivot_df.index.name = 'oa21cd'
        master_df = master_df.join(pivot_df, how='left')

    oac_df = pd.read_csv(OA_DATA_DIR / '[REDACTED_BY_SCRIPT]', index_col='oa21cd')
    oac_df.columns = [sanitize_col_name(c) for c in oac_df.columns]
    master_df = master_df.join(oac_df, how='left')

    hhd_comp_cols = [c for c in master_df.columns if c.startswith('hhd_comp_')]
    master_df['total_households'] = master_df[hhd_comp_cols].sum(axis=1)

    for col in master_df.columns:
        if 'oac' not in col and 'group' not in col:
             master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(NULL_SENTINEL)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    master_df.to_parquet(L1_NON_SPATIAL_OA_CACHE)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_df

def get_or_create_l2_oa_geospatial_artifact(df_l1):
    if os.path.exists(L2_GEOSPATIAL_OA_CACHE):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_file(L2_GEOSPATIAL_OA_CACHE)

    logging.info("[REDACTED_BY_SCRIPT]")
    oa_gdf = gpd.read_file(OA_BOUNDARIES_PATH).to_crs(TARGET_CRS)
    oa_gdf = oa_gdf.rename(columns={'OA21CD': 'oa21cd'})
    merged_gdf = oa_gdf[['oa21cd', 'geometry']].merge(df_l1, on='oa21cd', how='inner')
    
    merged_gdf.to_file(L2_GEOSPATIAL_OA_CACHE, driver='GeoParquet')
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return merged_gdf

def get_or_create_l3_oa_enriched_artifact(gdf_l2):
    if os.path.exists(L3_ENRICHED_OA_CACHE):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_file(L3_ENRICHED_OA_CACHE)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf = gdf_l2.copy()
    gdf['total_households'] = gdf['total_households'].replace(0, np.nan)

    gdf['oa_dep_multi_dim_idx'] = (
        gdf['[REDACTED_BY_SCRIPT]'] * 2 +
        gdf['[REDACTED_BY_SCRIPT]'] * 3 +
        gdf['[REDACTED_BY_SCRIPT]'] * 4
    ) / gdf['total_households']

    gdf['[REDACTED_BY_SCRIPT]'] = (
        gdf['[REDACTED_BY_SCRIPT]'] +
        gdf['[REDACTED_BY_SCRIPT]']
    ) / gdf['total_households']

    pop_cols = [f'pop_{year}' for year in range(2018, 2023)]
    time_axis = np.arange(len(pop_cols))
    def get_trend(row):
        pop_values = row[pop_cols].to_numpy(dtype=np.float64)
        valid_mask = pop_values > 0
        if np.sum(valid_mask) < 2: return 0.0
        slope, _, _, _, _ = stats.linregress(time_axis[valid_mask], pop_values[valid_mask])
        return slope
    gdf['oa_pop_trend'] = gdf.apply(get_trend, axis=1)

    gdf['[REDACTED_BY_SCRIPT]'] = gdf['pop_2022'] / (gdf.geometry.area / 1_000_000)

    oac_cols = ['supergroup', 'group', 'subgroup']
    existing_oac_cols = [col for col in oac_cols if col in gdf.columns]
    if existing_oac_cols:
        gdf = pd.get_dummies(gdf, columns=existing_oac_cols, prefix={'supergroup': 'oac_sg', 'group': 'oac_g', 'subgroup': 'oac_subg'}, dtype=int)

    index_cols = ['oa_dep_multi_dim_idx', '[REDACTED_BY_SCRIPT]', 'oa_pop_trend', '[REDACTED_BY_SCRIPT]']
    gdf[index_cols] = gdf[index_cols].fillna(NULL_SENTINEL)

    gdf.to_file(L3_ENRICHED_OA_CACHE, driver='GeoParquet')
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf

# --- MAIN EXECUTOR FUNCTION ---

def execute(master_gdf):
    """
    Executor entry point for integrating OA-level demographic and landscape features.
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

    try:
        df_l1 = get_or_create_l1_oa_non_spatial_artifact()
        gdf_l2 = get_or_create_l2_oa_geospatial_artifact(df_l1)
        gdf_l3 = get_or_create_l3_oa_enriched_artifact(gdf_l2)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    # --- Part A: Buffered Area-Weighted Average ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # V14 FIX: Work with a copy that has the index as a column
    master_gdf_processed = master_gdf.reset_index()
    site_buffers = master_gdf_processed.copy()
    site_buffers['geometry'] = site_buffers.geometry.buffer(1000)

    intersection = gpd.overlay(site_buffers, gdf_l3, how='intersection')
    intersection['fragment_area'] = intersection.geometry.area
    
    # Group by the authoritative ID column
    buffer_total_areas = intersection.groupby(id_col)['fragment_area'].transform('sum')
    intersection['weight'] = intersection['fragment_area'] / buffer_total_areas

    feature_cols = [c for c in gdf_l3.columns if c.startswith(('oa_', 'oac_')) and c not in ['geometry']]
    for col in feature_cols:
        intersection[col] = intersection[col] * intersection['weight']

    buffered_features = intersection.groupby(id_col)[feature_cols].sum()
    buffered_features.columns = [f"oa_buffered_{c}" for c in buffered_features.columns]

    # --- Part B: Dominant Character (Centroid) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    site_centroids = master_gdf_processed.copy()
    site_centroids['geometry'] = site_centroids.geometry.centroid
    
    # Use the same feature columns as Part A
    centroid_join = gpd.sjoin(site_centroids, gdf_l3[feature_cols + ['geometry']], how='left', predicate='within')
    
    centroid_features = centroid_join.drop_duplicates(subset=id_col, keep='first')
    centroid_features = centroid_features.drop(columns=['index_right', 'geometry'])
    centroid_features = centroid_features.set_index(id_col)
    centroid_features.columns = [f"oa_centroid_{c}" for c in centroid_features.columns]
    
    # --- Merge & Finalize ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # V14 FIX: Join back to the original master_gdf which has the correct index
    final_gdf = master_gdf.join(buffered_features)
    final_gdf = final_gdf.join(centroid_features)

    # --- Synthesize Final SICs ---
    logging.info("[REDACTED_BY_SCRIPT]")
    if 'oa_buffered_oa_pop_trend' in final_gdf.columns and '[REDACTED_BY_SCRIPT]' in final_gdf.columns:
        final_gdf['[REDACTED_BY_SCRIPT]'] = final_gdf['oa_buffered_oa_pop_trend'] * final_gdf['[REDACTED_BY_SCRIPT]']
    
    affluent_col = '[REDACTED_BY_SCRIPT]'
    if affluent_col not in final_gdf.columns: final_gdf[affluent_col] = 0
    if '[REDACTED_BY_SCRIPT]' in final_gdf.columns:
        final_gdf['[REDACTED_BY_SCRIPT]'] = final_gdf[affluent_col] * final_gdf['[REDACTED_BY_SCRIPT]']

    # Fill NaNs for all newly added columns
    new_cols = [col for col in final_gdf.columns if col not in master_gdf.columns and col != 'geometry']
    final_gdf[new_cols] = final_gdf[new_cols].fillna(NULL_SENTINEL)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf
