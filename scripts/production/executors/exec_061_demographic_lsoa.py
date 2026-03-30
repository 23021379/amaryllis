import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
import sys
import re
from tqdm import tqdm
import traceback

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input paths
BASE_DIR = r"[REDACTED_BY_SCRIPT]"
BOUNDARY_PATH = r"[REDACTED_BY_SCRIPT]"
LSOA_AHAH_PATH = os.path.join(BASE_DIR, 'AHAH_V4.csv')
LSOA_OAC_PATH = os.path.join(BASE_DIR, 'iuc2018.csv')
LSOA_COUNCIL_TAX_PATH = os.path.join(BASE_DIR, '[REDACTED_BY_SCRIPT]')
LSOA_NDVI_PATH = os.path.join(BASE_DIR, 'LSOA veg.csv')
LSOA_RUC_PATH = os.path.join(BASE_DIR, '[REDACTED_BY_SCRIPT]')
LSOA_BOUNDARIES_PATH = os.path.join(BOUNDARY_PATH, '[REDACTED_BY_SCRIPT]')
LPA_BOUNDARIES_PATH = os.path.join(BOUNDARY_PATH, '[REDACTED_BY_SCRIPT]')

# Artifact paths
L1_NON_SPATIAL_CACHE = os.path.join(BASE_DIR, 'artifacts', '[REDACTED_BY_SCRIPT]')
L2_GEOSPATIAL_CACHE = os.path.join(BASE_DIR, 'artifacts', '[REDACTED_BY_SCRIPT]')
L3_LPA_AGG_CACHE = os.path.join(BASE_DIR, 'artifacts', '[REDACTED_BY_SCRIPT]')

# Geospatial constants
TARGET_CRS = "EPSG:27700"
NULL_SENTINEL = 0

# --- HELPER FUNCTIONS ---
def _normalize_columns(df, prefix=None):
    cols = [col.lower().strip().replace(' ', '_') for col in df.columns]
    if prefix:
        df.columns = [f"{prefix}{col}" for col in cols]
    else:
        df.columns = cols
    return df

def _calculate_property_value_idx(row, prefix):
    bands = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7}
    numerator = 0
    denominator = row.get(f'{prefix}ct_total', 0)
    if pd.isna(denominator) or denominator == 0: return np.nan
    for band, weight in bands.items():
        count = row.get(f'[REDACTED_BY_SCRIPT]', 0)
        if pd.notna(count):
            numerator += float(weight) * float(count)
        
    return numerator / float(denominator)

def _find_lpa_identifier_column(gdf):
    patterns = [r'LAD\d*CD', r'LPA\d*CD', r'LAD\d*NM']
    for pattern in patterns:
        for col in gdf.columns:
            if re.fullmatch(pattern, col, re.IGNORECASE):
                return col
    raise ValueError(f"[REDACTED_BY_SCRIPT]")

# --- ARTIFACT CREATION FUNCTIONS ---

def get_or_create_l1_non_spatial_artifact():
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(L1_NON_SPATIAL_CACHE):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return pd.read_parquet(L1_NON_SPATIAL_CACHE)

    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Load and process each file, ensuring numeric conversion
    df_ahah = _normalize_columns(pd.read_csv(LSOA_AHAH_PATH))
    for col in df_ahah.columns:
        if col != 'lsoa21cd':
            df_ahah[col] = pd.to_numeric(df_ahah[col], errors='coerce')
    df_master = df_ahah.set_index('lsoa21cd')

    df_oac = _normalize_columns(pd.read_csv(LSOA_OAC_PATH))
    df_master = df_master.join(df_oac.set_index('lsoa11_cd'), how='left')

    df_ct = _normalize_columns(pd.read_csv(LSOA_COUNCIL_TAX_PATH, engine='python'))
    df_ct['lsoa21cd'] = df_ct['[REDACTED_BY_SCRIPT]'].str.extract(r'(E\d{8}|W\d{8})')
    df_ct = df_ct.dropna(subset=['lsoa21cd']).rename(columns={
        'total': 'ct_total', 'band_a': 'ct_band_a', 'band_b': 'ct_band_b', 'band_c': 'ct_band_c',
        'band_d': 'ct_band_d', 'band_e': 'ct_band_e', 'band_f': 'ct_band_f', 'band_g': 'ct_band_g'
    })
    # FIX: Convert council tax columns to numeric BEFORE the join.
    ct_cols_to_process = [col for col in df_ct.columns if col.startswith('ct_')]
    for col in ct_cols_to_process:
        df_ct[col] = pd.to_numeric(df_ct[col], errors='coerce')
    ct_cols = ['lsoa21cd'] + ct_cols_to_process
    df_master = df_master.join(df_ct[ct_cols].set_index('lsoa21cd'), how='left')

    df_ndvi = _normalize_columns(pd.read_csv(LSOA_NDVI_PATH))
    for col in df_ndvi.columns:
        if col != 'lsoa21cd':
            df_ndvi[col] = pd.to_numeric(df_ndvi[col], errors='coerce')
    df_master = df_master.join(df_ndvi.set_index('lsoa21cd'), how='left')

    df_ruc = _normalize_columns(pd.read_csv(LSOA_RUC_PATH))
    df_master = df_master.join(df_ruc[['lsoa21cd', 'ruc21cd']].set_index('lsoa21cd'), how='left')

    os.makedirs(os.path.dirname(L1_NON_SPATIAL_CACHE), exist_ok=True)
    df_master.to_parquet(L1_NON_SPATIAL_CACHE)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return df_master

def get_or_create_l2_geospatial_artifact(df_l1):
    if os.path.exists(L2_GEOSPATIAL_CACHE):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_file(L2_GEOSPATIAL_CACHE)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_lsoa_boundaries = gpd.read_file(LSOA_BOUNDARIES_PATH).to_crs(TARGET_CRS)
    
    gdf_l2_master = gdf_lsoa_boundaries.merge(df_l1, left_on='LSOA21CD', right_index=True, how='left')
    gdf_l2_master = gdf_l2_master.drop(columns=['lsoa21nm', 'lsoa21nmw'], errors='ignore')
    
    os.makedirs(os.path.dirname(L2_GEOSPATIAL_CACHE), exist_ok=True)
    gdf_l2_master.to_file(L2_GEOSPATIAL_CACHE, driver='GPKG')
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_l2_master

def get_or_create_l3_lpa_agg_artifact(gdf_l2_master):
    if os.path.exists(L3_LPA_AGG_CACHE):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return pd.read_parquet(L3_LPA_AGG_CACHE)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_lpa_boundaries = gpd.read_file(LPA_BOUNDARIES_PATH).to_crs(TARGET_CRS)
    lpa_id_col = _find_lpa_identifier_column(gdf_lpa_boundaries)

    gdf_lsoa_processed = gdf_l2_master.copy()
    # Ensure categorical columns are strings before getting dummies
    gdf_lsoa_processed['grp_label'] = gdf_lsoa_processed['grp_label'].astype(str)
    gdf_lsoa_processed['ruc21nm'] = gdf_lsoa_processed['ruc21nm'].astype(str)
    
    oac_dummies = pd.get_dummies(gdf_lsoa_processed['grp_label'], prefix='oac_pct', dtype=int)
    ruc_dummies = pd.get_dummies(gdf_lsoa_processed['ruc21nm'], prefix='ruc_pct', dtype=int)
    gdf_lsoa_processed = gdf_lsoa_processed.join([oac_dummies, ruc_dummies])

    gdf_lsoa_with_lpa = gpd.sjoin(gdf_lsoa_processed, gdf_lpa_boundaries[[lpa_id_col, 'geometry']], how='inner', predicate='intersects')
    gdf_lsoa_with_lpa.drop_duplicates(subset=['LSOA21CD'], keep='first', inplace=True)

    gdf_lsoa_with_lpa['property_value_idx'] = gdf_lsoa_with_lpa.apply(lambda row: _calculate_property_value_idx(row, ''), axis=1)
    gdf_lsoa_with_lpa['[REDACTED_BY_SCRIPT]'] = (gdf_lsoa_with_lpa['ah4no2_pct'] + gdf_lsoa_with_lpa['ah4so2_pct'] + gdf_lsoa_with_lpa['ah4pm10_pct']) - gdf_lsoa_with_lpa['ah4g_pct']
    ruc_is_rural_flag = gdf_lsoa_with_lpa['ruc21nm'].str.contains('Rural', na=False).astype(int)
    gdf_lsoa_with_lpa['[REDACTED_BY_SCRIPT]'] = (gdf_lsoa_with_lpa['ndvi_mean'] * (1 - gdf_lsoa_with_lpa['ndvi_std'])) * ruc_is_rural_flag

    index_cols = ['property_value_idx', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    pct_cols = list(oac_dummies.columns) + list(ruc_dummies.columns)
    agg_spec = {col: ['mean', 'std'] for col in index_cols}
    for col in pct_cols: agg_spec[col] = 'mean'

    df_lpa_agg = gdf_lsoa_with_lpa.groupby(lpa_id_col).agg(agg_spec)
    df_lpa_agg.columns = ['_'.join(col).strip() for col in df_lpa_agg.columns.values]
    df_lpa_agg = df_lpa_agg.rename(columns=lambda c: f"[REDACTED_BY_SCRIPT]'_mean', '').replace('_std', '_stddev')}")

    os.makedirs(os.path.dirname(L3_LPA_AGG_CACHE), exist_ok=True)
    df_lpa_agg.to_parquet(L3_LPA_AGG_CACHE)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return df_lpa_agg

# --- MAIN EXECUTOR FUNCTION ---

def execute(master_gdf):
    """
    Executor entry point for integrating LSOA-level demographic and landscape features.
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
        df_l1 = get_or_create_l1_non_spatial_artifact()
        gdf_l2 = get_or_create_l2_geospatial_artifact(df_l1)
        df_l3 = get_or_create_l3_lpa_agg_artifact(gdf_l2)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        logging.error(traceback.format_exc())
        return master_gdf

    # --- Phase 3: Site-Level Linkage ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # The master_gdf comes in with 'hex_id' as the index. The sjoin will preserve this.
    gdf_solar_enriched = gpd.sjoin(master_gdf, gdf_l2, how='left', predicate='intersects')
    
    # FIX: Deduplicate based on the index, which is the correct approach.
    gdf_solar_enriched = gdf_solar_enriched[~gdf_solar_enriched.index.duplicated(keep='first')]
    gdf_solar_enriched.drop(columns=['index_right'], inplace=True, errors='ignore')

    lsoa_cols = gdf_l2.columns.drop('geometry')
    rename_dict = {col: f'[REDACTED_BY_SCRIPT]' for col in lsoa_cols}
    gdf_solar_enriched.rename(columns=rename_dict, inplace=True)

    # --- Phase 4: LPA-Level Linkage ---
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_lpa_boundaries = gpd.read_file(LPA_BOUNDARIES_PATH).to_crs(TARGET_CRS)
    lpa_id_col = _find_lpa_identifier_column(gdf_lpa_boundaries)
    
    # The sjoin might add an LPA ID column; we ensure it's clean before the next join.
    if lpa_id_col in gdf_solar_enriched.columns:
        gdf_solar_enriched.drop(columns=[lpa_id_col], inplace=True)
        
    gdf_solar_with_lpa_id = gpd.sjoin(gdf_solar_enriched, gdf_lpa_boundaries[[lpa_id_col, 'geometry']], how='left', predicate='intersects')
    
    # FIX: Deduplicate based on the index again after the second join.
    gdf_solar_with_lpa_id = gdf_solar_with_lpa_id[~gdf_solar_with_lpa_id.index.duplicated(keep='first')]
    gdf_solar_with_lpa_id.drop(columns=['index_right'], inplace=True, errors='ignore')

    # The merge operation works correctly with the LPA ID column and the L3 index.
    final_gdf = gdf_solar_with_lpa_id.merge(df_l3, left_on=lpa_id_col, right_index=True, how='left')

    # --- Phase 5: Strategic Feature Creation ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Ensure columns exist before calculations
    if '[REDACTED_BY_SCRIPT]' in final_gdf.columns:
        final_gdf['[REDACTED_BY_SCRIPT]'] = final_gdf.apply(lambda row: _calculate_property_value_idx(row, 'site_lsoa_'), axis=1)
        final_gdf['[REDACTED_BY_SCRIPT]'] = (final_gdf['[REDACTED_BY_SCRIPT]'] + final_gdf['[REDACTED_BY_SCRIPT]'] + final_gdf['[REDACTED_BY_SCRIPT]']) - final_gdf['site_lsoa_ah4g_pct']
        ruc_is_rural_flag = final_gdf['site_lsoa_ruc21nm'].str.contains('Rural', na=False).astype(int)
        final_gdf['[REDACTED_BY_SCRIPT]'] = (final_gdf['site_lsoa_ndvi_mean'] * (1 - final_gdf['site_lsoa_ndvi_std'])) * ruc_is_rural_flag

        final_gdf['delta_property_value'] = final_gdf['[REDACTED_BY_SCRIPT]'] / (final_gdf['[REDACTED_BY_SCRIPT]'] + 0.01)
        final_gdf['delta_env_health_disadvantage'] = final_gdf['[REDACTED_BY_SCRIPT]'] / (final_gdf['[REDACTED_BY_SCRIPT]'] + 0.01)
    
    # One-hot encode site-level categoricals
    if 'site_lsoa_grp_label' in final_gdf.columns:
        final_gdf = pd.get_dummies(final_gdf, columns=['site_lsoa_grp_label'], prefix='site_lsoa_oac', dummy_na=False, dtype=int)
    if 'site_lsoa_ruc21nm' in final_gdf.columns:
        final_gdf = pd.get_dummies(final_gdf, columns=['site_lsoa_ruc21nm'], prefix='site_lsoa_ruc', dummy_na=False, dtype=int)

    # Fill NaNs for all newly added columns
    new_cols = [col for col in final_gdf.columns if col not in master_gdf.columns and col != 'geometry']
    final_gdf[new_cols] = final_gdf[new_cols].fillna(NULL_SENTINEL)

    # V14 Finalization: Ensure original columns are preserved
    for col in master_gdf.columns:
        if col not in final_gdf.columns:
            final_gdf[col] = master_gdf[col]

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf
