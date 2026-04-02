import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
import os
import re
from tqdm import tqdm

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
NG_DFES_LPA_FORECAST_CSV_PATH = r"[REDACTED_BY_SCRIPT]"
LPA_BOUNDARIES_PATH = r"[REDACTED_BY_SCRIPT]"
L1_DFES_LOOKUP_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Architectural Hyperparameters
TARGET_CRS = "EPSG:27700"
TARGET_SCENARIO = 'Holistic Transition'
HORIZON_YEARS = 5
NULL_SENTINEL = -1
EPSILON = 1e-6 # For numerical stability

def reconcile_authority_name(name):
    """[REDACTED_BY_SCRIPT]"""
    if not isinstance(name, str):
        return None
    name = name.lower()
    suffixes_to_remove = [' lpa', ' council', ' unitary authority', ' borough', ' district']
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return ''.join(filter(str.isalnum, name))

def get_or_create_l1_dfes_lookup(data_path, cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return pd.read_csv(cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    df_ng = pd.read_csv(data_path)
    required_cols = ['local_authority', 'year', 'scenario', 'energy_type', 'TWh_total', '[REDACTED_BY_SCRIPT]']
    if not all(col in df_ng.columns for col in required_cols):
        raise ValueError(f"[REDACTED_BY_SCRIPT]")

    df_ng = df_ng[df_ng['scenario'] == TARGET_SCENARIO].copy()
    for col in ['TWh_total', '[REDACTED_BY_SCRIPT]']:
        df_ng[col] = pd.to_numeric(df_ng[col], errors='coerce')
    df_ng.dropna(subset=['TWh_total', '[REDACTED_BY_SCRIPT]'], inplace=True)
    df_ng['year'] = df_ng['year'].astype(int)

    df_pivot = df_ng.pivot_table(
        index=['local_authority', 'year'], columns='energy_type', values='TWh_total'
    ).reset_index()
    df_pivot.columns.name = None
    df_pivot.rename(columns={'Generation': 'generation_twh', 'Demand': 'demand_twh', 'Storage': 'storage_twh'}, inplace=True)

    df_renewables = df_ng[['local_authority', 'year', '[REDACTED_BY_SCRIPT]']].drop_duplicates()
    df_analysis = pd.merge(df_pivot, df_renewables, on=['local_authority', 'year'])
    df_analysis['reconciled_key'] = df_analysis['local_authority'].apply(reconcile_authority_name)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_analysis.to_csv(cache_path, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return df_analysis

def calculate_lpa_context_features(solar_farm_row, dfes_lookup):
    """[REDACTED_BY_SCRIPT]"""
    submission_year = solar_farm_row['submission_year']
    reconciled_key = solar_farm_row['reconciled_key']
    target_future_year = submission_year + HORIZON_YEARS

    default_series = pd.Series({
        '[REDACTED_BY_SCRIPT]': NULL_SENTINEL,
        '[REDACTED_BY_SCRIPT]': NULL_SENTINEL,
        '[REDACTED_BY_SCRIPT]': NULL_SENTINEL
    })

    if pd.isna(reconciled_key) or pd.isna(submission_year):
        return default_series
        
    try:
        lpa_timeseries = dfes_lookup.loc[reconciled_key]
    except KeyError:
        return default_series

    present_data = lpa_timeseries[lpa_timeseries.index <= submission_year]
    present_row = present_data.loc[present_data.index.max()] if not present_data.empty else None

    future_data = lpa_timeseries[lpa_timeseries.index >= target_future_year]
    future_row = future_data.loc[future_data.index.min()] if not future_data.empty else None

    if present_row is None or future_row is None:
        return default_series

    gen_growth = future_row['generation_twh'] - present_row['generation_twh']
    dem_growth = future_row['demand_twh'] - present_row['demand_twh']

    gen_growth_pct = (gen_growth / (present_row['generation_twh'] + EPSILON)) * 100
    gen_dem_ratio = gen_growth / (dem_growth + EPSILON)
    renewables_target = future_row['[REDACTED_BY_SCRIPT]']

    return pd.Series({
        '[REDACTED_BY_SCRIPT]': gen_growth_pct,
        '[REDACTED_BY_SCRIPT]': gen_dem_ratio,
        '[REDACTED_BY_SCRIPT]': renewables_target
    })

def execute(master_gdf):
    """
    Executor entry point for calculating National Grid DFES features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    if 'submission_year' not in master_gdf.columns:
        logging.error("FATAL: 'submission_year'[REDACTED_BY_SCRIPT]")
        return master_gdf

    try:
        gdf_lpa = gpd.read_file(LPA_BOUNDARIES_PATH).to_crs(TARGET_CRS)
        dfes_lookup_table = get_or_create_l1_dfes_lookup(NG_DFES_LPA_FORECAST_CSV_PATH, L1_DFES_LOOKUP_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf_with_lpa = gpd.sjoin(master_gdf, gdf_lpa[['LPA23NM', 'geometry']], how='left', predicate='within')
    master_gdf_with_lpa = master_gdf_with_lpa.drop_duplicates(subset='hex_id', keep='first')

    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf_with_lpa['reconciled_key'] = master_gdf_with_lpa['LPA23NM'].apply(reconcile_authority_name)

    logging.info("[REDACTED_BY_SCRIPT]")
    dfes_lookup = dfes_lookup_table.set_index(['reconciled_key', 'year'])

    tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
    df_new_features = master_gdf_with_lpa.progress_apply(
        calculate_lpa_context_features, axis=1, dfes_lookup=dfes_lookup
    )

    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf_with_features = master_gdf_with_lpa.join(df_new_features)
    
    # Clean up join artifacts and fill nulls
    cols_to_drop = ['index_right', 'LPA23NM', 'reconciled_key']
    master_gdf_with_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    new_feature_cols = df_new_features.columns
    master_gdf_with_features[new_feature_cols] = master_gdf_with_features[new_feature_cols].fillna(NULL_SENTINEL)

    # Ensure the geometry column is correctly named before returning
    if 'geometry' in master_gdf_with_features.columns:
        master_gdf_with_features = master_gdf_with_features.set_geometry('geometry')

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf_with_features
