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
pd.set_option('[REDACTED_BY_SCRIPT]', True)

# Input Artifacts
LCT_INPUT = r"[REDACTED_BY_SCRIPT]"
PRIMARY_SUBS_AUTH_INPUT = r"[REDACTED_BY_SCRIPT]"
L2_LCT_SPATIOTEMPORAL_CACHE = r"[REDACTED_BY_SCRIPT]"

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
KNN_NEIGHBORS = 5

def clean_col_names(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
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

def get_or_create_l2_lct_artifact(lct_path: str, primary_subs_path: str, cache_path: str) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf_cached = gpd.read_parquet(cache_path)
        # V6.8 FIX: Backward compatibility for cached artifacts with old column names.
        # Check for a non-prefixed column and rename if necessary.
        if 'connected_month' in gdf_cached.columns and 'ng_lct_connected_month' not in gdf_cached.columns:
            logging.warning("[REDACTED_BY_SCRIPT]")
            rename_map = {col: f"ng_lct_{col}" for col in ['ev', 'es', 'hp', 'pv', 'connected_month', '[REDACTED_BY_SCRIPT]'] if col in gdf_cached.columns}
            gdf_cached.rename(columns=rename_map, inplace=True)
        return gdf_cached

    logging.info("[REDACTED_BY_SCRIPT]")
    
    df_lct = pd.read_csv(lct_path, dtype=str)
    df_lct = clean_col_names(df_lct, 'ng_lct')
    
    lct_cols = ['ng_lct_ev', 'ng_lct_es', 'ng_lct_hp', 'ng_lct_pv']
    for col in lct_cols:
        df_lct[col] = pd.to_numeric(df_lct[col], errors='coerce')
    df_lct['ng_lct_connected_month'] = pd.to_datetime(df_lct['ng_lct_connected_month'], errors='coerce')
    
    initial_count = len(df_lct)
    df_lct.dropna(subset=lct_cols + ['ng_lct_connected_month', '[REDACTED_BY_SCRIPT]'], inplace=True)
    if len(df_lct) < initial_count:
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    gdf_primary_subs = gpd.read_parquet(primary_subs_path)
    
    df_lct['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(df_lct['[REDACTED_BY_SCRIPT]'], errors='coerce').astype('Int64')
    gdf_primary_subs['substationnumber'] = pd.to_numeric(gdf_primary_subs['substationnumber'], errors='coerce').astype('Int64')
    
    gdf_merged = pd.merge(
        df_lct, gdf_primary_subs,
        left_on='[REDACTED_BY_SCRIPT]', right_on='substationnumber',
        how='left'
    )
    
    orphans = gdf_merged['geometry'].isna().sum()
    if orphans > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf_merged.dropna(subset=['geometry'], inplace=True)

    gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry='geometry', crs=TARGET_CRS)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_merged.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_merged

def synthesize_point_in_time_lct_features(solar_farm_row: pd.Series, lct_gdf: gpd.GeoDataFrame, primary_subs_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    primary_subs_gdf = primary_subs_gdf.copy()
    submission_date = solar_farm_row['submission_date']
    
    lct_snapshot = lct_gdf[lct_gdf['ng_lct_connected_month'] <= submission_date]
    
    if lct_snapshot.empty:
        primary_subs_gdf['ng_lct_total_connections'] = 0
        primary_subs_gdf['[REDACTED_BY_SCRIPT]'] = 0
        primary_subs_gdf['ng_lct_pv_ratio'] = 0
        primary_subs_gdf['ng_lct_ev_hp_ratio'] = 0
        primary_subs_gdf['[REDACTED_BY_SCRIPT]'] = -1
        return primary_subs_gdf

    agg_df = lct_snapshot.groupby('[REDACTED_BY_SCRIPT]')[['ng_lct_ev', 'ng_lct_es', 'ng_lct_hp', 'ng_lct_pv']].sum()
    
    agg_df['ng_lct_total_connections'] = agg_df.sum(axis=1)
    generation_lcts = agg_df['ng_lct_pv'] + agg_df['ng_lct_es']
    demand_lcts = agg_df['ng_lct_ev'] + agg_df['ng_lct_hp']
    
    agg_df['[REDACTED_BY_SCRIPT]'] = generation_lcts / (demand_lcts + 1)
    agg_df['ng_lct_pv_ratio'] = agg_df['ng_lct_pv'] / (agg_df['ng_lct_total_connections'] + 1)
    agg_df['ng_lct_ev_hp_ratio'] = demand_lcts / (agg_df['ng_lct_total_connections'] + 1)

    latest_lct_date = lct_snapshot['ng_lct_connected_month'].max()
    agg_df['[REDACTED_BY_SCRIPT]'] = round(((submission_date - latest_lct_date).days) / 30.44)

    enriched_subs = primary_subs_gdf.merge(
        agg_df, left_on='substationnumber', right_index=True, how='left'
    ).fillna(0)
    
    return enriched_subs

def execute(master_gdf):
    """
    Executor entry point for calculating temporally-aware LCT features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # V6.6: Artificially create submission date for predictive runs if not present.
    if 'submission_date' not in master_gdf.columns:
        logging.warning("'submission_date'[REDACTED_BY_SCRIPT]'s date.")
        prediction_date = pd.to_datetime("2025-11-16")
        master_gdf['submission_date'] = prediction_date
        # Also create 'submission_year' for downstream DFES analysis.
        if 'submission_year' not in master_gdf.columns:
            master_gdf['submission_year'] = prediction_date.year
    
    master_gdf['submission_date'] = pd.to_datetime(master_gdf['submission_date'], errors='coerce')
    if master_gdf['submission_date'].isnull().any():
        logging.warning("[REDACTED_BY_SCRIPT]")

    try:
        gdf_lct_spatiotemporal = get_or_create_l2_lct_artifact(LCT_INPUT, PRIMARY_SUBS_AUTH_INPUT, L2_LCT_SPATIOTEMPORAL_CACHE)
        gdf_primary_subs = gpd.read_parquet(PRIMARY_SUBS_AUTH_INPUT)
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]"); return master_gdf
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]"); return master_gdf

    primary_sub_coords = np.array(list(gdf_primary_subs.geometry.apply(lambda p: (p.x, p.y))))
    spatial_index = cKDTree(primary_sub_coords)
    solar_coords = np.array(list(master_gdf.geometry.apply(lambda p: (p.x, p.y))))
    _, nearest_indices_for_all_sites = spatial_index.query(solar_coords, k=KNN_NEIGHBORS)

    all_solar_features = []
    
    for i, (index, solar_farm_row) in enumerate(tqdm(master_gdf.iterrows(), total=len(master_gdf), desc="[REDACTED_BY_SCRIPT]")):
        # V6.7 FIX: 'amaryllis_id' is the index, not a column. Access it from the 'index' variable.
        site_features = {'amaryllis_id': index}
        
        if pd.isna(solar_farm_row['submission_date']):
            all_solar_features.append(site_features)
            continue

        gdf_enriched_subs = synthesize_point_in_time_lct_features(solar_farm_row, gdf_lct_spatiotemporal, gdf_primary_subs)
        
        nearest_sub_indices = nearest_indices_for_all_sites[i]
        gdf_nearest_subs = gdf_enriched_subs.iloc[nearest_sub_indices]
        nearest_sub_features = gdf_nearest_subs.iloc[0]
        
        feature_cols = [
            'ng_lct_total_connections', '[REDACTED_BY_SCRIPT]',
            'ng_lct_pv_ratio', 'ng_lct_ev_hp_ratio'
        ]
        agg_calcs = gdf_nearest_subs[feature_cols].agg(['mean', 'std'])

        site_features.update({
            '[REDACTED_BY_SCRIPT]': nearest_sub_features['[REDACTED_BY_SCRIPT]'],
            'ng_lct_nearest_total_connections': nearest_sub_features['ng_lct_total_connections'],
            '[REDACTED_BY_SCRIPT]': nearest_sub_features['[REDACTED_BY_SCRIPT]'],
            '[REDACTED_BY_SCRIPT]': nearest_sub_features['ng_lct_pv_ratio'],
            '[REDACTED_BY_SCRIPT]': nearest_sub_features['ng_lct_ev_hp_ratio'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['mean', 'ng_lct_total_connections'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['mean', '[REDACTED_BY_SCRIPT]'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['mean', 'ng_lct_pv_ratio'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['mean', 'ng_lct_ev_hp_ratio'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['std', 'ng_lct_total_connections'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['std', '[REDACTED_BY_SCRIPT]'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['std', 'ng_lct_pv_ratio'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['std', 'ng_lct_ev_hp_ratio'],
        })
        all_solar_features.append(site_features)

    df_lct_features = pd.DataFrame(all_solar_features)
    # V6.9 FIX: Merge on the index of master_gdf (left) and the 'amaryllis_id' column of df_lct_features (right).
    master_gdf = master_gdf.merge(df_lct_features, left_index=True, right_on='amaryllis_id', how='left')

    # After the merge, 'amaryllis_id' from the right dataframe is now a column.
    # We should set the index back to 'amaryllis_id' to maintain consistency.
    if 'amaryllis_id' in master_gdf.columns:
        master_gdf = master_gdf.set_index('amaryllis_id')

    # Fill NaNs for the newly added columns
    new_cols = [col for col in df_lct_features.columns if col != 'amaryllis_id']
    master_gdf[new_cols] = master_gdf[new_cols].fillna(0)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf
