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
ECR_SUB_1MW_PATH = r"[REDACTED_BY_SCRIPT]"
ECR_OVER_1MW_PATH = r"[REDACTED_BY_SCRIPT]"
ECR_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = 'EPSG:27700'

# Analysis parameters
RADII_METERS = [2000, 5000, 10000]
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

def get_or_create_l1_ecr_artifact(sub_1mw_path, over_1mw_path, cache_path):
    """[REDACTED_BY_SCRIPT]"""
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_sub1mw = gpd.read_file(sub_1mw_path).to_crs(TARGET_CRS)
    gdf_over1mw = gpd.read_file(over_1mw_path).to_crs(TARGET_CRS)

    gdf_sub1mw = clean_col_names(gdf_sub1mw, 'ukpn_ecr')
    gdf_over1mw = clean_col_names(gdf_over1mw, 'ukpn_ecr')

    gdf_sub1mw['capacity_scale'] = 'sub_1mw'
    gdf_over1mw['capacity_scale'] = 'over_1mw'

    ecr_master_gdf = pd.concat([gdf_sub1mw, gdf_over1mw], ignore_index=True)
    
    # The geometry column was renamed to 'ukpn_ecr_geometry' by clean_col_names
    # We must standardize it back to 'geometry' for the artifact contract
    if 'ukpn_ecr_geometry' in ecr_master_gdf.columns:
        ecr_master_gdf.rename(columns={'ukpn_ecr_geometry': 'geometry'}, inplace=True)
    
    ecr_master_gdf = gpd.GeoDataFrame(ecr_master_gdf, geometry='geometry', crs=TARGET_CRS)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    logging.info("[REDACTED_BY_SCRIPT]")
    capacity_cols = [c for c in ecr_master_gdf.columns if 'capacity_mw' in c or 'storage_mwh' in c]
    for col in capacity_cols:
        ecr_master_gdf[col] = pd.to_numeric(ecr_master_gdf[col], errors='coerce')

    ecr_master_gdf['[REDACTED_BY_SCRIPT]'] = ecr_master_gdf.filter(like='capacity_mw').sum(axis=1)
    ecr_master_gdf['total_storage_mwh'] = ecr_master_gdf.filter(like='storage_mwh').sum(axis=1)

    essential_cols = [
        'geometry', 'capacity_scale', 'ukpn_ecr_date_accepted', '[REDACTED_BY_SCRIPT]',
        'ukpn_ecr_energy_source_1', '[REDACTED_BY_SCRIPT]', 'total_storage_mwh'
    ]
    # Ensure all essential columns exist, adding missing ones with None
    for col in essential_cols:
        if col not in ecr_master_gdf.columns:
            ecr_master_gdf[col] = None
            
    gdf_distilled = ecr_master_gdf[essential_cols]

    gdf_filtered = gdf_distilled[gdf_distilled['[REDACTED_BY_SCRIPT]'].isin(['Connected', 'Accepted to Connect'])].copy()
    gdf_filtered['ukpn_ecr_date_accepted'] = pd.to_datetime(gdf_filtered['ukpn_ecr_date_accepted'], errors='coerce')
    gdf_filtered.dropna(subset=['ukpn_ecr_date_accepted'], inplace=True)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_filtered.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_filtered

def calculate_stratified_density_features(solar_site, ecr_sindex, gdf_ecr_final):
    """[REDACTED_BY_SCRIPT]"""
    features = {}
    site_geom = solar_site.geometry
    submission_date = solar_site.submission_date

    if pd.isna(submission_date):
        return pd.Series(features)

    for r in RADII_METERS:
        r_km = r // 1000
        possible_matches_index = list(ecr_sindex.intersection(site_geom.buffer(r).bounds))
        possible_matches = gdf_ecr_final.iloc[possible_matches_index]
        
        precise_spatial_matches = possible_matches[possible_matches.intersects(site_geom.buffer(r))]
        actual_matches = precise_spatial_matches[precise_spatial_matches['ukpn_ecr_date_accepted'] < submission_date]
        
        strata = {
            'sub1mw': actual_matches[actual_matches['capacity_scale'] == 'sub_1mw'],
            'over1mw': actual_matches[actual_matches['capacity_scale'] == 'over_1mw'],
        }
        
        for name, df_stratum in strata.items():
            count = len(df_stratum)
            capacity_mw = df_stratum['[REDACTED_BY_SCRIPT]'].sum()
            storage_mwh = df_stratum['total_storage_mwh'].sum()
            solar_count = df_stratum[df_stratum['ukpn_ecr_energy_source_1'] == 'Solar Photovoltaics'].shape[0]
            
            features[f'[REDACTED_BY_SCRIPT]'] = count
            features[f'[REDACTED_BY_SCRIPT]'] = capacity_mw
            features[f'[REDACTED_BY_SCRIPT]'] = storage_mwh
            features[f'[REDACTED_BY_SCRIPT]'] = solar_count / count if count > 0 else 0

        features[f'[REDACTED_BY_SCRIPT]'] = len(actual_matches)
        features[f'[REDACTED_BY_SCRIPT]'] = actual_matches['[REDACTED_BY_SCRIPT]'].sum()

    return pd.Series(features)

def execute(master_gdf):
    """
    Executor entry point for calculating UKPN Embedded Capacity Register (ECR) features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Critical CRS Check (Pattern 1 Defense)
    if master_gdf.crs is None:
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf.set_crs(TARGET_CRS, inplace=True)
    elif master_gdf.crs.to_string() != TARGET_CRS:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        master_gdf = master_gdf.to_crs(TARGET_CRS)

    if 'submission_date' not in master_gdf.columns:
        logging.error("FATAL: 'submission_date'[REDACTED_BY_SCRIPT]")
        return master_gdf
    
    master_gdf['submission_date'] = pd.to_datetime(master_gdf['submission_date'], errors='coerce')

    try:
        gdf_ecr_final = get_or_create_l1_ecr_artifact(ECR_SUB_1MW_PATH, ECR_OVER_1MW_PATH, ECR_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")
    ecr_sindex = gdf_ecr_final.sindex

    logging.info(f"[REDACTED_BY_SCRIPT]")
    tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
    der_features = master_gdf.progress_apply(
        calculate_stratified_density_features, 
        axis=1, 
        ecr_sindex=ecr_sindex, 
        gdf_ecr_final=gdf_ecr_final
    )

    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf_with_features = master_gdf.join(der_features)
    
    new_cols = der_features.columns
    master_gdf_with_features[new_cols] = master_gdf_with_features[new_cols].fillna(NULL_SENTINEL)

    # Artifact Validation: Ensure output remains a valid GeoDataFrame
    if not isinstance(master_gdf_with_features, gpd.GeoDataFrame):
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf_with_features = gpd.GeoDataFrame(master_gdf_with_features, geometry='geometry', crs=TARGET_CRS)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf_with_features