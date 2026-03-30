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

# Input files and their publication dates
LTDS_SOURCE_FILES = {
    r"[REDACTED_BY_SCRIPT]": "2018-10-13"
}
LTDS_CACHE_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = 'EPSG:27700'

# Analysis parameters
RADIUS_METERS = 10000
NULL_SENTINEL = 0

def get_or_create_l1_ltds_artifact(source_files_dict, cache_path):
    """
    Loads the unified and time-stamped LTDS artifact from cache, or creates it.
    This function implements the Temporal Stamping Protocol from the original script.
    """
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf = gpd.read_parquet(cache_path)
        # Ensure date columns are in datetime format after loading from parquet
        gdf['[REDACTED_BY_SCRIPT]'] = pd.to_datetime(gdf['[REDACTED_BY_SCRIPT]'])
        return gdf

    logging.info("[REDACTED_BY_SCRIPT]")
    all_ltds_gdfs = []
    for path, pub_date in source_files_dict.items():
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf = gpd.read_file(path)
        
        gdf['[REDACTED_BY_SCRIPT]'] = pd.to_datetime(pub_date)
        
        # Enforce CRS transformation: assume WGS84 source, transform to BNG
        gdf = gdf.set_crs('EPSG:4326', allow_override=True).to_crs(TARGET_CRS)
        all_ltds_gdfs.append(gdf)

    if not all_ltds_gdfs:
        raise ValueError("[REDACTED_BY_SCRIPT]")

    gdf_ltds_master = pd.concat(all_ltds_gdfs, ignore_index=True)
    gdf_ltds_master = gpd.GeoDataFrame(gdf_ltds_master, geometry='geometry', crs=TARGET_CRS)
    
    gdf_ltds_master.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True)

    # Type casting for temporal features
    gdf_ltds_master['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(gdf_ltds_master['[REDACTED_BY_SCRIPT]'], errors='coerce')
    gdf_ltds_master.dropna(subset=['[REDACTED_BY_SCRIPT]'], inplace=True)
    gdf_ltds_master['[REDACTED_BY_SCRIPT]'] = gdf_ltds_master['[REDACTED_BY_SCRIPT]'].astype(int)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    gdf_ltds_master.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_ltds_master

def calculate_ltds_features_for_site(site_row, gdf_ltds_master):
    """
    Calculates all LTDS features for a single site, respecting the temporal guard protocol.
    """
    # Initialize features with null sentinel values
    feature_names = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    features = {name: NULL_SENTINEL for name in feature_names}
    
    submission_date = site_row['submission_date']
    if pd.isna(submission_date):
        return pd.Series(features)

    # Temporal Filter: Use only LTDS data published before the site's submission date
    valid_ltds = gdf_ltds_master[gdf_ltds_master['[REDACTED_BY_SCRIPT]'] <= submission_date]
    if valid_ltds.empty:
        return pd.Series(features) # No LTDS data existed at this time.
        
    # Select the most recent LTDS data available at that point in time
    latest_pub_date = valid_ltds['[REDACTED_BY_SCRIPT]'].max()
    point_in_time_ltds = valid_ltds[valid_ltds['[REDACTED_BY_SCRIPT]'] == latest_pub_date].copy()
    if point_in_time_ltds.empty:
        return pd.Series(features)

    # --- Proximity & Imminence Features ---
    site_geom = site_row.geometry
    point_in_time_ltds['distance'] = point_in_time_ltds.geometry.distance(site_geom)
    nearest = point_in_time_ltds.loc[point_in_time_ltds['distance'].idxmin()]
    
    features['[REDACTED_BY_SCRIPT]'] = nearest['distance']
    features['[REDACTED_BY_SCRIPT]'] = nearest['[REDACTED_BY_SCRIPT]']
    features['[REDACTED_BY_SCRIPT]'] = nearest['[REDACTED_BY_SCRIPT]'] - submission_date.year

    # --- Density & Character Features (10km radius) ---
    buffer = site_geom.buffer(RADIUS_METERS)
    upgrades_in_10km = point_in_time_ltds[point_in_time_ltds.intersects(buffer)]
    
    features['[REDACTED_BY_SCRIPT]'] = len(upgrades_in_10km)
    
    if not upgrades_in_10km.empty:
        asset_text = upgrades_in_10km['[REDACTED_BY_SCRIPT]'].str.lower().fillna('')
        features['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('[REDACTED_BY_SCRIPT]').sum()
        features['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('transformer|tx').sum()
        features['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('[REDACTED_BY_SCRIPT]').sum()
        
    return pd.Series(features)

def execute(master_gdf):
    """
    Executor entry point for integrating UKPN Long Term Development Statement (LTDS) features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    if 'submission_date' not in master_gdf.columns:
        logging.error("FATAL: 'submission_date'[REDACTED_BY_SCRIPT]")
        return master_gdf
    
    master_gdf['submission_date'] = pd.to_datetime(master_gdf['submission_date'], errors='coerce')
    
    try:
        gdf_ltds_master = get_or_create_l1_ltds_artifact(LTDS_SOURCE_FILES, LTDS_CACHE_PATH)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info(f"[REDACTED_BY_SCRIPT]")
    tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
    
    new_features_df = master_gdf.progress_apply(
        lambda row: calculate_ltds_features_for_site(row, gdf_ltds_master), 
        axis=1
    )

    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf_with_features = master_gdf.join(new_features_df)
    master_gdf_with_features[new_features_df.columns] = master_gdf_with_features[new_features_df.columns].fillna(NULL_SENTINEL)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf_with_features
