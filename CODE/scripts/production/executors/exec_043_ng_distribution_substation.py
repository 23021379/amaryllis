import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
import os
from scipy.spatial import cKDTree
from tqdm import tqdm
from shapely import wkt

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
DIST_SUB_OPERATIONAL_INPUT = r"[REDACTED_BY_SCRIPT]"
DIST_SUB_LOCATION_INPUT = r"[REDACTED_BY_SCRIPT]"

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
PRIMARY_SUB_PROXY_NEIGHBORS = 10 # Number of nearby subs to aggregate for primary features

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

def create_l1_distribution_artifact(op_path: str, loc_path: str) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    df_op = pd.read_csv(op_path)
    df_loc = pd.read_csv(loc_path)
    
    df_op = clean_col_names(df_op, 'ng_dist')
    df_loc = clean_col_names(df_loc, 'ng_dist')

    gdf_fused = pd.merge(df_op, df_loc, on='[REDACTED_BY_SCRIPT]', how='inner')
    logging.info(f"[REDACTED_BY_SCRIPT]")

    gdf_fused.dropna(subset=['ng_dist_easting', 'ng_dist_northing'], inplace=True)
    gdf = gpd.GeoDataFrame(
        gdf_fused, 
        geometry=gpd.points_from_xy(gdf_fused.ng_dist_easting, gdf_fused.ng_dist_northing), 
        crs=TARGET_CRS
    )
    
    for col in ['[REDACTED_BY_SCRIPT]', 'ng_dist_day_md', 'ng_dist_night_md']:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

    with np.errstate(divide='ignore', invalid='ignore'):
        gdf['[REDACTED_BY_SCRIPT]'] = np.where(
            gdf['[REDACTED_BY_SCRIPT]'] > 0,
            gdf['ng_dist_day_md'] / gdf['[REDACTED_BY_SCRIPT]'], np.nan
        )

    logging.info("[REDACTED_BY_SCRIPT]")
    return gdf

def synthesize_primary_sub_aggregates(gdf_solar: gpd.GeoDataFrame, gdf_dist_subs: gpd.GeoDataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    dist_coords = np.array(list(gdf_dist_subs.geometry.apply(lambda p: (p.x, p.y))))
    solar_coords = np.array(list(gdf_solar.geometry.apply(lambda p: (p.x, p.y))))
    spatial_index = cKDTree(dist_coords)
    
    _, indices_knn = spatial_index.query(solar_coords, k=PRIMARY_SUB_PROXY_NEIGHBORS)
    
    kva_values = gdf_dist_subs['[REDACTED_BY_SCRIPT]'].to_numpy()[indices_knn]
    util_values = gdf_dist_subs['[REDACTED_BY_SCRIPT]'].to_numpy()[indices_knn]
    
    agg_features = pd.DataFrame({
        '[REDACTED_BY_SCRIPT]': np.nansum(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': PRIMARY_SUB_PROXY_NEIGHBORS,
        '[REDACTED_BY_SCRIPT]': np.nanmean(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': np.nanmax(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': np.nanmin(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': np.nanvar(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': np.nanmean(util_values, axis=1)
    }, index=gdf_solar.index)
    
    return agg_features

def execute(master_gdf):
    """
    Executor entry point for calculating National Grid distribution substation features.
    """
    log_prefix = "[REDACTED_BY_SCRIPT]"
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- V6.1 Resilience: Ensure input is a GeoDataFrame ---
    if not isinstance(master_gdf, gpd.GeoDataFrame):
        if 'geometry' in master_gdf.columns:
            try:
                master_gdf['geometry'] = master_gdf['geometry'].apply(wkt.loads)
                master_gdf = gpd.GeoDataFrame(master_gdf, geometry='geometry', crs=TARGET_CRS)
                logging.info("[REDACTED_BY_SCRIPT]")
            except Exception as e:
                logging.error(f"Failed to convert 'geometry'[REDACTED_BY_SCRIPT]")
                # Fallback or raise depending on desired strictness
                return master_gdf 
        else:
            logging.error("[REDACTED_BY_SCRIPT]'geometry' column.")
            raise ValueError(f"[REDACTED_BY_SCRIPT]'geometry' column.")

    try:
        gdf_dist_subs = create_l1_distribution_artifact(DIST_SUB_OPERATIONAL_INPUT, DIST_SUB_LOCATION_INPUT)
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    # --- Direct Feature Synthesis (Secondary Substation) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_joined = gpd.sjoin_nearest(master_gdf, gdf_dist_subs, how="left", distance_col="ng_dist_to_sec_sub_m")
    
    # De-duplicate in case of equidistant matches
    # V6.3: Use the authoritative key 'hex_id' for deduplication.
    gdf_joined = gdf_joined.drop_duplicates(subset='hex_id', keep='first')
    logging.info(f"Deduplicated on 'hex_id'[REDACTED_BY_SCRIPT]")

    # Rename and calculate direct features
    gdf_joined.rename(columns={'[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]'}, inplace=True)
    gdf_joined['[REDACTED_BY_SCRIPT]'] = np.where(
        gdf_joined['[REDACTED_BY_SCRIPT]'] > 0,
        gdf_joined['ng_dist_day_md'] / gdf_joined['[REDACTED_BY_SCRIPT]'], np.nan
    )
    
    # --- Aggregate Feature Synthesis (Primary Substation Proxy) ---
    df_primary_aggregates = synthesize_primary_sub_aggregates(master_gdf, gdf_dist_subs)

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Select only the direct features we need to join from the sjoin result
    sec_features_to_join = gdf_joined[[
        'hex_id', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'ng_dist_to_sec_sub_m'
    ]]
    
    # Merge direct (secondary) and aggregated (primary) features back to the master dataframe
    master_gdf = master_gdf.merge(sec_features_to_join, on='hex_id', how='left')
    master_gdf = master_gdf.merge(df_primary_aggregates, left_index=True, right_index=True, how='left')

    # --- Final Synthesis of Second-Order Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    with np.errstate(divide='ignore', invalid='ignore'):
        master_gdf['[REDACTED_BY_SCRIPT]'] = np.where(
            master_gdf['[REDACTED_BY_SCRIPT]'] > 0,
            master_gdf['[REDACTED_BY_SCRIPT]'] / master_gdf['[REDACTED_BY_SCRIPT]'], np.nan
        )
        if '[REDACTED_BY_SCRIPT]' in master_gdf.columns:
            master_gdf['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(master_gdf['[REDACTED_BY_SCRIPT]'], errors='coerce')
            master_gdf['[REDACTED_BY_SCRIPT]'] = np.where(
                master_gdf['[REDACTED_BY_SCRIPT]'] > 0,
                master_gdf['[REDACTED_BY_SCRIPT]'] / (master_gdf['[REDACTED_BY_SCRIPT]'] * 1000), np.nan
            )
        else:
            logging.warning("Column 'installed_capacity_mwelec' not found. Skipping 'ng_kva_per_mwelec_ratio' synthesis.")

    # Fill NaNs created by merges/calculations
    new_cols = list(sec_features_to_join.columns) + list(df_primary_aggregates.columns)
    if 'hex_id' in new_cols:
        new_cols.remove('hex_id')
    for col in new_cols:
        if col in master_gdf.columns:
            master_gdf[col] = master_gdf[col].fillna(0)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf
