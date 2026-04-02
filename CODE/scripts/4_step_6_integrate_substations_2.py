import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
PRIMARY_SUB_HEATMAP_INPUT  = r"[REDACTED_BY_SCRIPT]"
DIST_SUB_OPERATIONAL_INPUT = r"[REDACTED_BY_SCRIPT]"
DIST_SUB_LOCATION_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_SITES_INPUT = r'[REDACTED_BY_SCRIPT]'

# Output Artifact
L1_PRIMARY_SUBS_AUTH_OUTPUT = r'[REDACTED_BY_SCRIPT]'
L1_DIST_SUBS_AUTH_OUTPUT = r'[REDACTED_BY_SCRIPT]'
L28_OUTPUT_PATH = r'[REDACTED_BY_SCRIPT]'

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
PRIMARY_SUB_PROXY_NEIGHBORS = 10 # Number of nearby subs to aggregate for primary features

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    new_cols = []
    for col in df.columns:
        new_col = col.lower().strip().replace(' ', '_')
        new_col = ''.join(c if c.isalnum() else '_' for c in new_col)
        while '__' in new_col: new_col = new_col.replace('__', '_')
        new_col = new_col.strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def create_l1_distribution_artifact(op_path: str, loc_path: str) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    df_op = pd.read_csv(op_path)
    df_loc = pd.read_csv(loc_path)
    df_op = clean_col_names(df_op)
    df_loc = clean_col_names(df_loc)

    op_count, loc_count = len(df_op), len(df_loc)
    gdf_fused = pd.merge(df_op, df_loc, on='substation_number', how='inner')
    logging.info(f"[REDACTED_BY_SCRIPT]")

    gdf_fused.dropna(subset=['easting', 'northing'], inplace=True)
    gdf = gpd.GeoDataFrame(
        gdf_fused, geometry=gpd.points_from_xy(gdf_fused.easting, gdf_fused.northing), crs=TARGET_CRS
    ).to_crs(TARGET_CRS)
    
    for col in ['[REDACTED_BY_SCRIPT]', 'day_md', 'night_md']:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

    # Pre-computation of individual utilization for downstream aggregation
    with np.errstate(divide='ignore', invalid='ignore'):
        gdf['dist_sub_utilisation_pct'] = np.where(
            gdf['[REDACTED_BY_SCRIPT]'] > 0,
            gdf['day_md'] / gdf['[REDACTED_BY_SCRIPT]'], np.nan
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
    
    # Use NumPy indexing for high-performance aggregation
    kva_values = gdf_dist_subs['[REDACTED_BY_SCRIPT]'].to_numpy()[indices_knn]
    util_values = gdf_dist_subs['dist_sub_utilisation_pct'].to_numpy()[indices_knn]
    
    agg_features = pd.DataFrame({
        '[REDACTED_BY_SCRIPT]': np.nansum(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': PRIMARY_SUB_PROXY_NEIGHBORS, # Renamed for parity
        '[REDACTED_BY_SCRIPT]': np.nanmean(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': np.nanmax(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': np.nanmin(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': np.nanvar(kva_values, axis=1),
        '[REDACTED_BY_SCRIPT]': np.nanmean(util_values, axis=1) # New feature
    }, index=gdf_solar.index)
    
    return agg_features

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    try:
        gdf_dist_subs = create_l1_distribution_artifact(DIST_SUB_OPERATIONAL_INPUT, DIST_SUB_LOCATION_INPUT)
        df_solar = pd.read_csv(SOLAR_SITES_INPUT)
        gdf_solar = gpd.GeoDataFrame(
            df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS
        )
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]"); sys.exit(1)

    # --- Direct Feature Synthesis (Secondary Substation) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_solar_with_sec = gpd.sjoin_nearest(gdf_solar, gdf_dist_subs, how="left", distance_col="dist_to_sec_sub_m")
    
    # De-duplicate in case of equidistant matches
    gdf_solar_with_sec = gdf_solar_with_sec[~gdf_solar_with_sec.index.duplicated(keep='first')]

    with np.errstate(divide='ignore', invalid='ignore'):
        gdf_solar_with_sec['[REDACTED_BY_SCRIPT]'] = np.where(
            gdf_solar_with_sec['[REDACTED_BY_SCRIPT]'] > 0,
            gdf_solar_with_sec['day_md'] / gdf_solar_with_sec['[REDACTED_BY_SCRIPT]'], np.nan
        )
    
    gdf_solar_with_sec.rename(columns={'[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]'}, inplace=True)
    
    # --- Aggregate Feature Synthesis (Primary Substation Proxy) ---
    df_primary_aggregates = synthesize_primary_sub_aggregates(gdf_solar, gdf_dist_subs)

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Select only the direct features we need to join
    sec_features_to_join = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    
    # Join the direct (secondary) and aggregated (primary) features to the original solar data
    gdf_final = gdf_solar.join(gdf_solar_with_sec[sec_features_to_join]).join(df_primary_aggregates)

    # --- Final Synthesis of Second-Order Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    with np.errstate(divide='ignore', invalid='ignore'):
        # Synthesize Substation Characterization Ratio
        gdf_final['kva_per_transformer'] = np.where(
            gdf_final['[REDACTED_BY_SCRIPT]'] > 0,
            gdf_final['[REDACTED_BY_SCRIPT]'] / gdf_final['[REDACTED_BY_SCRIPT]'], np.nan
        )
        # Synthesize Project-to-Grid Context Ratio
        # Assumes '[REDACTED_BY_SCRIPT]' exists in the source solar CSV
        if '[REDACTED_BY_SCRIPT]' in gdf_final.columns:
            gdf_final['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(gdf_final['[REDACTED_BY_SCRIPT]'], errors='coerce')
            gdf_final['[REDACTED_BY_SCRIPT]'] = np.where(
                gdf_final['[REDACTED_BY_SCRIPT]'] > 0,
                gdf_final['[REDACTED_BY_SCRIPT]'] / (gdf_final['[REDACTED_BY_SCRIPT]'] * 1000), np.nan
            )
        else:
            logging.warning("Column 'installed_capacity_mwelec' not found. Skipping 'kva_per_mwelec_ratio' synthesis.")


    # --- Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_final = pd.DataFrame(gdf_final.drop(columns='geometry', errors='ignore'))
    
    if len(df_final) != len(gdf_solar):
        logging.error(f"[REDACTED_BY_SCRIPT]"); sys.exit(1)

    df_final.to_csv(L28_OUTPUT_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()