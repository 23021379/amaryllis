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
L29_OUTPUT_PATH = r'[REDACTED_BY_SCRIPT]'

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
SOURCE_CRS_WGS84 = "EPSG:4326"

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

def forge_l1_primary_artifact(heatmap_path: str) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(heatmap_path, low_memory=False)
    df = clean_col_names(df)

    # The Critical Filter: Isolate the true primary substation population
    df['type'] = df['type'].str.strip().str.lower()
    df_primary = df[df['type'] == 'primary'].copy()
    logging.info(f"[REDACTED_BY_SCRIPT]'Primary'.")

    df_primary.dropna(subset=['longitude', 'latitude'], inplace=True)
    gdf = gpd.GeoDataFrame(
        df_primary,
        geometry=gpd.points_from_xy(df_primary.longitude, df_primary.latitude),
        crs=SOURCE_CRS_WGS84
    ).to_crs(TARGET_CRS)
    
    gdf.to_parquet(L1_PRIMARY_SUBS_AUTH_OUTPUT)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf

def forge_l1_distribution_artifact(op_path: str, loc_path: str) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df_op = pd.read_csv(op_path)
    df_loc = pd.read_csv(loc_path)
    df_op = clean_col_names(df_op)
    df_loc = clean_col_names(df_loc)

    gdf_fused = pd.merge(df_op, df_loc, on='substation_number', how='inner')
    
    gdf_fused.dropna(subset=['easting', 'northing'], inplace=True)
    gdf = gpd.GeoDataFrame(
        gdf_fused,
        geometry=gpd.points_from_xy(gdf_fused.easting, gdf_fused.northing),
        crs=TARGET_CRS
    ).to_crs(TARGET_CRS) # Paranoid re-projection
    
    gdf.to_parquet(L1_DIST_SUBS_AUTH_OUTPUT)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    try:
        # Phase 2: Forge the two authoritative L1 artifacts
        gdf_primary_subs = forge_l1_primary_artifact(PRIMARY_SUB_HEATMAP_INPUT)
        gdf_dist_subs = forge_l1_distribution_artifact(DIST_SUB_OPERATIONAL_INPUT, DIST_SUB_LOCATION_INPUT)
        
        # Load main solar data
        df_solar = pd.read_csv(SOLAR_SITES_INPUT)
        gdf_solar = gpd.GeoDataFrame(
            df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS
        )
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]"); sys.exit(1)

    # --- Phase 3: The Corrected Hierarchical Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # De-duplicate distribution artifact to count unique SITES, not transformers
    gdf_dist_sites = gdf_dist_subs.drop_duplicates(subset='substation_number').copy()
    
    # Link the hierarchy: Find the governing primary sub for each distribution site
    gdf_linked_sites = gpd.sjoin_nearest(gdf_dist_sites, gdf_primary_subs, how="inner")
    
    # MANDATED DE-DUPLICATION GUARD RAIL: Enforce one-to-one parent-child relationship.
    # Use the unique key of the 'child' (distribution sub) to de-duplicate.
    gdf_linked_sites = gdf_linked_sites.drop_duplicates(subset='substation_number', keep='first')

    # Group and Count
    downstream_counts = gdf_linked_sites.groupby('substationid').size()
    downstream_counts.name = '[REDACTED_BY_SCRIPT]'
    
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # First, link solar sites to their governing primary substation
    gdf_solar_linked = gpd.sjoin_nearest(gdf_solar, gdf_primary_subs, how="left")
    
    # MANDATED DE-DUPLICATION GUARD RAIL: Immediately cleanse join artifacts.
    if gdf_solar_linked.index.has_duplicates:
        duplicate_count = gdf_solar_linked.index.duplicated().sum()
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf_solar_linked = gdf_solar_linked[~gdf_solar_linked.index.duplicated(keep='first')]

    # Merge the calculated counts using the primary substation ID
    # Note: We merge with the original solar dataframe to avoid carrying over all columns from gdf_primary_subs
    df_final = pd.merge(
        gdf_solar_linked,
        downstream_counts,
        left_on='substationid',
        right_index=True,
        how='left'
    )
    
    # --- Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Drop geometry and intermediate join columns
    cols_to_drop = ['geometry', 'index_right', 'substationid'] + [c for c in df_final.columns if c in gdf_primary_subs.columns and c != 'geometry']
    df_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    if len(df_final) != len(gdf_solar):
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    df_final.to_csv(L29_OUTPUT_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()