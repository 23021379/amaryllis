import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
PROJECT_CRS = "EPSG:27700"
SENTINEL_VALUE = -1.0

# --- Input Paths ---
# This path points to the DNO license area boundaries.
DNO_SOURCE_PATH = Path(r"[REDACTED_BY_SCRIPT]")

def _prepare_dno_boundaries() -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    dno_gdf = gpd.read_file(DNO_SOURCE_PATH)
    if dno_gdf.crs.to_string() != PROJECT_CRS:
        dno_gdf = dno_gdf.to_crs(PROJECT_CRS)
    
    dno_gdf.columns = dno_gdf.columns.str.lower()
    return dno_gdf[['dno', 'geometry']]

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    This executor performs DNO stratification on the input grid centroids.
    It spatially joins the master GeoDataFrame with DNO boundaries to assign a DNO to each point.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Load DNO Boundaries ---
    dno_boundaries = _prepare_dno_boundaries()

    # --- Stratify by DNO ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Perform a spatial join between the grid centroids and the DNO boundaries.
    # If both GeoDataFrames have a 'dno' column, sjoin will create 'dno_left' and 'dno_right'.
    # We are interested in the 'dno' from the boundaries, which will be 'dno_right'.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    stratified_gdf = gpd.sjoin(master_gdf, dno_boundaries, how='inner', predicate='within')

    # The 'dno' column from the right dataframe is what we need.
    # We can drop the index_right column that sjoin adds.
    if 'index_right' in stratified_gdf.columns:
        stratified_gdf = stratified_gdf.drop(columns=['index_right'])

    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # After sjoin, the dno column from the boundaries is named 'dno_right'
    dno_col_from_join = 'dno_right'
    if dno_col_from_join not in stratified_gdf.columns:
        # If there was no column conflict, it might just be 'dno'.
        if 'dno' in stratified_gdf.columns:
            dno_col_from_join = 'dno'
        else:
            raise KeyError(f"Column '{dno_col_from_join}' or 'dno'[REDACTED_BY_SCRIPT]")
        
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # Filter for only UKPN and NGED as per the original logic's focus
    stratified_gdf = stratified_gdf[stratified_gdf[dno_col_from_join].isin(['UKPN', 'NGED'])].copy()
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # The original script created a 'dno_source' column. We will rename the joined dno column to 'dno_source' for consistency.
    # Also drop the original 'dno_left' if it exists.
    if 'dno_left' in stratified_gdf.columns:
        stratified_gdf = stratified_gdf.drop(columns=['dno_left'])
    stratified_gdf.rename(columns={dno_col_from_join: 'dno_source'}, inplace=True)

    logging.info("[REDACTED_BY_SCRIPT]")
    return stratified_gdf
