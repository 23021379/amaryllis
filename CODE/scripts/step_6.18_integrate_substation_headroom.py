import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
SERVICE_AREA_INPUT = r"[REDACTED_BY_SCRIPT]"
L23_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output Artifact
L24_DATA_OUTPUT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
TARGET_CRS = "EPSG:27700"


def parse_and_clean_service_areas(gdf_raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Executes the robust parsing and decontamination protocol for the raw service area data.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # MANDATE: Proactive Schema Normalization.
    gdf_raw.columns = [
        col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') 
        for col in gdf_raw.columns
    ]
    
    # MANDATE: Robust Parsing of string-formatted percentage column.
    def parse_pct(series):
        # This function is retained as the percentage column is confirmed to be a string.
        return pd.to_numeric(series.str.extract(r'(\d+\.?\d*)')[0], errors='coerce') / 100.0

    # The MW columns are already numeric; simply rename them to the project standard.
    # The traceback confirms the post-normalization key is 'firmcapacitywinter'.
    gdf_raw.rename(columns={
        'firmcapacitywinter': '[REDACTED_BY_SCRIPT]',
        'firmcapacitysummer': '[REDACTED_BY_SCRIPT]'
    }, inplace=True)
    
    # Apply the necessary parsing only to the column that requires it.
    gdf_raw['[REDACTED_BY_SCRIPT]'] = gdf_raw['demand']/100.0  #'demand' is already numeric
    
    # MANDATE: One-hot encode categorical features with descriptive prefixes.
    gdf_raw = pd.get_dummies(gdf_raw, columns=['demandrag'], prefix='demandrag', dtype=int)
    gdf_raw = pd.get_dummies(gdf_raw, columns=['seasonofconstraint'], prefix='constraint_season', dtype=int)
    
    # Schema Pruning: Select only the essential, cleaned columns for the final artifact.
    essential_cols = [
        'geometry', 'primary_site_floc', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ] + [col for col in gdf_raw.columns if 'demandrag_' in col or 'constraint_season_' in col]
    
    # Defensive check to ensure all essential columns exist before returning
    final_cols = [col for col in essential_cols if col in gdf_raw.columns]
    return gdf_raw[final_cols]


def main():
    """
    Main function to execute the fault-tolerant containment protocol.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest, Decontaminate, and Prepare L1 Artifact ---
    try:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf_service_areas_raw = gpd.read_file(SERVICE_AREA_INPUT)
        
        # MANDATE: Uncompromising CRS Unification.
        gdf_service_areas_raw = gdf_service_areas_raw.to_crs(TARGET_CRS)
        
        # Execute the robust parsing protocol.
        gdf_service_areas_l1 = parse_and_clean_service_areas(gdf_service_areas_raw)
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load main solar dataset
        df_l23 = pd.read_csv(L23_DATA_INPUT)
        gdf_solar = gpd.GeoDataFrame(
            df_l23, geometry=gpd.points_from_xy(df_l23.easting_x, df_l23.northing_x), crs=TARGET_CRS
        )
        # Create a stable key for duplicate resolution.
        gdf_solar['solar_farm_id'] = gdf_solar.index
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- Phase 2: The Two-Stage, Fault-Tolerant Spatial Join (Amaryllis Alpha) ---
    logging.info("[REDACTED_BY_SCRIPT]'within')...")
    gdf_joined_raw = gpd.sjoin(gdf_solar, gdf_service_areas_l1, how='left', predicate='within')

    # MANDATE: Install Duplicate Resolution Gate to prevent data corruption from overlapping polygons.
    gdf_joined = gdf_joined_raw.drop_duplicates(subset=['solar_farm_id'], keep='first')
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Identify successes and orphans
    success_mask = gdf_joined['index_right'].notna()
    orphan_mask = ~success_mask
    
    gdf_joined['[REDACTED_BY_SCRIPT]'] = np.nan
    gdf_joined.loc[success_mask, '[REDACTED_BY_SCRIPT]'] = 'within'
    
    if orphan_mask.any():
        logging.warning(f"[REDACTED_BY_SCRIPT]'nearest')...")
        
        # Create a list of attribute columns from the right frame, explicitly excluding its geometry column.
        cols_to_drop = [col for col in gdf_service_areas_l1.columns if col != 'geometry'] + ['index_right']
        
        # Isolate orphans, dropping only the non-geometric columns from the failed join.
        # This preserves the geometry column, ensuring 'orphans' remains a GeoDataFrame.
        orphans = gdf_joined[orphan_mask].drop(columns=cols_to_drop)
        fallback_join = gpd.sjoin_nearest(orphans, gdf_service_areas_l1, how='left')
        
        # Re-integrate the fallback results into the main dataframe
        gdf_joined.update(fallback_join)
        gdf_joined.loc[orphan_mask, '[REDACTED_BY_SCRIPT]'] = 'nearest'
    else:
        logging.info("[REDACTED_BY_SCRIPT]")

    # Verification: Ensure no data loss
    assert len(gdf_joined) == len(gdf_solar), "[REDACTED_BY_SCRIPT]"
    assert gdf_joined['[REDACTED_BY_SCRIPT]'].notna().all(), "[REDACTED_BY_SCRIPT]"
    
    # --- Phase 3: Feature Generation ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # 1. Distance to Governing Substation Centroid
    # We need to map the polygon geometry back for this calculation
    governing_area_geoms_series = gdf_joined['index_right'].map(gdf_service_areas_l1.geometry)
    
    # MANDATE: Explicitly cast the pandas.Series of geometries to a GeoSeries to access geospatial properties.
    governing_area_geoseries = gpd.GeoSeries(governing_area_geoms_series, crs=TARGET_CRS)
    
    gdf_joined['[REDACTED_BY_SCRIPT]'] = gdf_joined.geometry.distance(governing_area_geoseries.centroid) / 1000.0
    
    # 2. Forecast vs. Empirical Headroom Delta
    # Defensive check for required columns from previous directives
    if '[REDACTED_BY_SCRIPT]' in gdf_joined.columns and '[REDACTED_BY_SCRIPT]' in gdf_joined.columns:
        forecast_headroom_pct = (gdf_joined['[REDACTED_BY_SCRIPT]'] * 1000) / gdf_joined['[REDACTED_BY_SCRIPT]']
        gdf_joined['[REDACTED_BY_SCRIPT]'] = gdf_joined['[REDACTED_BY_SCRIPT]'] - forecast_headroom_pct
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")

    # 3. Governing Substation ID Mismatch
    if '[REDACTED_BY_SCRIPT]' in gdf_joined.columns: # Example column name
        gdf_joined['[REDACTED_BY_SCRIPT]'] = (
            gdf_joined['[REDACTED_BY_SCRIPT]'] != gdf_joined['primary_site_floc']
        ).astype(int)
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")

    # --- Finalization ---
    logging.info("Finalizing artifact...")
    
    # Drop intermediate and redundant columns
    cols_to_drop = ['index_right', 'geometry']
    df_final = gdf_joined.drop(columns=cols_to_drop, errors='ignore')
    
    df_final.to_csv(L24_DATA_OUTPUT, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()