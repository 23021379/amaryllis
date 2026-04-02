import logging
import geopandas as gpd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000]
BMV_GRADES = [1, 2, 3]
INPUT_GPKG = Path(r"[REDACTED_BY_SCRIPT]")
RAW_INPUT_GEOJSON = Path(r"[REDACTED_BY_SCRIPT]")

def _load_and_prepare_alc_data() -> gpd.GeoDataFrame:
    """
    Loads and prepares the Agricultural Land Classification (ALC) data.
    If a pre-processed GeoPackage exists, it'[REDACTED_BY_SCRIPT]'s created from the raw GeoJSON.
    """
    if INPUT_GPKG.exists():
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_file(INPUT_GPKG)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(RAW_INPUT_GEOJSON)
    
    # Standardize CRS
    if gdf.crs.to_string() != PROJECT_CRS:
        gdf = gdf.to_crs(PROJECT_CRS)
        
    # Standardize column names
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    
    # Repair invalid geometries
    invalid_geom_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_geom_count > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf['geometry'] = gdf.geometry.buffer(0)

    # Feature Enrichment
    gdf['alc_grade'] = gdf['alc_grade'].str.extract(r'(\d)').astype(float).astype('Int64')
    gdf['is_bmv'] = gdf['alc_grade'].isin(BMV_GRADES).astype(int)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(INPUT_GPKG, driver='GPKG')
    
    return gdf

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Integrates Agricultural Land Classification (ALC) features into the master GeoDataFrame.

    Args:
        master_gdf (gpd.GeoDataFrame): The master GeoDataFrame with solar farm locations.

    Returns:
        gpd.GeoDataFrame: The master GeoDataFrame with added ALC features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Load and Prepare Data ---
    alc_gdf = _load_and_prepare_alc_data()
    
    if master_gdf.crs.to_string() != PROJECT_CRS:
        master_gdf = master_gdf.to_crs(PROJECT_CRS)

    # --- Direct Site Classification ---
    logging.info("[REDACTED_BY_SCRIPT]")
    joined_within = gpd.sjoin(master_gdf, alc_gdf, how='left', predicate='within')
    joined_within = joined_within[~joined_within.index.duplicated(keep='first')]
    
    master_gdf['alc_grade_at_site'] = joined_within['alc_grade']
    master_gdf['alc_is_bmv_at_site'] = joined_within['is_bmv']

    # --- Proximity to Risk Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_bmv = alc_gdf[alc_gdf['is_bmv'] == 1].copy()
    gdf_grade1 = alc_gdf[alc_gdf['alc_grade'] == 1].copy()

    if not gdf_bmv.empty:
        joined_nearest_bmv = gpd.sjoin_nearest(master_gdf, gdf_bmv, how='left', distance_col='[REDACTED_BY_SCRIPT]')
        master_gdf['[REDACTED_BY_SCRIPT]'] = joined_nearest_bmv.loc[~joined_nearest_bmv.index.duplicated(keep='first')]['[REDACTED_BY_SCRIPT]']
    else:
        master_gdf['[REDACTED_BY_SCRIPT]'] = -1

    if not gdf_grade1.empty:
        joined_nearest_g1 = gpd.sjoin_nearest(master_gdf, gdf_grade1, how='left', distance_col='[REDACTED_BY_SCRIPT]')
        master_gdf['[REDACTED_BY_SCRIPT]'] = joined_nearest_g1.loc[~joined_nearest_g1.index.duplicated(keep='first')]['[REDACTED_BY_SCRIPT]']
    else:
        master_gdf['[REDACTED_BY_SCRIPT]'] = -1

    # --- Landscape Character Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf['temp_id'] = master_gdf.index

    for r in RADII_METERS:
        r_km = r // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")

        buffered_farms = master_gdf[['temp_id', 'geometry']].copy()
        buffered_farms['geometry'] = buffered_farms.buffer(r)
        buffered_farms['buffer_area'] = buffered_farms.geometry.area

        if not gdf_bmv.empty:
            intersections = gpd.overlay(buffered_farms, gdf_bmv, how='intersection', keep_geom_type=False)
            intersections['intersection_area'] = intersections.geometry.area
            bmv_area_in_buffer = intersections.groupby('temp_id')['intersection_area'].sum()
            
            merged = buffered_farms.set_index('temp_id').join(bmv_area_in_buffer)
            merged['intersection_area'].fillna(0, inplace=True)
            
            pct_col = f'[REDACTED_BY_SCRIPT]'
            master_gdf[pct_col] = (merged['intersection_area'] / merged['buffer_area']).values * 100
        else:
            master_gdf[f'[REDACTED_BY_SCRIPT]'] = 0


    master_gdf.drop(columns=['temp_id'], inplace=True)

    # --- Finalization and Cleanup ---
    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf['alc_grade_at_site'].fillna(-1, inplace=True)
    master_gdf['alc_is_bmv_at_site'].fillna(0, inplace=True)
    master_gdf['[REDACTED_BY_SCRIPT]'].fillna(-1, inplace=True)
    master_gdf['[REDACTED_BY_SCRIPT]'].fillna(-1, inplace=True)
    for r_km in [r // 1000 for r in RADII_METERS]:
        if f'[REDACTED_BY_SCRIPT]' in master_gdf.columns:
            master_gdf[f'[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)

    logging.info("[REDACTED_BY_SCRIPT]")
    return master_gdf