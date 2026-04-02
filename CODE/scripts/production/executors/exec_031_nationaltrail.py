
import logging
import geopandas as gpd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000, 20_000]
INPUT_GPKG = Path(r"[REDACTED_BY_SCRIPT]")
RAW_INPUT_GEOJSON = Path(r"[REDACTED_BY_SCRIPT]")

def _load_and_prepare_nt_data() -> gpd.GeoDataFrame:
    """
    Loads and prepares the National Trails data.
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
        
    # Standardize and distill schema
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    gdf = gdf[['name', 'geometry']].copy()
    gdf.rename(columns={'name': 'nt_name'}, inplace=True)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(INPUT_GPKG, driver='GPKG')
    
    return gdf

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Integrates National Trails features into the master GeoDataFrame.

    Args:
        master_gdf (gpd.GeoDataFrame): The master GeoDataFrame with solar farm locations.

    Returns:
        gpd.GeoDataFrame: The master GeoDataFrame with added National Trails features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Load and Prepare Data ---
    nt_gdf = _load_and_prepare_nt_data()
    
    if master_gdf.crs.to_string() != PROJECT_CRS:
        master_gdf = master_gdf.to_crs(PROJECT_CRS)

    # --- Proximity Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    joined_nearest = gpd.sjoin_nearest(master_gdf, nt_gdf, how='left', distance_col='nt_dist_to_nearest_m')
    joined_nearest = joined_nearest[~joined_nearest.index.duplicated(keep='first')]

    master_gdf['nt_dist_to_nearest_m'] = joined_nearest['nt_dist_to_nearest_m']
    master_gdf['nt_nearest_name'] = joined_nearest['nt_name']

    # --- Intersection Flag ---
    intersects_join = gpd.sjoin(master_gdf, nt_gdf, how='left', predicate='intersects')
    intersects_bool = intersects_join['index_right'].notna()
    intersects_bool = intersects_bool[~intersects_bool.index.duplicated(keep='first')]
    master_gdf['nt_intersects_site_bool'] = master_gdf.index.map(intersects_bool).fillna(False).astype(int)

    # --- Multi-Radii Density Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf['temp_id'] = master_gdf.index

    for r in RADII_METERS:
        r_km = r // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")

        buffered_farms = master_gdf[['temp_id', 'geometry']].copy()
        buffered_farms['geometry'] = buffered_farms.buffer(r)

        intersections = gpd.overlay(buffered_farms, nt_gdf, how='intersection', keep_geom_type=False)
        intersections['clipped_length_m'] = intersections.geometry.length

        length_agg = intersections.groupby('temp_id')['clipped_length_m'].sum()
        count_agg = intersections.groupby('temp_id')['nt_name'].nunique()

        master_gdf[f'nt_length_in_{r_km}km'] = length_agg / 1000  # Convert to km
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = count_agg

    master_gdf.drop(columns=['temp_id'], inplace=True)

    # --- Finalization and Cleanup ---
    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf['nt_dist_to_nearest_m'].fillna(-1, inplace=True)
    master_gdf['nt_nearest_name'].fillna('None', inplace=True)
    for r_km in [r // 1000 for r in RADII_METERS]:
        master_gdf[f'nt_length_in_{r_km}km'].fillna(0, inplace=True)
        master_gdf[f'[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)

    logging.info("[REDACTED_BY_SCRIPT]")
    return master_gdf
