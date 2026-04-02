import logging
import geopandas as gpd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000, 20_000]

# Define paths to the L1 artifacts
CROW_ACCESS_L1 = Path(r"[REDACTED_BY_SCRIPT]")
CROW_S4_RCL_L1 = Path(r"[REDACTED_BY_SCRIPT]")
CROW_S15_L1 = Path(r"[REDACTED_BY_SCRIPT]")

# Define paths to the raw data for artifact creation
RAW_CROW_ACCESS = Path(r"[REDACTED_BY_SCRIPT]")
RAW_CROW_S4_RCL = Path(r"[REDACTED_BY_SCRIPT]")
RAW_CROW_S15 = Path(r"[REDACTED_BY_SCRIPT]")


def _create_l1_artifact(raw_path: Path, output_path: Path, layer_name: str):
    """[REDACTED_BY_SCRIPT]"""
    if output_path.exists():
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(raw_path)
    
    if gdf.crs.to_string() != PROJECT_CRS:
        gdf = gdf.to_crs(PROJECT_CRS)
        
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    
    invalid_geom_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_geom_count > 0:
        gdf['geometry'] = gdf.geometry.buffer(0)
        
    gdf.to_file(output_path, driver='GPKG', layer=layer_name)
    logging.info(f"[REDACTED_BY_SCRIPT]")

def _load_and_prepare_crow_data() -> tuple:
    """[REDACTED_BY_SCRIPT]'t exist."""
    _create_l1_artifact(RAW_CROW_ACCESS, CROW_ACCESS_L1, 'crow_access_england')
    _create_l1_artifact(RAW_CROW_S4_RCL, CROW_S4_RCL_L1, 'crow_s4_rcl_england')
    _create_l1_artifact(RAW_CROW_S15, CROW_S15_L1, 'crow_s15_england')

    access_gdf = gpd.read_file(CROW_ACCESS_L1)
    s4_rcl_gdf = gpd.read_file(CROW_S4_RCL_L1)
    s15_gdf = gpd.read_file(CROW_S15_L1)
    
    return access_gdf, s4_rcl_gdf, s15_gdf

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Integrates CRoW (Countryside and Rights of Way Act) features into the master GeoDataFrame.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Load Data ---
    access_gdf, s4_rcl_gdf, s15_gdf = _load_and_prepare_crow_data()
    
    if master_gdf.crs.to_string() != PROJECT_CRS:
        master_gdf = master_gdf.to_crs(PROJECT_CRS)

    # --- Primary Geometric Features (from Access Layer) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    joined_nearest = gpd.sjoin_nearest(master_gdf, access_gdf, how='left', distance_col='[REDACTED_BY_SCRIPT]')
    joined_nearest = joined_nearest[~joined_nearest.index.duplicated(keep='first')]
    master_gdf['[REDACTED_BY_SCRIPT]'] = joined_nearest['[REDACTED_BY_SCRIPT]']
    
    within_join = gpd.sjoin(master_gdf, access_gdf, how='left', predicate='within')
    is_within = within_join['index_right'].notna()
    is_within = is_within[~is_within.index.duplicated(keep='first')]
    master_gdf['crow_is_within'] = master_gdf.index.map(is_within).fillna(False).astype(int)

    # --- Multi-Radii Density Features ---
    master_gdf['temp_id'] = master_gdf.index
    for r in RADII_METERS:
        r_km = r // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")
        buffered_farms = master_gdf[['temp_id', 'geometry']].copy()
        buffered_farms['geometry'] = buffered_farms.buffer(r)
        
        intersections = gpd.overlay(buffered_farms, access_gdf, how='intersection', keep_geom_type=False)
        intersections['[REDACTED_BY_SCRIPT]'] = intersections.geometry.area / 1_000_000
        area_agg = intersections.groupby('temp_id')['[REDACTED_BY_SCRIPT]'].sum()
        
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = master_gdf['temp_id'].map(area_agg).fillna(0)

    # --- Synthetic Enrichment Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Find nearest access polygon and check its type
    if 'index_right' in joined_nearest.columns:
        nearest_indices = joined_nearest['index_right'].dropna().unique()
        if len(nearest_indices) > 0:
            nearest_access_polygons = access_gdf.loc[nearest_indices].copy()
            nearest_access_polygons['original_index'] = nearest_access_polygons.index
            
            rcl_intersection = gpd.sjoin(nearest_access_polygons, s4_rcl_gdf, how='left', predicate='intersects')
            s15_intersection = gpd.sjoin(nearest_access_polygons, s15_gdf, how='left', predicate='intersects')
            
            is_rcl = rcl_intersection.groupby('original_index')['index_right'].nunique() > 0
            is_s15 = s15_intersection.groupby('original_index')['index_right'].nunique() > 0
            
            master_gdf['crow_nearest_is_rcl'] = joined_nearest['index_right'].map(is_rcl).fillna(0).astype(int)
            master_gdf['crow_nearest_is_s15'] = joined_nearest['index_right'].map(is_s15).fillna(0).astype(int)
        else:
            master_gdf['crow_nearest_is_rcl'] = 0
            master_gdf['crow_nearest_is_s15'] = 0
    else:
        master_gdf['crow_nearest_is_rcl'] = 0
        master_gdf['crow_nearest_is_s15'] = 0


    # --- Landscape Character Ratio ---
    logging.info("[REDACTED_BY_SCRIPT]")
    buffer_5k = master_gdf[['temp_id', 'geometry']].copy()
    buffer_5k['geometry'] = buffer_5k.buffer(5000)
    
    rcl_in_5k_intersections = gpd.overlay(buffer_5k, s4_rcl_gdf, how='intersection', keep_geom_type=False)
    rcl_in_5k_intersections['rcl_area_sqkm'] = rcl_in_5k_intersections.geometry.area / 1_000_000
    rcl_area_agg = rcl_in_5k_intersections.groupby('temp_id')['rcl_area_sqkm'].sum()
    
    master_gdf['[REDACTED_BY_SCRIPT]'] = master_gdf['temp_id'].map(rcl_area_agg).fillna(0)
    
    total_crow_area_5km = master_gdf['[REDACTED_BY_SCRIPT]']
    rcl_area_5km = master_gdf['[REDACTED_BY_SCRIPT]']
    
    # Defensive division
    master_gdf['[REDACTED_BY_SCRIPT]'] = (rcl_area_5km / total_crow_area_5km * 100).where(total_crow_area_5km > 0, 0)
    master_gdf.drop(columns=['[REDACTED_BY_SCRIPT]'], inplace=True)

    # --- Cleanup and Finalization ---
    master_gdf.drop(columns=['temp_id'], inplace=True)
    master_gdf['[REDACTED_BY_SCRIPT]'].fillna(-1, inplace=True)

    logging.info("[REDACTED_BY_SCRIPT]")
    return master_gdf