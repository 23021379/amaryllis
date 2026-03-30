import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000, 20_000]
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Helper Functions ---
def load_and_restore_l38_state(l38_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(l38_path, index_col='solar_farm_id')
    
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['easting_x'], df['northing_x']), crs=PROJECT_CRS
    )
    return gdf

# --- Main Orchestration ---
def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Phase 1: Load Inputs ---
    l38_path = Path(r"[REDACTED_BY_SCRIPT]")
    access_l1_path = Path(r"[REDACTED_BY_SCRIPT]")
    s4_rcl_l1_path = Path(r"[REDACTED_BY_SCRIPT]")
    s15_l1_path = Path(r"[REDACTED_BY_SCRIPT]")

    solar_farms_gdf = load_and_restore_l38_state(l38_path)
    access_gdf = gpd.read_file(access_l1_path)
    s4_rcl_gdf = gpd.read_file(s4_rcl_l1_path)
    s15_gdf = gpd.read_file(s15_l1_path)

    # --- Phase 2: Primary Geometric Features (from Access Layer ONLY) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    joined_nearest = gpd.sjoin_nearest(solar_farms_gdf, access_gdf, how='left', distance_col='[REDACTED_BY_SCRIPT]')
    joined_nearest = joined_nearest[~joined_nearest.index.duplicated(keep='first')]
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = joined_nearest['[REDACTED_BY_SCRIPT]']
    
    within_join = gpd.sjoin(solar_farms_gdf, access_gdf, how='left', predicate='within')
    is_within = within_join['index_right'].notna()
    is_within = is_within[~is_within.index.duplicated(keep='first')]
    solar_farms_gdf['crow_is_within'] = solar_farms_gdf.index.map(is_within).fillna(False).astype(int)

    # Multi-radii density features
    solar_farms_gdf['solar_farm_id_col'] = solar_farms_gdf.index
    for r in RADII_METERS:
        r_km = r // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")
        buffered_farms = solar_farms_gdf[['solar_farm_id_col', 'geometry']].copy()
        buffered_farms['geometry'] = buffered_farms.buffer(r)
        intersections = gpd.overlay(buffered_farms, access_gdf, how='intersection', keep_geom_type=False)
        intersections['[REDACTED_BY_SCRIPT]'] = intersections.geometry.area / 1_000_000
        area_agg = intersections.groupby('solar_farm_id_col')['[REDACTED_BY_SCRIPT]'].sum()
        solar_farms_gdf[f'[REDACTED_BY_SCRIPT]'] = area_agg
        solar_farms_gdf[f'[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)

    # --- Phase 3: Synthetic Enrichment Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Identify the specific nearest access polygon for each farm
    nearest_access_polygons = access_gdf.loc[joined_nearest['index_right'].dropna().unique()].copy()
    nearest_access_polygons['original_index'] = nearest_access_polygons.index
    
    # Check if these nearest polygons intersect with S4 or S15 land
    rcl_intersection = gpd.sjoin(nearest_access_polygons, s4_rcl_gdf, how='left', predicate='intersects')
    s15_intersection = gpd.sjoin(nearest_access_polygons, s15_gdf, how='left', predicate='intersects')
    
    # Create flags
    is_rcl = rcl_intersection.groupby('original_index')['index_right'].nunique() > 0
    is_s15 = s15_intersection.groupby('original_index')['index_right'].nunique() > 0
    
    # Map flags back to solar farms via the nearest polygon index
    solar_farms_gdf['crow_nearest_is_rcl'] = joined_nearest['index_right'].map(is_rcl).fillna(0).astype(int)
    solar_farms_gdf['crow_nearest_is_s15'] = joined_nearest['index_right'].map(is_s15).fillna(0).astype(int)

    # Landscape character ratio (RCL % of CRoW land in 5km)
    logging.info("[REDACTED_BY_SCRIPT]")
    buffer_5k = solar_farms_gdf[['solar_farm_id_col', 'geometry']].copy()
    buffer_5k['geometry'] = buffer_5k.buffer(5000)
    rcl_in_5k_intersections = gpd.overlay(buffer_5k, s4_rcl_gdf, how='intersection', keep_geom_type=False)
    rcl_in_5k_intersections['rcl_area_sqkm'] = rcl_in_5k_intersections.geometry.area / 1_000_000
    rcl_area_agg = rcl_in_5k_intersections.groupby('solar_farm_id_col')['rcl_area_sqkm'].sum()
    
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = rcl_area_agg
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)
    
    # Calculate percentage, defensively handling division by zero
    total_crow_area = solar_farms_gdf['[REDACTED_BY_SCRIPT]']
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = (solar_farms_gdf['[REDACTED_BY_SCRIPT]'] / total_crow_area * 100).fillna(0)
    solar_farms_gdf.drop(columns=['[REDACTED_BY_SCRIPT]'], inplace=True)

    # --- Phase 4: Capstone Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    dist_cols = ['aw_dist_to_nearest_m', 'aonb_dist_to_nearest_m', 'sssi_dist_to_nearest_m', 'np_dist_to_nearest_m', 'sac_dist_to_nearest_m', 'spa_dist_to_nearest_m', 'nt_dist_to_nearest_m', '[REDACTED_BY_SCRIPT]']
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = solar_farms_gdf[dist_cols].min(axis=1)
    
    constraint_map = {
        'aw_dist_to_nearest_m': 'AncientWoodland', 'aonb_dist_to_nearest_m': 'AONB',
        'sssi_dist_to_nearest_m': 'SSSI', 'np_dist_to_nearest_m': 'NationalPark',
        'sac_dist_to_nearest_m': 'SAC', 'spa_dist_to_nearest_m': 'SPA',
        'nt_dist_to_nearest_m': 'NationalTrail', '[REDACTED_BY_SCRIPT]': 'CRoW'
    }
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = solar_farms_gdf[dist_cols].idxmin(axis=1).map(constraint_map)

    # --- Phase 5: Final Assembly & Persistence ---
    logging.info("[REDACTED_BY_SCRIPT]")
    final_df = solar_farms_gdf.drop(columns=['geometry', 'solar_farm_id_col'])
    
    # Final NaN handling
    final_df['[REDACTED_BY_SCRIPT]'].fillna(-1, inplace=True)

    output_path = Path(r"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()