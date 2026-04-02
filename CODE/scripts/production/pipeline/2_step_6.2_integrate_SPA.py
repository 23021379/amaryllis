import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000, 20_000]
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Helper Functions ---
def load_and_restore_l35_state(l35_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(l35_path, index_col='solar_farm_id')
    
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
    l35_path = Path(r"[REDACTED_BY_SCRIPT]")
    spa_l1_path = Path(r"[REDACTED_BY_SCRIPT]")
    lpa_boundaries_path = Path(r"[REDACTED_BY_SCRIPT]")

    solar_farms_gdf = load_and_restore_l35_state(l35_path)
    spa_gdf = gpd.read_file(spa_l1_path)
    lpa_gdf = gpd.read_file(lpa_boundaries_path)

    # --- Phase 2: Proximity & Direct Impact Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    joined_nearest = gpd.sjoin_nearest(solar_farms_gdf, spa_gdf, how='left', distance_col='spa_dist_to_nearest_m')
    joined_nearest = joined_nearest[~joined_nearest.index.duplicated(keep='first')]

    solar_farms_gdf['spa_dist_to_nearest_m'] = joined_nearest['spa_dist_to_nearest_m']
    solar_farms_gdf['spa_nearest_name'] = joined_nearest['spa_name']
    solar_farms_gdf['spa_nearest_area_ha'] = (joined_nearest.geometry.area / 10_000).fillna(0)

    within_join = gpd.sjoin(solar_farms_gdf, spa_gdf, how='left', predicate='within')
    is_within = within_join['index_right'].notna()
    is_within = is_within[~is_within.index.duplicated(keep='first')]
    solar_farms_gdf['spa_is_within'] = solar_farms_gdf.index.map(is_within).fillna(False).astype(int)

    # --- Phase 3: Multi-Radii Density Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    solar_farms_gdf['solar_farm_id_col'] = solar_farms_gdf.index

    for r in RADII_METERS:
        r_km = r // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")

        buffered_farms = solar_farms_gdf[['solar_farm_id_col', 'geometry']].copy()
        buffered_farms['geometry'] = buffered_farms.buffer(r)

        intersections = gpd.overlay(buffered_farms, spa_gdf, how='intersection', keep_geom_type=False)
        intersections['intersection_area_ha'] = intersections.geometry.area / 10_000

        area_agg = intersections.groupby('solar_farm_id_col')['intersection_area_ha'].sum()
        count_agg = intersections.groupby('solar_farm_id_col')['spa_name'].nunique()

        solar_farms_gdf[f'[REDACTED_BY_SCRIPT]'] = area_agg
        solar_farms_gdf[f'[REDACTED_BY_SCRIPT]'] = count_agg

    solar_farms_gdf.drop(columns=['solar_farm_id_col'], inplace=True)

    # Fill NaNs from no-intersection with 0 for count/area features
    for r in [2, 5, 10, 20]:
        solar_farms_gdf[f'[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)
        solar_farms_gdf[f'spa_count_in_{r}km'].fillna(0, inplace=True)

    # --- Phase 4: LPA Context Feature ---
    logging.info("[REDACTED_BY_SCRIPT]")
    lpa_gdf['lpa_area_sqkm'] = lpa_gdf.geometry.area / 1_000_000
    spa_lpa_intersection = gpd.overlay(lpa_gdf, spa_gdf, how='intersection')
    spa_lpa_intersection['[REDACTED_BY_SCRIPT]'] = spa_lpa_intersection.geometry.area / 1_000_000
    lpa_coverage = spa_lpa_intersection.groupby('LPA23NM')['[REDACTED_BY_SCRIPT]'].sum()
    
    lpa_gdf = lpa_gdf.join(lpa_coverage)
    lpa_gdf['lpa_spa_coverage_pct'] = (lpa_gdf['[REDACTED_BY_SCRIPT]'] / lpa_gdf['lpa_area_sqkm'] * 100).fillna(0)
    
    solar_farms_with_lpa = gpd.sjoin(solar_farms_gdf, lpa_gdf[['LPA23NM', 'lpa_spa_coverage_pct', 'geometry']], how='left', predicate='within')
    solar_farms_with_lpa = solar_farms_with_lpa[~solar_farms_with_lpa.index.duplicated(keep='first')]
    solar_farms_gdf['lpa_spa_coverage_pct'] = solar_farms_with_lpa['lpa_spa_coverage_pct']

    # --- Phase 5: Capstone Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    dist_cols = ['aw_dist_to_nearest_m', 'aonb_dist_to_nearest_m', 'sssi_dist_to_nearest_m', 'np_dist_to_nearest_m', 'sac_dist_to_nearest_m', 'spa_dist_to_nearest_m']
    dist_df = solar_farms_gdf[dist_cols].copy()
    
    # Update env_min_dist_to_any_constraint_m
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.min(axis=1)
    
    # Update env_dominant_constraint_type
    constraint_map = {
        'aw_dist_to_nearest_m': 'AncientWoodland',
        'aonb_dist_to_nearest_m': 'AONB',
        'sssi_dist_to_nearest_m': 'SSSI',
        'np_dist_to_nearest_m': 'NationalPark',
        'sac_dist_to_nearest_m': 'SAC',
        'spa_dist_to_nearest_m': 'SPA'
    }
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.idxmin(axis=1).map(constraint_map)

    # Create New Synthesis Feature: env_count_of_european_designations_in_5km
    logging.info("[REDACTED_BY_SCRIPT]")
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = \
        solar_farms_gdf['sac_count_in_5km'] + solar_farms_gdf['spa_count_in_5km']

    # --- Phase 6: Final Assembly & Persistence ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    final_df = solar_farms_gdf.drop(columns=['geometry'])
    
    # Final NaN handling for new features
    final_df['spa_dist_to_nearest_m'].fillna(-1, inplace=True)
    final_df['spa_nearest_name'].fillna('None', inplace=True)
    final_df['lpa_spa_coverage_pct'].fillna(0, inplace=True)

    output_path = Path(r"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()