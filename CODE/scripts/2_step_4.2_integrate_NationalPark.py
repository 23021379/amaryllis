import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000, 20_000]
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Helper Functions ---
def load_and_restore_l33_state(l33_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(l33_path, index_col='solar_farm_id')
    
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
    l33_path = Path(r"[REDACTED_BY_SCRIPT]")
    np_l1_path = Path(r"[REDACTED_BY_SCRIPT]")
    # This path is assumed from prior directives for the LPA context feature
    lpa_boundaries_path = Path(r"[REDACTED_BY_SCRIPT]")

    solar_farms_gdf = load_and_restore_l33_state(l33_path)
    np_gdf = gpd.read_file(np_l1_path)
    lpa_gdf = gpd.read_file(lpa_boundaries_path) # Needed for LPA context feature

    # --- Phase 2: Proximity & Direct Impact Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Use sjoin_nearest to find the single closest NP for each farm
    joined_nearest = gpd.sjoin_nearest(solar_farms_gdf, np_gdf, how='left', distance_col='np_dist_to_nearest_m')
    # Deduplicate to ensure one solar farm per row, keeping the first (and nearest) match
    joined_nearest = joined_nearest[~joined_nearest.index.duplicated(keep='first')]

    # Extract features
    solar_farms_gdf['np_dist_to_nearest_m'] = joined_nearest['np_dist_to_nearest_m']
    solar_farms_gdf['np_nearest_name'] = joined_nearest['name']
    # Calculate area from the geometry of the nearest NP
    solar_farms_gdf['np_nearest_area_sqkm'] = joined_nearest.geometry.area / 1_000_000

    # Generate `np_is_within` flag
    within_join = gpd.sjoin(solar_farms_gdf, np_gdf, how='left', predicate='within')
    is_within = within_join['index_right'].notna()
    is_within = is_within[~is_within.index.duplicated(keep='first')]
    solar_farms_gdf['np_is_within'] = solar_farms_gdf.index.map(is_within).fillna(False).astype(int)

    # --- Phase 3: Multi-Radii Density Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Add a unique ID to the solar farms to group by after the overlay
    solar_farms_gdf['solar_farm_id_col'] = solar_farms_gdf.index

    for r in RADII_METERS:
        r_km = r // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")

        # Create a temporary GeoDataFrame of buffered solar farm locations
        buffered_farms = solar_farms_gdf[['solar_farm_id_col', 'geometry']].copy()
        buffered_farms['geometry'] = buffered_farms.buffer(r)

        # Use overlay to find the true geometric intersection
        # This is the only architecturally sound way to calculate intersected area
        intersections = gpd.overlay(buffered_farms, np_gdf, how='intersection', keep_geom_type=False)

        # Calculate the area of the resulting intersection polygons
        intersections['[REDACTED_BY_SCRIPT]'] = intersections.geometry.area / 1_000_000

        # Aggregate the results back to the solar farm level
        # Sum the areas in case a buffer intersects multiple parks
        area_agg = intersections.groupby('solar_farm_id_col')['[REDACTED_BY_SCRIPT]'].sum()
        # Count the unique parks intersected
        count_agg = intersections.groupby('solar_farm_id_col')['name'].nunique()

        # Assign the new, correct features back to the main GeoDataFrame
        area_col = f'[REDACTED_BY_SCRIPT]'
        count_col = f'np_count_in_{r_km}km'
        
        solar_farms_gdf[area_col] = area_agg
        solar_farms_gdf[count_col] = count_agg

        # Ensure farms with no intersections get a value of 0, not NaN
        solar_farms_gdf[area_col].fillna(0, inplace=True)
        solar_farms_gdf[count_col].fillna(0, inplace=True)

    # Clean up the temporary ID column
    solar_farms_gdf.drop(columns=['solar_farm_id_col'], inplace=True)

    # --- Phase 4: LPA Context Feature ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # This logic assumes a pre-calculated lookup or can be calculated on the fly
    # For robustness, we calculate it here.
    lpa_gdf['lpa_area_sqkm'] = lpa_gdf.geometry.area / 1_000_000
    np_lpa_intersection = gpd.overlay(lpa_gdf, np_gdf, how='intersection')
    np_lpa_intersection['[REDACTED_BY_SCRIPT]'] = np_lpa_intersection.geometry.area / 1_000_000
    lpa_coverage = np_lpa_intersection.groupby('LPA23NM')['[REDACTED_BY_SCRIPT]'].sum()
    
    lpa_gdf = lpa_gdf.join(lpa_coverage)
    lpa_gdf['lpa_np_coverage_pct'] = (lpa_gdf['[REDACTED_BY_SCRIPT]'] / lpa_gdf['lpa_area_sqkm'] * 100).fillna(0)
    
    # Join this back to the solar farms
    solar_farms_with_lpa = gpd.sjoin(solar_farms_gdf, lpa_gdf[['LPA23NM', 'lpa_np_coverage_pct', 'geometry']], how='left', predicate='within')
    solar_farms_with_lpa = solar_farms_with_lpa[~solar_farms_with_lpa.index.duplicated(keep='first')]
    solar_farms_gdf['lpa_np_coverage_pct'] = solar_farms_with_lpa['lpa_np_coverage_pct']

    # --- Phase 5: Capstone Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Create a DataFrame of all constraint distances
    dist_df = solar_farms_gdf[['aw_dist_to_nearest_m', 'aonb_dist_to_nearest_m', 'sssi_dist_to_nearest_m', 'np_dist_to_nearest_m']].copy()
    
    # Update `env_min_dist_to_any_constraint_m`
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.min(axis=1)
    
    # Update `env_dominant_constraint_type`
    # Map column names to the required string values
    constraint_map = {
        'aw_dist_to_nearest_m': 'AncientWoodland',
        'aonb_dist_to_nearest_m': 'AONB',
        'sssi_dist_to_nearest_m': 'SSSI',
        'np_dist_to_nearest_m': 'NationalPark'
    }
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.idxmin(axis=1).map(constraint_map)

    # --- Phase 6: Final Assembly & Persistence ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Drop geometry column for CSV export
    final_df = solar_farms_gdf.drop(columns=['geometry'])
    
    # Fill NaNs with project-standard defaults
    fill_zero_cols = [col for col in final_df.columns if 'np_' in col and ('count' in col or 'area' in col or 'pct' in col)]
    final_df[fill_zero_cols] = final_df[fill_zero_cols].fillna(0)
    final_df['np_dist_to_nearest_m'].fillna(-1, inplace=True) # Use -1 for no nearby constraint
    final_df['np_nearest_name'].fillna('None', inplace=True)

    output_path = Path(r"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()