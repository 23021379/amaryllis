import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000, 20_000]
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Helper Functions ---
def load_and_restore_l37_state(l37_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(l37_path, index_col='solar_farm_id')
    
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
    l37_path = Path(r"[REDACTED_BY_SCRIPT]")
    nt_l1_path = Path(r"[REDACTED_BY_SCRIPT]")

    solar_farms_gdf = load_and_restore_l37_state(l37_path)
    nt_gdf = gpd.read_file(nt_l1_path)

    # --- Phase 2: Proximity Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    joined_nearest = gpd.sjoin_nearest(solar_farms_gdf, nt_gdf, how='left', distance_col='nt_dist_to_nearest_m')
    joined_nearest = joined_nearest[~joined_nearest.index.duplicated(keep='first')]

    solar_farms_gdf['nt_dist_to_nearest_m'] = joined_nearest['nt_dist_to_nearest_m']
    solar_farms_gdf['nt_nearest_name'] = joined_nearest['nt_name']

    # Generate `nt_is_within` (correctly, `nt_intersects`) flag
    intersects_join = gpd.sjoin(solar_farms_gdf, nt_gdf, how='left', predicate='intersects')
    intersects_bool = intersects_join['index_right'].notna()
    intersects_bool = intersects_bool[~intersects_bool.index.duplicated(keep='first')]
    solar_farms_gdf['nt_intersects_site_bool'] = solar_farms_gdf.index.map(intersects_bool).fillna(False).astype(int)


    # --- Phase 3: Multi-Radii Density Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    solar_farms_gdf['solar_farm_id_col'] = solar_farms_gdf.index

    for r in RADII_METERS:
        r_km = r // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")

        buffered_farms = solar_farms_gdf[['solar_farm_id_col', 'geometry']].copy()
        buffered_farms['geometry'] = buffered_farms.buffer(r)

        intersections = gpd.overlay(buffered_farms, nt_gdf, how='intersection', keep_geom_type=False)
        
        # Calculate length of the clipped line segments
        intersections['clipped_length_m'] = intersections.geometry.length

        # Aggregate results back to the solar farm level
        length_agg = intersections.groupby('solar_farm_id_col')['clipped_length_m'].sum()
        count_agg = intersections.groupby('solar_farm_id_col')['nt_name'].nunique()

        # Assign new features, converting length to km as per directive
        solar_farms_gdf[f'nt_length_in_{r_km}km'] = length_agg / 1000
        solar_farms_gdf[f'[REDACTED_BY_SCRIPT]'] = count_agg

    solar_farms_gdf.drop(columns=['solar_farm_id_col'], inplace=True)

    # Fill NaNs from no-intersection with 0 for count/length features
    for r_km in [r // 1000 for r in RADII_METERS]:
        solar_farms_gdf[f'nt_length_in_{r_km}km'].fillna(0, inplace=True)
        solar_farms_gdf[f'[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)

    # --- Phase 4: Capstone Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    dist_cols = ['aw_dist_to_nearest_m', 'aonb_dist_to_nearest_m', 'sssi_dist_to_nearest_m', 
                 'np_dist_to_nearest_m', 'sac_dist_to_nearest_m', 'spa_dist_to_nearest_m', 
                 'nt_dist_to_nearest_m']
    dist_df = solar_farms_gdf[dist_cols].copy()
    
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.min(axis=1)
    
    constraint_map = {
        'aw_dist_to_nearest_m': 'AncientWoodland', 'aonb_dist_to_nearest_m': 'AONB',
        'sssi_dist_to_nearest_m': 'SSSI', 'np_dist_to_nearest_m': 'NationalPark',
        'sac_dist_to_nearest_m': 'SAC', 'spa_dist_to_nearest_m': 'SPA',
        'nt_dist_to_nearest_m': 'NationalTrail'
    }
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.idxmin(axis=1).map(constraint_map)

    # --- Phase 5: Final Assembly & Persistence ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    final_df = solar_farms_gdf.drop(columns=['geometry'])
    
    # Final NaN handling for new features
    final_df['nt_dist_to_nearest_m'].fillna(-1, inplace=True)
    final_df['nt_nearest_name'].fillna('None', inplace=True)

    output_path = Path(r"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()