import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000]
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Helper Functions ---
def load_and_restore_l36_state(l36_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(l36_path, index_col='solar_farm_id')
    
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
    l36_path = Path(r"[REDACTED_BY_SCRIPT]")
    alc_l1_path = Path(r"[REDACTED_BY_SCRIPT]")

    solar_farms_gdf = load_and_restore_l36_state(l36_path)
    alc_gdf = gpd.read_file(alc_l1_path)

    # --- Phase 2: Direct Site Classification Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Since ALC is a "wallpaper", a 'within' join is appropriate and comprehensive.
    joined_within = gpd.sjoin(solar_farms_gdf, alc_gdf, how='left', predicate='within')
    # Keep only the first match in the rare case of overlapping polygons
    joined_within = joined_within[~joined_within.index.duplicated(keep='first')]
    
    solar_farms_gdf['alc_grade_at_site'] = joined_within['alc_grade']
    solar_farms_gdf['alc_is_bmv_at_site'] = joined_within['is_bmv']

    # --- Phase 3: Proximity to Risk Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Create subsets for efficient nearest joins
    gdf_bmv = alc_gdf[alc_gdf['is_bmv'] == 1].copy()
    gdf_grade1 = alc_gdf[alc_gdf['alc_grade'] == 1].copy()

    # Join to nearest BMV polygon
    joined_nearest_bmv = gpd.sjoin_nearest(solar_farms_gdf, gdf_bmv, how='left', distance_col='[REDACTED_BY_SCRIPT]')
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = joined_nearest_bmv.loc[~joined_nearest_bmv.index.duplicated(keep='first')]['[REDACTED_BY_SCRIPT]']

    # Join to nearest Grade 1 polygon
    joined_nearest_g1 = gpd.sjoin_nearest(solar_farms_gdf, gdf_grade1, how='left', distance_col='[REDACTED_BY_SCRIPT]')
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = joined_nearest_g1.loc[~joined_nearest_g1.index.duplicated(keep='first')]['[REDACTED_BY_SCRIPT]']

    # --- Phase 4: Landscape Character Features ---
    logging.info("[REDACTED_BY_SCRIPT]")
    solar_farms_gdf['solar_farm_id_col'] = solar_farms_gdf.index

    for r in RADII_METERS:
        r_km = r // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")

        buffered_farms = solar_farms_gdf[['solar_farm_id_col', 'geometry']].copy()
        buffered_farms['geometry'] = buffered_farms.buffer(r)
        buffered_farms['buffer_area'] = buffered_farms.geometry.area

        intersections = gpd.overlay(buffered_farms, gdf_bmv, how='intersection', keep_geom_type=False)
        intersections['intersection_area'] = intersections.geometry.area

        bmv_area_in_buffer = intersections.groupby('solar_farm_id_col')['intersection_area'].sum()
        
        # Merge back and calculate percentage
        merged = buffered_farms.set_index('solar_farm_id_col').join(bmv_area_in_buffer)
        merged['intersection_area'].fillna(0, inplace=True)
        
        pct_col = f'[REDACTED_BY_SCRIPT]'
        solar_farms_gdf[pct_col] = (merged['intersection_area'] / merged['buffer_area']).values

    solar_farms_gdf.drop(columns=['solar_farm_id_col'], inplace=True)

    # --- Phase 5: Final Assembly & Persistence ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    final_df = solar_farms_gdf.drop(columns=['geometry'])
    
    # Final NaN handling
    final_df['alc_grade_at_site'].fillna(-1, inplace=True) # Use -1 for sites outside any ALC polygon (should be rare)
    final_df['alc_is_bmv_at_site'].fillna(0, inplace=True)
    final_df['[REDACTED_BY_SCRIPT]'].fillna(-1, inplace=True)
    final_df['[REDACTED_BY_SCRIPT]'].fillna(-1, inplace=True)
    for r_km in [r // 1000 for r in RADII_METERS]:
        final_df[f'[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)

    output_path = Path(r"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()