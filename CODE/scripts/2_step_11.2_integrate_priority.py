import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2_000, 5_000, 10_000, 20_000]
MAX_RADIUS_METERS = max(RADII_METERS)
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Helper Functions ---
def load_and_restore_l40_state(l40_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(l40_path, index_col='solar_farm_id')
    
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['easting_x'], df['northing_x']), crs=PROJECT_CRS
    )
    return gdf

# --- Main Orchestration ---
def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Phase 1: Load Inputs (Revised) ---
    l40_path = Path(r"[REDACTED_BY_SCRIPT]")
    phi_l2_path = Path(r"[REDACTED_BY_SCRIPT]")

    solar_farms_gdf = load_and_restore_l40_state(l40_path)
    # DO NOT load the 6GB file. We only need its path.
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- Phase 2: The Iterative Spatial Query Loop ---
    logging.info("[REDACTED_BY_SCRIPT]")
    results_list = []
    
    for solar_farm_id, farm_row in tqdm(solar_farms_gdf.iterrows(), total=len(solar_farms_gdf)):
        farm_geom = farm_row.geometry
        result_dict = {'solar_farm_id': solar_farm_id}

        # 1. Define the maximum query area
        query_bbox = farm_geom.buffer(MAX_RADIUS_METERS).bounds

        # 2. Execute a spatially filtered read from the GeoPackage ON DISK
        try:
            local_phi_gdf = gpd.read_file(phi_l2_path, bbox=query_bbox)
        except ValueError: # Handles cases where bbox doesn't intersect layer
            local_phi_gdf = gpd.GeoDataFrame([], columns=['is_priority', 'hab_group', 'geometry'], crs=PROJECT_CRS)

        # 3. Handle edge case of no nearby habitats
        if local_phi_gdf.empty:
            result_dict.update({
                'ph_dist_to_nearest_m': -1, 'ph_nearest_is_priority': 0,
                '[REDACTED_BY_SCRIPT]': 'None', 'ph_is_within': 0
            })
            for r in RADII_METERS:
                r_km = r // 1000
                result_dict[f'[REDACTED_BY_SCRIPT]'] = 0
                result_dict[f'[REDACTED_BY_SCRIPT]'] = 0
            results_list.append(result_dict)
            continue

        # 4. Perform Direct Impact & Characterization on the local subset
        farm_gdf_single = gpd.GeoDataFrame([farm_row], crs=PROJECT_CRS)
        
        # Distance
        result_dict['ph_dist_to_nearest_m'] = farm_geom.distance(local_phi_gdf.union_all())

        # Nearest attributes
        nearest_join = gpd.sjoin_nearest(farm_gdf_single, local_phi_gdf, how='left')
        result_dict['ph_nearest_is_priority'] = nearest_join['is_priority'].iloc[0]
        result_dict['[REDACTED_BY_SCRIPT]'] = nearest_join['hab_group'].iloc[0]

        # Within
        within_join = gpd.sjoin(farm_gdf_single, local_phi_gdf, how='left', predicate='within')
        result_dict['ph_is_within'] = int(not within_join['index_right'].isna().all())

        # 5. Perform Multi-Radii Density Calculations on the local subset
        for r in RADII_METERS:
            r_km = r // 1000
            buffer_geom = farm_geom.buffer(r)
            intersections = gpd.overlay(
                gpd.GeoDataFrame([{'geometry': buffer_geom}], crs=PROJECT_CRS),
                local_phi_gdf, how='intersection', keep_geom_type=False
            )
            
            if intersections.empty:
                total_area_ha = 0
                priority_area_ha = 0
            else:
                intersections['area_ha'] = intersections.geometry.area / 10_000
                total_area_ha = intersections['area_ha'].sum()
                priority_area_ha = intersections[intersections['is_priority'] == 1]['area_ha'].sum()

            result_dict[f'[REDACTED_BY_SCRIPT]'] = total_area_ha
            pct = (priority_area_ha / total_area_ha * 100) if total_area_ha > 0 else 0
            result_dict[f'[REDACTED_BY_SCRIPT]'] = pct

        results_list.append(result_dict)

    # --- Phase 3: Post-Loop Assembly ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_results = pd.DataFrame(results_list).set_index('solar_farm_id')
    solar_farms_gdf = solar_farms_gdf.join(df_results)

    # --- Phase 4: Capstone Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    dist_cols = solar_farms_gdf.columns[solar_farms_gdf.columns.str.contains('_dist_to_nearest_m')].tolist()
    constraint_map = {
        'aw_dist_to_nearest_m': 'AncientWoodland', 'aonb_dist_to_nearest_m': 'AONB',
        'np_dist_to_nearest_m': 'NationalPark', 'nt_dist_to_nearest_m': 'NationalTrail',
        'hp_dist_to_nearest_m': 'HistoricParkland', 'ph_dist_to_nearest_m': 'PriorityHabitat',
        'sssi_dist_to_nearest_m': 'SSSI', 'sac_dist_to_nearest_m': 'SAC',
        'spa_dist_to_nearest_m': 'SPA', '[REDACTED_BY_SCRIPT]': 'CRoW'
    }
    dist_df = solar_farms_gdf[list(constraint_map.keys())].copy()
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.min(axis=1)
    solar_farms_gdf['[REDACTED_BY_SCRIPT]'] = dist_df.idxmin(axis=1).map(constraint_map)

    # --- Phase 5: Final Assembly & Persistence ---
    logging.info("[REDACTED_BY_SCRIPT]")
    final_df = solar_farms_gdf.drop(columns=['geometry'])
    
    output_path = Path(r"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()