import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')


# Inputs for this directive
OHL_POINTS_INPUT = r"[REDACTED_BY_SCRIPT]"
OHL_LINES_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
LOCAL_NETWORK_BUFFER_METERS = 2000

from typing import Tuple
def load_and_stratify_ohl_structures(filepath: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads, re-projects, and stratifies the OHL structure data into poles and towers.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_structures = gpd.read_file(filepath)

    # MANDATE: Unconditional CRS Unification.
    gdf_structures = gdf_structures.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Schema Normalization.
    gdf_structures.columns = [col.lower().strip() for col in gdf_structures.columns]

    # MANDATE: Stratify by Visual Impact.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_poles = gdf_structures[gdf_structures['ob_class'].str.contains("pole", case=False, na=False)].copy()
    gdf_towers = gdf_structures[gdf_structures['ob_class'].str.contains("tower", case=False, na=False)].copy()
    
    return gdf_structures, gdf_poles, gdf_towers


def calculate_visual_impact_features(gdf_solar: gpd.GeoDataFrame, gdf_structures: gpd.GeoDataFrame, gdf_poles: gpd.GeoDataFrame, gdf_towers: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculates specific proximity and local network character features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Specific Proximity Features ---
    gdf_solar_with_poles = gpd.sjoin_nearest(gdf_solar, gdf_poles, distance_col="[REDACTED_BY_SCRIPT]", how="left")
    gdf_solar_with_poles = gdf_solar_with_poles.drop_duplicates(subset='solar_farm_id', keep='first')

    gdf_solar_with_towers = gpd.sjoin_nearest(gdf_solar, gdf_towers, distance_col="[REDACTED_BY_SCRIPT]", how="left")
    gdf_solar_with_towers = gdf_solar_with_towers.drop_duplicates(subset='solar_farm_id', keep='first')

    # --- Local Network Character Features ---
    gdf_solar_buffers = gdf_solar.copy()
    gdf_solar_buffers['geometry'] = gdf_solar_buffers.geometry.buffer(LOCAL_NETWORK_BUFFER_METERS)
    
    # Join buffers with all structures to get total count
    structures_in_buffer = gpd.sjoin(gdf_structures, gdf_solar_buffers, how="inner", predicate="within")
    structure_counts = structures_in_buffer.groupby('solar_farm_id').size().reset_index(name='ohl_local_structure_count')
    
    # Join buffers with towers to get tower count
    towers_in_buffer = gpd.sjoin(gdf_towers, gdf_solar_buffers, how="inner", predicate="within")
    tower_counts = towers_in_buffer.groupby('solar_farm_id').size().reset_index(name='ohl_local_tower_count')

    # --- Combine all features ---
    df_features = gdf_solar[['solar_farm_id']].copy()
    df_features = df_features.merge(gdf_solar_with_poles[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(gdf_solar_with_towers[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
    df_features = df_features.merge(structure_counts, on='solar_farm_id', how='left')
    df_features = df_features.merge(tower_counts, on='solar_farm_id', how='left')
    
    # MANDATE: Calculate the Alpha ratio feature with smoothing.
    df_features['ohl_local_tower_ratio'] = df_features['ohl_local_tower_count'] / (df_features['ohl_local_structure_count'] + 1)
    
    return df_features


def calculate_reconciliation_feature(gdf_solar: gpd.GeoDataFrame, gdf_structures: gpd.GeoDataFrame, ohl_lines_path: str) -> pd.DataFrame:
    """
    Calculates the average distance between nearest structures and the OHL line data.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_ohl_lines = gpd.read_file(ohl_lines_path).to_crs("EPSG:27700")
    
    # Create spatial trees for efficient queries
    structures_tree = BallTree(np.array(list(zip(gdf_structures.geometry.x, gdf_structures.geometry.y))))
    
    results = []
    for index, solar_app in gdf_solar.iterrows():
        app_coords = [[solar_app.geometry.x, solar_app.geometry.y]]
        
        # Find 10 nearest structures
        distances, indices = structures_tree.query(app_coords, k=min(10, len(gdf_structures)))
        nearest_structures = gdf_structures.iloc[indices[0]]
        
        # For each structure, find distance to nearest line
        distances_to_line = nearest_structures.geometry.apply(lambda p: gdf_ohl_lines.distance(p).min())
        
        avg_dist = distances_to_line.mean() if not distances_to_line.empty else np.nan
        
        results.append({
            'solar_farm_id': solar_app['solar_farm_id'],
            '[REDACTED_BY_SCRIPT]': avg_dist
        })
        
    return pd.DataFrame(results)


def main():
    """
    Main function to execute the OHL point data synthesis pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest and Stratify ---
    gdf_structures, gdf_poles, gdf_towers = load_and_stratify_ohl_structures(OHL_POINTS_INPUT)
    
    # --- Load Solar Data ---
    df_solar_l19 = pd.read_csv(SOLAR_DATA_INPUT)
    gdf_solar = gpd.GeoDataFrame(
        df_solar_l19, geometry=gpd.points_from_xy(df_solar_l19.easting_x, df_solar_l19.northing_x), crs="EPSG:27700"
    )
    gdf_solar['solar_farm_id'] = gdf_solar.index

    # --- Execute Feature Generation Phases ---
    visual_features = calculate_visual_impact_features(gdf_solar, gdf_structures, gdf_poles, gdf_towers)
    reconciliation_feature = calculate_reconciliation_feature(gdf_solar, gdf_structures, OHL_LINES_INPUT)

    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_enriched = df_solar_l19.merge(visual_features, left_index=True, right_on='solar_farm_id', how='left')
    df_enriched = df_enriched.merge(reconciliation_feature, on='solar_farm_id', how='left')
    
    # --- Finalization ---
    new_cols = [col for col in df_enriched.columns if col not in df_solar_l19.columns and col != 'solar_farm_id']
    df_enriched[new_cols] = df_enriched[new_cols].fillna(0)
    df_enriched.drop(columns=['solar_farm_id'], inplace=True, errors='ignore')
    
    # --- Persist Final Artifact ---
    output_path = OUTPUT_ARTIFACT
    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()