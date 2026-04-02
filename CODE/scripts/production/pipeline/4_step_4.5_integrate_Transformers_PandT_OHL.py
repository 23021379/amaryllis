import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, box
from tqdm import tqdm
from functools import partial
import multiprocessing

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
# MANDATE: The input MUST be the output of the PREVIOUS pipeline stage to prevent data amnesia.
SOLAR_DATA_INPUT = r'[REDACTED_BY_SCRIPT]'
L1_MASTER_ASSET_ARTIFACT = r'[REDACTED_BY_SCRIPT]'
SUBSTATION_L1_ARTIFACT = r"[REDACTED_BY_SCRIPT]"

# Output Artifact
# MANDATE: The output version MUST reflect the new intelligence layer.
L24_DATA_OUTPUT = r"[REDACTED_BY_SCRIPT]"

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
DENSITY_RADII_M = [2000, 5000, 10000, 20000]
MAX_RADIUS_M = max(DENSITY_RADII_M)

def process_solar_site(solar_site_tuple):
    """
    Implements Doctrine: The Solar-Centric Universe.
    Processes a SINGLE solar site by querying only local assets.
    """
    solar_farm_id, site_geom, nearest_sub_geom = solar_site_tuple
    features = {'solar_farm_id': solar_farm_id}

    # MANDATE: Define the Bounding Box, our surgical weapon.
    bbox = box(
        site_geom.x - MAX_RADIUS_M, site_geom.y - MAX_RADIUS_M,
        site_geom.x + MAX_RADIUS_M, site_geom.y + MAX_RADIUS_M
    )
    
    # MANDATE: Spatially-indexed read. This is the core of the performance gain.
    try:
        local_points = gpd.read_file(L1_MASTER_ASSET_ARTIFACT, layer='points', bbox=bbox)
        local_lines = gpd.read_file(L1_MASTER_ASSET_ARTIFACT, layer='lines', bbox=bbox)
    except Exception as e:
        # If a layer doesn't exist or there's an error, log it and return empty features.
        # This is a fault-tolerant measure.
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return features

    # --- Feature Engineering on LOCAL data ---

    # Nearest point of each type
    if not local_points.empty:
        local_points['distance_m'] = local_points.distance(site_geom)
        nearest_points = local_points.loc[local_points.groupby(['voltage', 'asset_tag'])['distance_m'].idxmin()]
        for _, row in nearest_points.iterrows():
            features[f"[REDACTED_BY_SCRIPT]"] = row.distance_m

    # Point densities
    for radius in DENSITY_RADII_M:
        site_buffer = site_geom.buffer(radius)
        points_in_buffer = local_points[local_points.intersects(site_buffer)]
        if not points_in_buffer.empty:
            point_counts = points_in_buffer.groupby(['voltage', 'asset_tag']).size()
            for (voltage, tag), count in point_counts.items():
                radius_km = radius / 1000
                area_km2 = np.pi * (radius_km ** 2)
                features[f"[REDACTED_BY_SCRIPT]"] = count
                features[f"[REDACTED_BY_SCRIPT]"] = count / area_km2

    # Line processing
    if not local_lines.empty:
        # Pre-calculate corridors for local lines
        local_lines['corridor_geom'] = local_lines.apply(lambda r: r.geometry.buffer(r.buffer_m), axis=1)
        
        # Nearest corridor of each type
        local_lines['distance_to_corridor_m'] = local_lines['corridor_geom'].distance(site_geom)
        nearest_lines = local_lines.loc[local_lines.groupby(['voltage', 'asset_tag'])['distance_to_corridor_m'].idxmin()]
        for _, row in nearest_lines.iterrows():
            features[f"[REDACTED_BY_SCRIPT]"] = row.distance_to_corridor_m

        # Intersections with site polygon
        intersecting_lines = local_lines[local_lines['corridor_geom'].intersects(site_geom)]
        if not intersecting_lines.empty:
            for _, row in intersecting_lines.iterrows():
                 features[f"[REDACTED_BY_SCRIPT]"] = 1

        # Connection Friction
        if nearest_sub_geom and not nearest_sub_geom.is_empty:
            connection_path = LineString([site_geom, nearest_sub_geom])
            friction_lines = local_lines[local_lines['corridor_geom'].intersects(connection_path)]
            if not friction_lines.empty:
                friction_counts = friction_lines.groupby(['voltage', 'asset_tag']).size()
                for (voltage, tag), count in friction_counts.items():
                    features[f"[REDACTED_BY_SCRIPT]"] = count

    return features

def main():
    logging.info("[REDACTED_BY_SCRIPT]")

    df_solar = pd.read_csv(SOLAR_DATA_INPUT)
    gdf_solar = gpd.GeoDataFrame(
        df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS
    ).reset_index().rename(columns={'index': 'solar_farm_id'})

    substation_gdf = gpd.read_file(SUBSTATION_L1_ARTIFACT).to_crs(TARGET_CRS)

    # MANDATE: Pre-calculate nearest substation for all sites in a single vectorized operation.
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Step 1: Use sjoin_nearest to find the index of the nearest substation. This is its primary, reliable function.
    # The 'index_right' column will contain the index from the original substation_gdf.
    gdf_solar_with_index = gpd.sjoin_nearest(gdf_solar, substation_gdf[['geometry']], how="left")
    
    # Step 2: Explicitly create the '[REDACTED_BY_SCRIPT]' column using a standard merge.
    # This uses the returned 'index_right' to look up the geometry from the original substation GDF.
    # This two-step process is more robust than relying on geometry column renaming within a single join.
    substation_geoms = substation_gdf[['geometry']].rename(columns={'geometry': '[REDACTED_BY_SCRIPT]'})
    gdf_solar = gdf_solar_with_index.merge(substation_geoms, left_on='index_right', right_index=True, how='left')

    # Clean up the intermediate index column from the join.
    gdf_solar = gdf_solar.drop(columns=['index_right'], errors='ignore')

    # MANDATE: Create a lightweight list of tuples for multiprocessing. Avoids pickling large GDFs.
    # Now includes the pre-calculated nearest substation geometry.
    solar_site_tuples = list(zip(gdf_solar.solar_farm_id, gdf_solar.geometry, gdf_solar.nearest_substation_geom))

    num_workers = multiprocessing.cpu_count() - 1
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # MANDATE: The worker function no longer needs the full substation GDF.
    worker_func = process_solar_site
    
    all_results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm for a progress bar, which is essential for long-running tasks.
        with tqdm(total=len(solar_site_tuples), desc="[REDACTED_BY_SCRIPT]") as pbar:
            for result in pool.imap_unordered(worker_func, solar_site_tuples):
                all_results.append(result)
                pbar.update()

    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Load the TRUE input artifact which contains ALL prior features.
    df_previous_artifact = pd.read_csv(SOLAR_DATA_INPUT)
    
    df_previous_artifact = df_previous_artifact.reset_index().rename(columns={'index': 'solar_farm_id'})
    
    df_features = pd.DataFrame(all_results).fillna(-1.0)

    # MANDATE: Merge the new features into the complete dataframe from the previous step.
    df_final = pd.merge(df_previous_artifact, df_features, on='solar_farm_id', how='left')

    # --- Synthesize capstone 'Alpha' features ---
    logging.info("[REDACTED_BY_SCRIPT]'Alpha' features...")
    for radius_m in DENSITY_RADII_M:
        radius_km_str = f"[REDACTED_BY_SCRIPT]"
        count_cols = [col for col in df_final.columns if f'[REDACTED_BY_SCRIPT]' in col]
        if count_cols:
            df_final[f'[REDACTED_BY_SCRIPT]'] = df_final[count_cols].sum(axis=1)

    intersection_cols = [col for col in df_final.columns if '_connection_path_intersection_count' in col]
    if intersection_cols:
        df_final['[REDACTED_BY_SCRIPT]'] = df_final[intersection_cols].sum(axis=1)

    logging.info("[REDACTED_BY_SCRIPT]")
    df_final = df_final.drop(columns=['geometry', '[REDACTED_BY_SCRIPT]'], errors='ignore')
    df_final.to_csv(L24_DATA_OUTPUT, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()