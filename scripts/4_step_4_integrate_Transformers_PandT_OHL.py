import os
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import multiprocessing
from functools import partial
from itertools import chain


# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
SOLAR_DATA_INPUT = r'[REDACTED_BY_SCRIPT]'
SUBSTATION_L1_ARTIFACT = '[REDACTED_BY_SCRIPT]'
L0_ARTIFACT_DIR = r"[REDACTED_BY_SCRIPT]"
L1_MASTER_ARTIFACT = r'[REDACTED_BY_SCRIPT]'

# Output Artifact
L24_DATA_OUTPUT = r'[REDACTED_BY_SCRIPT]'

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
DENSITY_RADII_M = [2000, 5000, 10000, 20000] # Define multiple radii for density analysis

# MANDATE: The configuration-driven campaign definition. This is the "brain" of the orchestrator.
ASSET_PROCESSING_CAMPAIGN = [
    # --- 11kV (Distribution Low Voltage) ---
    {"file": "[REDACTED_BY_SCRIPT]", "type": "line", "voltage": "11kv", "asset_tag": "dlv_cable", "buffer": 5},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "line", "voltage": "11kv", "asset_tag": "dlv_ohl", "buffer": 5},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "11kv", "asset_tag": "pole"},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "11kv", "asset_tag": "tower"},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "11kv", "asset_tag": "transformer"},
    
    # --- 33kV ---
    {"file": "[REDACTED_BY_SCRIPT]", "type": "line", "voltage": "33kv", "asset_tag": "ohl_cable", "buffer": 20},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "line", "voltage": "33kv", "asset_tag": "ohl", "buffer": 20},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "33kv", "asset_tag": "pole"},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "33kv", "asset_tag": "tower"},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "33kv", "asset_tag": "transformer"},

    # --- 66kV ---
    {"file": "[REDACTED_BY_SCRIPT]", "type": "line", "voltage": "66kv", "asset_tag": "ohl_cable", "buffer": 25},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "line", "voltage": "66kv", "asset_tag": "ohl", "buffer": 25},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "66kv", "asset_tag": "pole"},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "66kv", "asset_tag": "tower"},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "66kv", "asset_tag": "transformer"},

    # --- 132kV ---
    {"file": "[REDACTED_BY_SCRIPT]", "type": "line", "voltage": "132kv", "asset_tag": "ohl_cable", "buffer": 30},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "line", "voltage": "132kv", "asset_tag": "ohl", "buffer": 30},
    {"file": "[REDACTED_BY_SCRIPT]", "type": "point", "voltage": "132kv", "asset_tag": "pole"},
    # Note: No 132kV tower or transformer files in the provided list, but the engine is ready for them.
]

# Architectural Parameters
LOCAL_NETWORK_BUFFER_METERS = 2000
TARGET_CRS = "EPSG:27700"

def forge_master_artifact():
    """
    Implements Doctrine: Unify the Battlefield (Correctly).
    Creates a single, spatially-indexed master GeoPackage from all L0 assets.
    This is a one-time operation.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Ensure the old artifact is removed to prevent append errors on re-runs.
    if os.path.exists(L1_MASTER_ARTIFACT):
        os.remove(L1_MASTER_ARTIFACT)
        logging.info(f"[REDACTED_BY_SCRIPT]")

    for i, config in enumerate(ASSET_PROCESSING_CAMPAIGN):
        asset_path = os.path.join(L0_ARTIFACT_DIR, config['file'])
        layer_name = f"{config['type']}s" # 'points' or 'lines'
        
        try:
            logging.info(f"[REDACTED_BY_SCRIPT]'{layer_name}'")
            gdf_asset = gpd.read_file(asset_path)

            if gdf_asset.empty:
                logging.warning(f"[REDACTED_BY_SCRIPT]")
                continue

            # MANDATE: Unconditional CRS unification
            if gdf_asset.crs != TARGET_CRS:
                gdf_asset = gdf_asset.to_crs(TARGET_CRS)
            
            # MANDATE: Add strategic metadata
            gdf_asset['voltage'] = config['voltage']
            gdf_asset['asset_tag'] = config['asset_tag']
            if config['type'] == 'line':
                gdf_asset['buffer_m'] = config['buffer']
            
            # Select only necessary columns
            cols_to_keep = ['geometry', 'voltage', 'asset_tag']
            if config['type'] == 'line':
                cols_to_keep.append('buffer_m')
            
            gdf_asset = gdf_asset[cols_to_keep]

            # MANDATE: Append to the GeoPackage. This creates the file on the first write.
            # This is memory-efficient as it writes chunk-by-chunk.
            gdf_asset.to_file(L1_MASTER_ARTIFACT, layer=layer_name, driver='GPKG', mode='a')

        except Exception as e:
            logging.error(f"[REDACTED_BY_SCRIPT]")
            # This is a critical failure. Stop the process.
            return

    logging.info(f"[REDACTED_BY_SCRIPT]")

def load_and_unify_assets(campaign, asset_dir, target_crs):
    """
    Implements Doctrine: Unify the Battlefield.
    Loads all assets, segregates by geometry type, and unifies them into master GeoDataFrames.
    """
    points_list = []
    lines_list = []
    logging.info("[REDACTED_BY_SCRIPT]")

    for config in campaign:
        asset_path = os.path.join(asset_dir, config['file'])
        try:
            gdf_asset = gpd.read_file(asset_path)
            if gdf_asset.crs != target_crs:
                gdf_asset = gdf_asset.to_crs(target_crs)
            
            gdf_asset['voltage'] = config['voltage']
            gdf_asset['asset_tag'] = config['asset_tag']
            
            if config['type'] == 'point':
                points_list.append(gdf_asset[['geometry', 'voltage', 'asset_tag']])
            elif config['type'] == 'line':
                gdf_asset['buffer_m'] = config['buffer']
                lines_list.append(gdf_asset)

        except Exception as e:
            logging.warning(f"[REDACTED_BY_SCRIPT]")
            continue
    
    gdf_points_unified = gpd.GeoDataFrame(pd.concat(points_list, ignore_index=True), crs=target_crs)
    gdf_lines_unified = gpd.GeoDataFrame(pd.concat(lines_list, ignore_index=True), crs=target_crs)
    
    # Pre-buffer line assets to create wayleave corridors
    gdf_lines_unified['corridor_geom'] = gdf_lines_unified.apply(lambda row: row.geometry.buffer(row.buffer_m), axis=1)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_points_unified, gdf_lines_unified

def process_solar_batch(gdf_solar_batch, gdf_points_unified, gdf_lines_unified, gdf_substations):
    """
    Implements Doctrine: Solar-Centric Processing.
    Processes a single batch of solar sites against the unified asset GDFs.
    """
    results = []

    # --- Pre-computation for the batch ---
    # Find nearest substation for each site in the batch (once)
    batch_with_subs = gpd.sjoin_nearest(gdf_solar_batch, gdf_substations, how='left', distance_col='dist_to_sub_m')
    batch_with_subs = batch_with_subs.drop_duplicates(subset='solar_farm_id', keep='first')

    # --- Point Asset Processing ---
    # 1. Find nearest of each type
    nearest_points = gpd.sjoin_nearest(
        gdf_solar_batch[['solar_farm_id', 'geometry']],
        gdf_points_unified,
        how='left',
        distance_col='distance_m'
    )
    # Pivot to get one row per solar farm with distances to nearest of each type
    df_nearest_points = nearest_points.pivot_table(
        index='solar_farm_id',
        columns=['voltage', 'asset_tag'],
        values='distance_m',
        aggfunc='min'
    )
    df_nearest_points.columns = [f"[REDACTED_BY_SCRIPT]" for v, t in df_nearest_points.columns]

    # 2. Calculate densities for all radii
    point_features = {row.solar_farm_id: {} for _, row in gdf_solar_batch.iterrows()}
    for radius_m in DENSITY_RADII_M:
        buffer_geom = gdf_solar_batch.geometry.buffer(radius_m)
        gdf_batch_buffers = gpd.GeoDataFrame(gdf_solar_batch[['solar_farm_id']], geometry=buffer_geom, crs=TARGET_CRS)
        
        points_in_buffer = gpd.sjoin(gdf_points_unified, gdf_batch_buffers, how="inner", predicate="intersects")
        
        if not points_in_buffer.empty:
            point_counts = points_in_buffer.groupby(['solar_farm_id', 'voltage', 'asset_tag']).size()
            for (solar_id, voltage, tag), count in point_counts.items():
                radius_km_str = f"[REDACTED_BY_SCRIPT]"
                count_col = f"[REDACTED_BY_SCRIPT]"
                density_col = f"[REDACTED_BY_SCRIPT]"
                buffer_area_km2 = np.pi * ((radius_m / 1000) ** 2)
                
                point_features[solar_id][count_col] = count
                point_features[solar_id][density_col] = count / buffer_area_km2

    # --- Line Asset Processing ---
    line_features = {row.solar_farm_id: {} for _, row in gdf_solar_batch.iterrows()}
    gdf_line_corridors = gdf_lines_unified.set_geometry('corridor_geom')

    # 1. Find nearest corridor of each type
    nearest_lines = gpd.sjoin_nearest(
        gdf_solar_batch[['solar_farm_id', 'geometry']],
        gdf_line_corridors,
        how='left',
        distance_col='distance_m'
    )
    df_nearest_lines = nearest_lines.pivot_table(
        index='solar_farm_id',
        columns=['voltage', 'asset_tag'],
        values='distance_m',
        aggfunc='min'
    )
    df_nearest_lines.columns = [f"[REDACTED_BY_SCRIPT]" for t, v in df_nearest_lines.columns]

    # 2. Calculate densities, intersections, and friction
    for _, site in batch_with_subs.iterrows():
        solar_id = site.solar_farm_id
        site_geom = site.geometry
        
        # Create connection path
        sub_geom = gdf_substations.loc[site['index_right']].geometry if pd.notna(site['index_right']) else None
        connection_path = LineString([site_geom, sub_geom]) if sub_geom else None
        
        # Use spatial index for efficient querying of lines
        possible_matches_idx = list(gdf_line_corridors.sindex.intersection(site_geom.buffer(max(DENSITY_RADII_M)).bounds))
        nearby_lines = gdf_line_corridors.iloc[possible_matches_idx]

        for _, line in nearby_lines.iterrows():
            voltage = line['voltage']
            tag = line['asset_tag']
            
            # Intersection
            intersects_col = f"[REDACTED_BY_SCRIPT]"
            if line['corridor_geom'].intersects(site_geom):
                line_features[solar_id][intersects_col] = 1

            # Connection Friction
            if connection_path and line['corridor_geom'].intersects(connection_path):
                friction_col = f"[REDACTED_BY_SCRIPT]"
                line_features[solar_id][friction_col] = line_features[solar_id].get(friction_col, 0) + 1

        # Density
        for radius_m in DENSITY_RADII_M:
            site_buffer = site_geom.buffer(radius_m)
            clipped_lines = gpd.overlay(gdf_lines_unified.set_geometry('geometry'), gpd.GeoDataFrame(geometry=[site_buffer], crs=TARGET_CRS), how='intersection')
            if not clipped_lines.empty:
                clipped_lines['length_km'] = clipped_lines.geometry.length / 1000
                density_summary = clipped_lines.groupby(['voltage', 'asset_tag'])['length_km'].sum()
                for (voltage, tag), length in density_summary.items():
                    radius_km_str = f"[REDACTED_BY_SCRIPT]"
                    density_col = f"[REDACTED_BY_SCRIPT]"
                    buffer_area_km2 = np.pi * ((radius_m / 1000) ** 2)
                    line_features[solar_id][density_col] = length / buffer_area_km2

    # --- Assemble final results for the batch ---
    for _, site in gdf_solar_batch.iterrows():
        solar_id = site.solar_farm_id
        final_features = {'solar_farm_id': solar_id}
        final_features.update(point_features.get(solar_id, {}))
        final_features.update(line_features.get(solar_id, {}))
        # Add pivoted nearest point distances
        if solar_id in df_nearest_points.index:
            final_features.update(df_nearest_points.loc[solar_id].to_dict())
        # Add pivoted nearest line distances
        if solar_id in df_nearest_lines.index:
            final_features.update(df_nearest_lines.loc[solar_id].to_dict())
        results.append(final_features)

    return results



def main():
    """
    Main orchestration function for the PARALLELIZED topological feature engine.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Phase 1: Load Global Read-Only Data ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_solar_full = pd.read_csv(SOLAR_DATA_INPUT)
    gdf_solar_full = gpd.GeoDataFrame(
        df_solar_full, geometry=gpd.points_from_xy(df_solar_full.easting, df_solar_full.northing), crs=TARGET_CRS
    )
    gdf_solar_full['solar_farm_id'] = gdf_solar_full.index

    gdf_substations = gpd.read_file(SUBSTATION_L1_ARTIFACT).to_crs(TARGET_CRS)
    
    gdf_points_unified, gdf_lines_unified = load_and_unify_assets(ASSET_PROCESSING_CAMPAIGN, L0_ARTIFACT_DIR, TARGET_CRS)
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Phase 2: Parallel Batch Processing ---
    # MANDATE: Use one fewer core than total to leave resources for OS.
    num_workers = 1
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Chunk the solar sites dataframe into a list of smaller dataframes for the workers.
    solar_chunks = np.array_split(gdf_solar_full, num_workers)
    
    # Use functools.partial to create a worker function with the large, read-only
    # dataframes already "baked in". This is essential for pool.map.
    worker_func = partial(process_solar_batch, 
                          gdf_points_unified=gdf_points_unified, 
                          gdf_lines_unified=gdf_lines_unified, 
                          gdf_substations=gdf_substations)

    all_results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        logging.info("[REDACTED_BY_SCRIPT]")
        # pool.map applies the worker_func to each chunk in the solar_chunks list.
        # It blocks until all processes are complete.
        list_of_results_lists = pool.map(worker_func, solar_chunks)
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Flatten the list of lists into a single list of results.
        all_results = list(chain.from_iterable(list_of_results_lists))

    # --- Phase 3: Final Merge and Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_features = pd.DataFrame(all_results).fillna(-1.0)
    gdf_solar_full = gdf_solar_full.merge(df_features, on='solar_farm_id', how='left')
    
    logging.info("[REDACTED_BY_SCRIPT]'Alpha' features...")
    for radius_m in DENSITY_RADII_M:
        radius_km_str = f"[REDACTED_BY_SCRIPT]"
        count_cols = [col for col in gdf_solar_full.columns if f'[REDACTED_BY_SCRIPT]' in col]
        if count_cols:
            gdf_solar_full[f'[REDACTED_BY_SCRIPT]'] = gdf_solar_full[count_cols].sum(axis=1)

    intersection_cols = [col for col in gdf_solar_full.columns if '_connection_path_intersection_count' in col]
    if intersection_cols:
        gdf_solar_full['[REDACTED_BY_SCRIPT]'] = gdf_solar_full[intersection_cols].sum(axis=1)

    # --- Phase 4: Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_final = gdf_solar_full.drop(columns=['geometry'], errors='ignore')
    df_final.to_csv(L24_DATA_OUTPUT, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")


if __name__ == "__main__":
    #main()
    forge_master_artifact()