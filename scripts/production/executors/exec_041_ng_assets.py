import os
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, box
from tqdm import tqdm
import concurrent.futures
from functools import partial

# --- Project Setup ---
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
SUBSTATION_L1_ARTIFACT = r"[REDACTED_BY_SCRIPT]"
L1_MASTER_ASSET_ARTIFACT = r'[REDACTED_BY_SCRIPT]'

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
DENSITY_RADII_M = [2000, 5000, 10000, 20000]
MAX_RADIUS_M = max(DENSITY_RADII_M)
# V5.3 Performance Tuning: Set max workers for multiprocessing. None uses os.cpu_count().
MAX_WORKERS = 8 

def process_single_site(site_tuple, master_asset_path, substation_gdf):
    """
    Processes a single site to generate NG asset features.
    This is the core "solar-centric" logic.
    """
    # V5.3 hex_id FIX: The tuple now contains hex_id, not amaryllis_id
    hex_id, site_geom = site_tuple
    features = {'hex_id': hex_id}

    # Define the surgical bounding box for this site
    bbox = box(
        site_geom.x - MAX_RADIUS_M, site_geom.y - MAX_RADIUS_M,
        site_geom.x + MAX_RADIUS_M, site_geom.y + MAX_RADIUS_M
    )
    
    try:
        # Spatially-indexed read for local assets, then reproject.
        # This two-step process avoids the pyogrio warning with GPKG drivers.
        local_points = gpd.read_file(master_asset_path, layer='points', bbox=bbox)
        if not local_points.empty:
            local_points = local_points.to_crs(TARGET_CRS)
        
        local_lines = gpd.read_file(master_asset_path, layer='lines', bbox=bbox)
        if not local_lines.empty:
            local_lines = local_lines.to_crs(TARGET_CRS)

    except Exception as e:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return features

    # --- Feature Engineering on LOCAL data ---

    # Nearest point of each type (poles, towers, transformers)
    if not local_points.empty:
        local_points['distance_m'] = local_points.distance(site_geom)
        nearest_points = local_points.loc[local_points.groupby(['voltage', 'asset_tag'])['distance_m'].idxmin()]
        for _, row in nearest_points.iterrows():
            features[f"[REDACTED_BY_SCRIPT]"] = row.distance_m

    # Point densities within various radii
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

    # Line processing (cables, ohl)
    if not local_lines.empty:
        # V5.3 Performance: Ensure 'buffer_m' column exists before using it.
        if 'buffer_m' in local_lines.columns:
            local_lines['corridor_geom'] = local_lines.apply(lambda r: r.geometry.buffer(r.buffer_m), axis=1)
        else:
            # If buffer_m is not defined, default to a reasonable value (e.g., 10m)
            local_lines['corridor_geom'] = local_lines.geometry.buffer(10)

        # Nearest corridor of each type
        local_lines['distance_to_corridor_m'] = local_lines['corridor_geom'].distance(site_geom)
        nearest_lines = local_lines.loc[local_lines.groupby(['voltage', 'asset_tag'])['distance_to_corridor_m'].idxmin()]
        for _, row in nearest_lines.iterrows():
            features[f"[REDACTED_BY_SCRIPT]"] = row.distance_to_corridor_m

        # Intersections with site polygon
        # V5.3 FIX: The site geometry is a point, so we need to buffer it to check for intersections.
        # Using a small buffer (e.g., 1 meter) to represent the site area.
        site_poly = site_geom.buffer(1)
        intersecting_lines = local_lines[local_lines['corridor_geom'].intersects(site_poly)]
        if not intersecting_lines.empty:
            for _, row in intersecting_lines.iterrows():
                 features[f"[REDACTED_BY_SCRIPT]"] = 1

        # Connection Friction (path to nearest substation)
        if not substation_gdf.empty:
            nearest_sub_series = substation_gdf.distance(site_geom).idxmin()
            nearest_sub_geom = substation_gdf.loc[nearest_sub_series].geometry
            
            if nearest_sub_geom and not nearest_sub_geom.is_empty:
                connection_path = LineString([site_geom, nearest_sub_geom])
                friction_lines = local_lines[local_lines['corridor_geom'].intersects(connection_path)]
                if not friction_lines.empty:
                    friction_counts = friction_lines.groupby(['voltage', 'asset_tag']).size()
                    for (voltage, tag), count in friction_counts.items():
                        features[f"[REDACTED_BY_SCRIPT]"] = count

    return features

def execute(master_gdf):
    """
    Executor entry point for calculating National Grid asset features using a solar-centric approach.
    V5.3 Update: Now uses multiprocessing for significant performance improvement.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    if 'geometry' not in master_gdf.columns or 'hex_id' not in master_gdf.columns:
        logging.error("[REDACTED_BY_SCRIPT]'geometry' and 'hex_id' columns.")
        return master_gdf

    # --- Phase 1: Load Global Read-Only Data ---
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        substation_gdf = gpd.read_file(SUBSTATION_L1_ARTIFACT).to_crs(TARGET_CRS)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf # Return unchanged GDF

    # --- Phase 2: Parallel Site Processing ---
    num_sites = len(master_gdf)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    site_tuples = list(zip(master_gdf.hex_id, master_gdf.geometry))
    
    # Create a partial function with fixed arguments for the worker processes
    processing_func = partial(process_single_site, 
                              master_asset_path=L1_MASTER_ASSET_ARTIFACT, 
                              substation_gdf=substation_gdf)

    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use tqdm to visualize progress of the parallel execution
        # The map function distributes the site_tuples across the worker pool
        results_iterator = executor.map(processing_func, site_tuples)
        all_results = list(tqdm(results_iterator, total=num_sites, desc="[REDACTED_BY_SCRIPT]"))

    # --- Phase 3: Final Merge and Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    if not all_results:
        logging.warning("[REDACTED_BY_SCRIPT]")
        return master_gdf

    df_features = pd.DataFrame(all_results)
    
    # V5.3 Merge Fix: Ensure the join key 'hex_id' is of a compatible type.
    master_gdf['hex_id'] = master_gdf['hex_id'].astype(df_features['hex_id'].dtype)

    # Merge the new features back into the master dataframe using the authoritative key
    master_gdf = master_gdf.merge(df_features, on='hex_id', how='left')
    
    logging.info("[REDACTED_BY_SCRIPT]'Alpha' features for NG assets...")
    for radius_m in DENSITY_RADII_M:
        radius_km_str = f"[REDACTED_BY_SCRIPT]"
        # Sum of all structure counts (poles, towers) in radius
        count_cols = [col for col in master_gdf.columns if f'[REDACTED_BY_SCRIPT]' in col and ('pole' in col or 'tower' in col) and 'ng_' in col]
        if count_cols:
            master_gdf[f'[REDACTED_BY_SCRIPT]'] = master_gdf[count_cols].sum(axis=1)

    # Sum of all intersections on the connection path to the substation
    intersection_cols = [col for col in master_gdf.columns if '_connection_path_intersection_count' in col and 'ng_' in col]
    if intersection_cols:
        master_gdf['[REDACTED_BY_SCRIPT]'] = master_gdf[intersection_cols].sum(axis=1)

    # Fill NaNs created by the merge with a neutral value like 0.
    # V5.3 FIX: Use 'hex_id' instead of 'amaryllis_id'
    feature_columns = [col for col in df_features.columns if col != 'hex_id']
    master_gdf[feature_columns] = master_gdf[feature_columns].fillna(0)

    # V5.4 GeoDataFrame Restoration: Ensure the output is a GeoDataFrame
    master_gdf = gpd.GeoDataFrame(master_gdf, geometry='geometry', crs=TARGET_CRS)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf

