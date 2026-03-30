import logging
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from sklearn.neighbors import KDTree
import os
import sys

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = -1.0
BUFFER_RADII = [2000, 5000, 10000, 20000]
DEBUG_SAMPLE_SIZE = None # Set to None for full run

# --- Data Paths ---
# Centralized path configuration for different DNOs
DNO_ASSET_PATHS = {
    'ukpn': {
        'poles': r"[REDACTED_BY_SCRIPT]",
        'towers': r"[REDACTED_BY_SCRIPT]",
        'ohl_11kv': r"[REDACTED_BY_SCRIPT]",
        'ohl_33kv': r"[REDACTED_BY_SCRIPT]",
        'ohl_132kv': r"[REDACTED_BY_SCRIPT]",
        'substations': r"[REDACTED_BY_SCRIPT]"
    },
    'ng': {
        'master_artifact': r'[REDACTED_BY_SCRIPT]',
        'substations': r"[REDACTED_BY_SCRIPT]"
    }
}

# Map variations of DNO names to the keys in DNO_ASSET_PATHS
DNO_MAPPING = {
    'ukpn': 'ukpn',
    'nged': 'ng',
    'ng': 'ng',      # Explicit mapping for 'NG' source
    'national grid': 'ng'
}

# --- Helper Functions ---

def _load_data(path, layer=None, query_str=None):
    """[REDACTED_BY_SCRIPT]"""
    try:
        engine = 'pyogrio' if 'pyogrio' in sys.modules else None
        
        # Load data
        gdf = gpd.read_file(path, layer=layer, engine=engine)
        
        # 1. Apply Filter or Log Unfiltered Load
        if query_str:
            try:
                initial_count = len(gdf)
                gdf = gdf.query(query_str)
                filtered_count = len(gdf)
                if filtered_count == 0:
                    logging.warning(f"Query '{query_str}'[REDACTED_BY_SCRIPT]")
                    logging.info(f"[REDACTED_BY_SCRIPT]")
                else:
                    logging.info(f"[REDACTED_BY_SCRIPT]'{query_str}'")
            except Exception as qe:
                logging.error(f"Query '{query_str}'[REDACTED_BY_SCRIPT]")
                return gpd.GeoDataFrame(geometry=[], crs=PROJECT_CRS)
        else:
            logging.info(f"[REDACTED_BY_SCRIPT]")

        # 2. STRICT: Drop all columns except geometry immediately
        gdf = gdf[['geometry']]

        # 3. Reproject
        if gdf.crs != PROJECT_CRS:
            gdf = gdf.to_crs(PROJECT_CRS)

        # --- V7 Debug Mode ---
        if DEBUG_SAMPLE_SIZE is not None:
            logging.warning(f"[REDACTED_BY_SCRIPT]")
            gdf = gdf.head(DEBUG_SAMPLE_SIZE)
            
        return gdf
    except Exception as e:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return gpd.GeoDataFrame(geometry=[], crs=PROJECT_CRS)

def calculate_proximity_features(master_gdf, asset_gdf, prefix):
    """[REDACTED_BY_SCRIPT]"""
    if asset_gdf.empty:
        return pd.DataFrame({f'[REDACTED_BY_SCRIPT]': [NULL_SENTINEL] * len(master_gdf)}, index=master_gdf.index)

    asset_gdf = asset_gdf.to_crs(master_gdf.crs)
    
    # OPTIMIZATION BRANCH: Check geometry type
    geom_type = asset_gdf.geometry.iloc[0].geom_type
    
    if geom_type == 'Point':
        # --- KDTree Approach (Super Fast for Points) ---
        master_coords = np.vstack([master_gdf.geometry.x, master_gdf.geometry.y]).T
        asset_coords = np.vstack([asset_gdf.geometry.x, asset_gdf.geometry.y]).T
        
        # Build Tree (Sklearn)
        tree = KDTree(asset_coords)
        
        # Query k=1 for nearest neighbor
        # Sklearn returns (distances, indices)
        distances, _ = tree.query(master_coords, k=1)
        
        # Flatten distances array
        return pd.DataFrame({f'[REDACTED_BY_SCRIPT]': distances.flatten()}, index=master_gdf.index)
    
    else:
        # --- Spatial Index Approach (Lines/Polygons) ---
        try:
            sindex = asset_gdf.sindex
            
            # FIX: Explicitly pass geometry values (numpy array) to avoid GeoSeries/GeometryArray type conflicts
            # This resolves the "[REDACTED_BY_SCRIPT]" warning.
            nearest_indices = sindex.nearest(master_gdf.geometry.values, return_all=False)
            
            # Map indices to actual geometries
            left_idx = nearest_indices[0] # Indices into master_gdf
            right_idx = nearest_indices[1] # Indices into asset_gdf
            
            # Get geometries
            master_geoms = master_gdf.geometry.iloc[left_idx]
            asset_geoms = asset_gdf.geometry.iloc[right_idx]
            
            # Calculate distance using robust zipped iteration
            # This avoids the (GeoSeries, GeometryArray) type conflict while remaining O(N)
            distances = [m.distance(a) for m, a in zip(master_geoms, asset_geoms)]
            
            # Map results back to the master index
            result = pd.Series(distances, index=master_gdf.index[left_idx])
            result = result.reindex(master_gdf.index).fillna(NULL_SENTINEL)
            
            return pd.DataFrame({f'[REDACTED_BY_SCRIPT]': result}, index=master_gdf.index)

        except Exception as e:
            logging.warning(f"[REDACTED_BY_SCRIPT]")
            distances = []
            unary = asset_gdf.unary_union
            for point in master_gdf.geometry:
                distances.append(point.distance(unary))
            return pd.DataFrame({f'[REDACTED_BY_SCRIPT]': distances}, index=master_gdf.index)

def calculate_density_features(master_gdf, asset_gdf, prefix):
    """[REDACTED_BY_SCRIPT]"""
    density_features = pd.DataFrame(index=master_gdf.index)
    
    for r in BUFFER_RADII:
        density_features[f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL if asset_gdf.empty else 0

    if asset_gdf.empty:
        return density_features

    asset_gdf = asset_gdf.to_crs(master_gdf.crs)
    geom_type = asset_gdf.geometry.iloc[0].geom_type

    if geom_type == 'Point':
        # --- Sklearn KDTree Approach ---
        master_coords = np.vstack([master_gdf.geometry.x, master_gdf.geometry.y]).T
        asset_coords = np.vstack([asset_gdf.geometry.x, asset_gdf.geometry.y]).T
        
        # Build tree on ASSETS
        tree = KDTree(asset_coords)
        
        for r in BUFFER_RADII:
            # Valid Sklearn method: query_radius with count_only=True
            # This returns a simple array of integers, saving massive RAM
            counts = tree.query_radius(master_coords, r, count_only=True)
            
            col_name = f'[REDACTED_BY_SCRIPT]'
            density_features[col_name] = counts
            
    else:
        # --- Fallback for LineStrings ---
        logging.info(f"[REDACTED_BY_SCRIPT]")
        for r in BUFFER_RADII:
            buffers = master_gdf.geometry.buffer(r)
            buffered_gdf = gpd.GeoDataFrame(geometry=buffers, index=master_gdf.index)
            
            # Preserve index before join to allow accurate counting
            buffered_gdf['index_right_tracker'] = buffered_gdf.index
            
            join = gpd.sjoin(asset_gdf, buffered_gdf, how="inner", predicate="intersects")
            
            counts = join['index_right_tracker'].value_counts()
            col_name = f'[REDACTED_BY_SCRIPT]'
            density_features[col_name] = counts.reindex(master_gdf.index).fillna(0)
            
    return density_features

def calculate_intersection_features(master_gdf, ohl_gdf, prefix, substations_gdf):
    """
    Calculates the number of times a line from a master point to the nearest substation
    intersects with an overhead line.
    """
    # V6.8 FIX: Create a unique column name using the prefix to avoid downstream conflicts.
    col_name = f'[REDACTED_BY_SCRIPT]'

    if ohl_gdf.empty or substations_gdf.empty:
        # V6.6 FIX: Return only the new feature column as a DataFrame
        return pd.DataFrame({col_name: [NULL_SENTINEL] * len(master_gdf)}, index=master_gdf.index)

    # Ensure CRS match
    ohl_gdf = ohl_gdf.to_crs(master_gdf.crs)
    substations_gdf = substations_gdf.to_crs(master_gdf.crs)

    # Pre-build spatial indexes
    ohl_sindex = ohl_gdf.sindex
    substations_sindex = substations_gdf.sindex

    # --- Vectorized Approach ---
    # 1. Find nearest substation for all sites at once
    # The result of nearest is an array of indices into the input geometries (left) and the tree geometries (right)
    nearest_indices = substations_sindex.nearest(master_gdf.geometry, return_all=False)
    
    # Create a mapping from the site's original index to the index of the nearest substation
    site_indices = nearest_indices[:, 0]
    substation_indices = nearest_indices[:, 1]
    
    # Get the geometries using the correct indices
    site_geoms = master_gdf.geometry.iloc[site_indices]
    nearest_sub_geoms = substations_gdf.geometry.iloc[substation_indices]

    # 2. Create connection lines
    # The geometries are now correctly aligned by their position in the nearest_indices result.
    # We can create the lines directly.
    # V7.1 FIX: The LineString constructor expects a single list of points, not two separate arguments.
    connection_lines = gpd.GeoSeries(
        [LineString([s, n]) for s, n in zip(site_geoms, nearest_sub_geoms)],
        index=site_geoms.index, # Use the index from the site geometries to keep the link to the original master_gdf
        crs=master_gdf.crs
    )
    connection_lines_gdf = gpd.GeoDataFrame(geometry=connection_lines)
    connection_lines_gdf.index.name = 'hex_id' # Name the index for the join

    # 3. Find intersections between connection lines and OHLs
    # Use a spatial join. This is much faster than iterating.
    intersecting_ohl = gpd.sjoin(connection_lines_gdf, ohl_gdf, how='inner', predicate='intersects')

    # 4. Count intersections per site (hex_id)
    # The result of the sjoin will have one row for each intersection.
    intersection_counts = intersecting_ohl.index.value_counts()

    # 5. Align counts with the original master_gdf
    final_counts = intersection_counts.reindex(master_gdf.index).fillna(0)
        
    # V6.8 FIX: Return the uniquely named column.
    return pd.DataFrame({col_name: final_counts}, index=master_gdf.index)

def process_dno_group(master_gdf_group, dno_prefix):
    """[REDACTED_BY_SCRIPT]"""
    dno_prefix = DNO_MAPPING.get(dno_prefix)
    if not dno_prefix:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return master_gdf_group

    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # --- Load DNO-specific Assets ---
    if dno_prefix == 'ng':
        paths = DNO_ASSET_PATHS['ng']
        master_artifact = paths['master_artifact']
        # Pass queries into the loader so filtering happens BEFORE columns are stripped
        gdf_poles = _load_data(master_artifact, layer='points', query_str="asset_tag == 'pole'")
        gdf_towers = _load_data(master_artifact, layer='points', query_str="asset_tag == 'tower'")
        gdf_ohl_33kv = _load_data(master_artifact, layer='lines', query_str="voltage == '33kv' and asset_tag == 'ohl'")
        gdf_ohl_132kv = _load_data(master_artifact, layer='lines', query_str="voltage == '132kv' and asset_tag == 'ohl'")
        gdf_substations = _load_data(paths['substations'])
    elif dno_prefix == 'ukpn':
        paths = DNO_ASSET_PATHS['ukpn']
        gdf_poles = _load_data(paths['poles'])
        gdf_towers = _load_data(paths['towers'])
        gdf_ohl_33kv = _load_data(paths['ohl_33kv'])
        gdf_ohl_132kv = _load_data(paths['ohl_132kv'])
        gdf_substations = _load_data(paths['substations'])
    else:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return master_gdf_group

    # --- Feature Generation ---
    # V6.5 FIX: Initialize with an empty list to avoid joining the group to itself.
    all_features = []
    
    # Proximity
    all_features.append(calculate_proximity_features(master_gdf_group, gdf_poles, f'{dno_prefix}_pole'))
    all_features.append(calculate_proximity_features(master_gdf_group, gdf_towers, f'{dno_prefix}_tower'))
    all_features.append(calculate_proximity_features(master_gdf_group, gdf_ohl_33kv, f'[REDACTED_BY_SCRIPT]'))
    all_features.append(calculate_proximity_features(master_gdf_group, gdf_ohl_132kv, f'[REDACTED_BY_SCRIPT]'))

    # Density
    all_features.append(calculate_density_features(master_gdf_group, gdf_poles, f'{dno_prefix}_pole'))
    all_features.append(calculate_density_features(master_gdf_group, gdf_towers, f'{dno_prefix}_tower'))

    # Intersections
    # V6.8 FIX: The function now returns the correctly named column directly, so no rename is needed.
    all_features.append(calculate_intersection_features(master_gdf_group, gdf_ohl_33kv, f'[REDACTED_BY_SCRIPT]', gdf_substations))
    all_features.append(calculate_intersection_features(master_gdf_group, gdf_ohl_132kv, f'[REDACTED_BY_SCRIPT]', gdf_substations))

    # --- Merge all generated features for the group ---
    # V6.5 FIX: Consolidate all new feature dataframes at once to avoid fragmentation.
    if not all_features:
        return master_gdf_group # Return original group if no features were generated

    consolidated_features = pd.concat(all_features, axis=1)
    result_gdf = master_gdf_group.join(consolidated_features, how='left')

    # Calculate total intersections
    intersection_cols = [col for col in result_gdf.columns if 'intersection_count' in col and dno_prefix in col]
    if intersection_cols:
        result_gdf[f'[REDACTED_BY_SCRIPT]'] = result_gdf[intersection_cols].sum(axis=1)

    return result_gdf

# --- Main Executor ---

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Executor for integrating Overhead Line (OHL), Pole, and Tower features,
    dynamically loading data based on the DNO of each site.
    This function now ensures 'hex_id' is treated as a column internally.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # V17 ID Management: Ensure hex_id is a column for robust processing.
    id_col = 'hex_id'
    if master_gdf.index.name == id_col:
        gdf = master_gdf.reset_index()
    elif id_col in master_gdf.columns:
        gdf = master_gdf.copy() # Use a copy to avoid modifying the original in place
    else:
        logging.error(f"FATAL: '{id_col}'[REDACTED_BY_SCRIPT]")
        return master_gdf
    
    if 'dno_source' not in gdf.columns:
        logging.error("FATAL: 'dno_source'[REDACTED_BY_SCRIPT]")
        # Set index back before returning to maintain consistency with orchestrator expectations
        return gdf.set_index(id_col)

    # --- Process each DNO group separately ---
    processed_groups = []
    # V20 FIX: Get original columns from the dataframe that has 'hex_id' as a column.
    original_columns = gdf.columns.tolist()
    # Group by 'dno_source' on the DataFrame where hex_id is a column
    for dno_name, group in gdf.groupby('dno_source'):
        # The group still has hex_id as a column. Set it as index for the helper functions.
        group_indexed = group.set_index(id_col)
        processed_group = process_dno_group(group_indexed, dno_name.lower())
        # V20.1 FIX: Reset index to bring hex_id back as a column before appending
        processed_groups.append(processed_group.reset_index())

    if not processed_groups:
        logging.warning("[REDACTED_BY_SCRIPT]")
        return gdf.set_index(id_col)

    # --- Combine results and finalize ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # V19 FIX: Concatenate groups. This creates a single DataFrame with all original columns
    # plus all new feature columns. New columns for non-applicable DNOs will be NaN.
    combined_gdf = pd.concat(processed_groups, ignore_index=True, sort=False)

    # Identify all newly added columns across all groups
    new_cols = [col for col in combined_gdf.columns if col not in original_columns]

    # Fill NaNs only in the new columns.
    combined_gdf[new_cols] = combined_gdf[new_cols].fillna(NULL_SENTINEL)

    # V23 FEATURE: Create a unified 'total_intersections' column for the ML model.
    # This sums the DNO-specific totals (e.g., ng_total..., ukpn_total...) into one agnostic feature.
    # Since a site only belongs to one DNO, this safely collapses the sparse columns.
    total_cols = [c for c in combined_gdf.columns if 'total_connection_path_intersections' in c]
    if total_cols:
        # We use sum(min_count=1) to preserve NaNs if all inputs are NaN, 
        # but here we want 0 if no intersections found, so standard sum is fine 
        # (assuming NaNs were filled with Sentinel or 0 above).
        # We filter out Sentinels (-1) before summing to avoid negative totals.
        clean_totals = combined_gdf[total_cols].replace(NULL_SENTINEL, 0)
        combined_gdf['total_connection_path_intersections'] = clean_totals.sum(axis=1)
        new_cols.append('total_connection_path_intersections')
        logging.info(f"Created unified feature 'total_connection_path_intersections' from {total_cols}")

    # Ensure the result is a GeoDataFrame with the correct CRS
    final_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry', crs=master_gdf.crs)

    # LOGGING: Explicitly log the new columns found to confirm generation
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # V22 FIX: Bypass 'final_cols' filtering. Return ALL columns to prevent accidental dropping.
    # We only ensure hex_id is the index.
    final_gdf.set_index(id_col, inplace=True)
    
    # V21 FIX: Defragment the DataFrame before returning.
    final_gdf = final_gdf.copy()
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf