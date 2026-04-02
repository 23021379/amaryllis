import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Paths to land use/cover datasets
# ASSUMPTION: Files are in the same directory as the previously referenced L1_osm_landuse.gpkg
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
OSM_LAND_USE_PATH = os.path.join(BASE_DATA_DIR, "L1_osm_landuse.gpkg")
OSM_BUILDINGS_PATH = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
CLC_LAND_COVER_PATH = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")

PROJECT_CRS = "EPSG:27700"
RADII_METERS = [1000, 2000, 5000, 10000] # 1km, 2km, 5km, 10km

# Mapping for output features.
# Keys are the feature category names (e.g. 'agricultural' -> '[REDACTED_BY_SCRIPT]')
# Values are lists of source categories/codes to look for in the respective datasets.
LAND_USE_CATEGORY_MAP = {
    'agricultural': {
        'source': 'clc',
        'codes': ['211', '212', '213', '221', '222', '223', '231', '241', '242', '243', '244'] # Example CLC agricultural codes
    },
    'urban_fabric': {
        'source': 'combined', # Special handling: buildings + osm landuse
        'osm_fclasses': ['residential', 'commercial', 'retail', 'industrial'],
        'clc_codes': ['111', '112'] # Continuous/Discontinuous urban fabric
    },
    '[REDACTED_BY_SCRIPT]': {
        'source': 'osm_landuse',
        'fclasses': ['commercial', 'retail', 'industrial']
    },
    'natural_protected': {
        'source': 'clc', # Fallback/Placeholder as specific protected areas usually come from separate datasets (SSSI, etc.)
        'codes': ['311', '312', '313', '321', '322', '323', '324', '411', '412', '421', '422', '423'] # Forests, scrub, wetlands
    }
}

def calculate_land_use_percentages(grid_gdf, land_use_gdf, category_col, target_categories, feature_prefix):
    """
    Calculates the percentage of area for specific categories within various radii.
    
    Args:
        grid_gdf: The master grid GeoDataFrame.
        land_use_gdf: The source land use GeoDataFrame.
        category_col: The column name in land_use_gdf to check for categories.
        target_categories: A list of values in category_col to include.
        feature_prefix: The prefix for the output column (e.g., 'area_pct_agricultural').
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # Filter land use data early to reduce processing
    relevant_land_use = land_use_gdf[land_use_gdf[category_col].isin(target_categories)].copy()
    
    if relevant_land_use.empty:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        for radius in RADII_METERS:
            feature_name = f"[REDACTED_BY_SCRIPT]"
            grid_gdf[feature_name] = 0.0
        return grid_gdf

    for radius in RADII_METERS:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        
        # Create buffers
        # Note: Doing this inside the loop to avoid keeping massive buffered objects in memory if not needed
        buffered_grid = grid_gdf[['geometry']].copy() # Only keep geometry for the join
        buffered_grid['geometry'] = buffered_grid.geometry.buffer(radius)
        buffered_grid['hex_id'] = grid_gdf.index # Preserve index for mapping back
        buffer_area = np.pi * radius**2

        # Spatial Join / Overlay
        # 'intersection' keeps only the overlapping areas
        intersections = gpd.overlay(buffered_grid, relevant_land_use, how='intersection')
        
        if intersections.empty:
             feature_name = f"[REDACTED_BY_SCRIPT]"
             grid_gdf[feature_name] = 0.0
             continue

        # Calculate area
        intersections['intersected_area'] = intersections.geometry.area
        
        # Group by hex_id and sum area
        total_area_per_hex = intersections.groupby('hex_id')['intersected_area'].sum()
        
        # Calculate percentage
        percentage = (total_area_per_hex / buffer_area) * 100
        
        feature_name = f"[REDACTED_BY_SCRIPT]"
        grid_gdf[feature_name] = percentage
        grid_gdf[feature_name] = grid_gdf[feature_name].fillna(0) # Fill missing hexes with 0
        
    return grid_gdf

def execute(master_gdf):
    """
    Main execution function for the land use module.
    Implements 'Smart Skip' to avoid processing if features exist.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Ensure hex_id is the index
    if master_gdf.index.name != 'hex_id':
        if 'hex_id' in master_gdf.columns:
            master_gdf.set_index('hex_id', inplace=True)
        else:
            logging.error("Critical error: 'hex_id'[REDACTED_BY_SCRIPT]")
            raise KeyError("'hex_id'[REDACTED_BY_SCRIPT]")

    # --- Smart Skip Check ---
    # Check if a representative feature from each category exists
    # Legacy Naming: pct_area_agricultural_1km, pct_area_industrial_commercial_1km, etc.
    required_features = [
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]"
    ]
    
    missing_features = [f for f in required_features if f not in master_gdf.columns]
    
    if not missing_features:
        logging.info("[REDACTED_BY_SCRIPT]")
        logging.info("[REDACTED_BY_SCRIPT]")
        return master_gdf
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Reset index for processing if needed, but our helper functions handle index mapping
    # We will work directly on master_gdf or copies of it.
    
    # Ensure CRS
    if master_gdf.crs.to_string() != PROJECT_CRS:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        master_gdf = master_gdf.to_crs(PROJECT_CRS)

    # --- Load Data Sources (Lazy Loading) ---
    # We only load what we need.
    
    # 1. OSM Land Use (Commercial/Industrial)
    if any('[REDACTED_BY_SCRIPT]' in f for f in missing_features):
        if os.path.exists(OSM_LAND_USE_PATH):
            logging.info(f"[REDACTED_BY_SCRIPT]")
            osm_landuse = gpd.read_file(OSM_LAND_USE_PATH)
            if osm_landuse.crs.to_string() != PROJECT_CRS:
                osm_landuse = osm_landuse.to_crs(PROJECT_CRS)
            
            master_gdf = calculate_land_use_percentages(
                master_gdf, osm_landuse, 'fclass', 
                LAND_USE_CATEGORY_MAP['[REDACTED_BY_SCRIPT]']['fclasses'], 
                '[REDACTED_BY_SCRIPT]'
            )
        else:
            logging.error(f"[REDACTED_BY_SCRIPT]")

    # 2. CLC Land Cover (Agricultural, Natural/Protected)
    # Checking if we need to load CLC
    need_clc = any('agricultural' in f for f in missing_features) or any('natural_protected' in f for f in missing_features)
    
    if need_clc:
        if os.path.exists(CLC_LAND_COVER_PATH):
            logging.info(f"[REDACTED_BY_SCRIPT]")
            clc_gdf = gpd.read_file(CLC_LAND_COVER_PATH)
            if clc_gdf.crs.to_string() != PROJECT_CRS:
                clc_gdf = clc_gdf.to_crs(PROJECT_CRS)
            
            # Agricultural
            if any('agricultural' in f for f in missing_features):
                master_gdf = calculate_land_use_percentages(
                    master_gdf, clc_gdf, 'code_18', # Assuming 'code_18' is the column for CLC codes
                    LAND_USE_CATEGORY_MAP['agricultural']['codes'],
                    'pct_area_agricultural'
                )
            
            # Natural / Protected (Approximation using CLC)
            if any('natural_protected' in f for f in missing_features):
                master_gdf = calculate_land_use_percentages(
                    master_gdf, clc_gdf, 'code_18',
                    LAND_USE_CATEGORY_MAP['natural_protected']['codes'],
                    'pct_area_protected_natural'
                )
        else:
            logging.error(f"[REDACTED_BY_SCRIPT]")

    # 3. Urban Fabric (Combined)
    # This is complex. For now, we will implement a simplified version using just OSM Landuse 'residential' + 'commercial' + 'retail'
    # If OSM Buildings is available, we could add it, but 'Smart Skip' implies we likely won't run this often.
    if any('urban_fabric' in f for f in missing_features):
        if os.path.exists(OSM_LAND_USE_PATH):
             # We might have already loaded it, but for simplicity/robustness we reload or check if variable exists (not easy in this scope).
             # We'[REDACTED_BY_SCRIPT]'s fast enough for this fallback path.
             # Optimization: Check if we can reuse osm_landuse from step 1? 
             # Since we didn't persist it in a wider scope, we reload.
             logging.info(f"[REDACTED_BY_SCRIPT]")
             osm_landuse = gpd.read_file(OSM_LAND_USE_PATH)
             if osm_landuse.crs.to_string() != PROJECT_CRS:
                osm_landuse = osm_landuse.to_crs(PROJECT_CRS)

             master_gdf = calculate_land_use_percentages(
                master_gdf, osm_landuse, 'fclass',
                LAND_USE_CATEGORY_MAP['urban_fabric']['osm_fclasses'],
                'pct_area_urban_fabric'
            )
        else:
             logging.error("[REDACTED_BY_SCRIPT]")

    logging.info("[REDACTED_BY_SCRIPT]")
    return master_gdf

