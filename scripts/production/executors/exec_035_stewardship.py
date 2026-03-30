import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
CS_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "data", "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2000, 5000, 10000]
NULL_SENTINEL_FLOAT = 0.0

# --- Module-level State for Performance ---
gdf_cs, sindex_cs = None, None
gdf_cs_hightier, sindex_cs_hightier = None, None
gdf_cs_midtier, sindex_cs_midtier = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_cs, sindex_cs, gdf_cs_hightier, sindex_cs_hightier, gdf_cs_midtier, sindex_cs_midtier
    if gdf_cs is not None:
        return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load the sanitized L1 artifact
        gdf_cs_local = gpd.read_file(CS_L1_ARTIFACT)
        assert gdf_cs_local.crs.to_string() == PROJECT_CRS, "[REDACTED_BY_SCRIPT]"
        
        # Pre-process for performance: Ensure dates are datetime objects for fast filtering
        gdf_cs_local['startdate'] = pd.to_datetime(gdf_cs_local['startdate'], errors='coerce')
        gdf_cs_local['enddate'] = pd.to_datetime(gdf_cs_local['enddate'], errors='coerce')

        # Pre-create dedicated, indexed subsets for tier-specific queries
        gdf_cs_hightier_local = gdf_cs_local[gdf_cs_local['cs_tier'] == 'Higher Tier'].copy()
        gdf_cs_midtier_local = gdf_cs_local[gdf_cs_local['cs_tier'] == 'Mid Tier'].copy()
        
        # Set final module state atomically
        gdf_cs = gdf_cs_local
        sindex_cs = gdf_cs.sindex
        gdf_cs_hightier = gdf_cs_hightier_local
        sindex_cs_hightier = gdf_cs_hightier.sindex
        gdf_cs_midtier = gdf_cs_midtier_local
        sindex_cs_midtier = gdf_cs_midtier.sindex

        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_cs = "INIT_FAILED"

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates all temporally-aware Countryside Stewardship features for the entire master GeoDataFrame.
    This executor is designed to handle temporal queries based on an 'application_date' column.
    """
    if sindex_cs is None:
        _initialize_module_state()
    if sindex_cs == "INIT_FAILED":
        logging.error("[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Temporal Guard ---
    if 'application_date' not in master_gdf.columns:
        logging.warning("EXEC_CS_WARN: 'application_date'[REDACTED_BY_SCRIPT]")
        master_gdf['application_date'] = pd.to_datetime("2025-11-15")
    else:
        master_gdf['application_date'] = pd.to_datetime(master_gdf['application_date'], errors='coerce')

    # --- Feature Initialization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf['cs_on_site_bool'] = 0
    master_gdf['cs_on_site_area_ha'] = 0.0
    master_gdf['cs_on_site_pct_area'] = 0.0
    master_gdf['cs_on_site_total_value'] = 0.0
    master_gdf['cs_on_site_highest_tier'] = 'None'
    for r in RADII_METERS:
        r_km = r // 1000
        master_gdf[f'cs_count_{r_km}km'] = 0
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = 0.0
        master_gdf[f'cs_density_{r_km}km'] = 0.0
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = 0.0
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = 0
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = 0

    # --- Row-wise Processing for Temporal Features ---
    # A row-wise approach is necessary here because each grid point has a unique
    # application_date, requiring a point-in-time analysis of active CS agreements.
    logging.info("[REDACTED_BY_SCRIPT]")
    
    def calculate_cs_features_for_row(row):
        app_date = row['application_date']
        geom = row['geometry']
        
        # If no valid application date, return default features (already set).
        if pd.isnull(app_date):
            return row

        # Create point-in-time snapshots of active agreements for this specific date
        active_cs = gdf_cs[(gdf_cs['startdate'] <= app_date) & (gdf_cs['enddate'] > app_date)]
        if active_cs.empty:
            return row # No active agreements, no features to add.
        
        active_cs_hightier = gdf_cs_hightier[(gdf_cs_hightier['startdate'] <= app_date) & (gdf_cs_hightier['enddate'] > app_date)]
        active_cs_midtier = gdf_cs_midtier[(gdf_cs_midtier['startdate'] <= app_date) & (gdf_cs_midtier['enddate'] > app_date)]

        # --- Direct Impact Features ---
        intersecting_indices = active_cs.sindex.query(geom, predicate='intersects')
        if intersecting_indices.size > 0:
            intersecting_cs = active_cs.iloc[intersecting_indices]
            row['cs_on_site_bool'] = 1
            
            # Use gpd.overlay for accurate area calculation
            intersection_geo = gpd.overlay(
                gpd.GeoDataFrame(geometry=[geom], crs=PROJECT_CRS), 
                intersecting_cs, 
                how='intersection'
            )
            row['cs_on_site_area_ha'] = intersection_geo.area.sum() / 10000
            row['cs_on_site_pct_area'] = (intersection_geo.area.sum() / geom.area * 100) if geom.area > 0 else 0
            row['cs_on_site_total_value'] = intersecting_cs['totval_no'].sum()
            
            tier_order = pd.CategoricalDtype(['None', 'Other', 'Capital Grants', 'Mid Tier', 'Higher Tier'], ordered=True)
            row['cs_on_site_highest_tier'] = intersecting_cs['cs_tier'].astype(tier_order).max()

        # --- Risk Gradient Features (Multi-Radii) ---
        for r in RADII_METERS:
            r_km = r // 1000
            buffer_geom = geom.buffer(r)
            
            # Query against the point-in-time active dataframes
            cs_in_buffer_indices = active_cs.sindex.query(buffer_geom, predicate='intersects')
            if cs_in_buffer_indices.size > 0:
                cs_in_buffer = active_cs.iloc[cs_in_buffer_indices]
                row[f'cs_count_{r_km}km'] = len(cs_in_buffer)
                # Accurate area of intersection
                intersected_area = gpd.overlay(gpd.GeoDataFrame(geometry=[buffer_geom], crs=PROJECT_CRS), cs_in_buffer, how='intersection').area.sum()
                row[f'[REDACTED_BY_SCRIPT]'] = intersected_area / 10000
                row[f'cs_density_{r_km}km'] = (intersected_area / buffer_geom.area) if buffer_geom.area > 0 else 0
                row[f'[REDACTED_BY_SCRIPT]'] = cs_in_buffer['totval_no'].mean()

            ht_in_buffer_indices = active_cs_hightier.sindex.query(buffer_geom, predicate='intersects')
            row[f'[REDACTED_BY_SCRIPT]'] = ht_in_buffer_indices.size
            
            mt_in_buffer_indices = active_cs_midtier.sindex.query(buffer_geom, predicate='intersects')
            row[f'[REDACTED_BY_SCRIPT]'] = mt_in_buffer_indices.size
        
        return row

    # Apply the function row-wise. This is necessary due to the temporal component.
    # Using .progress_apply for visibility on long runs.
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
        updated_gdf = master_gdf.progress_apply(calculate_cs_features_for_row, axis=1)
    except ImportError:
        updated_gdf = master_gdf.apply(calculate_cs_features_for_row, axis=1)


    # --- Finalization and Cleanup ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # Ensure correct dtypes and fill any potential NaNs that slipped through
    final_fill_values = {
        'cs_avg_value_2km': 0.0,
        'cs_avg_value_5km': 0.0,
        'cs_avg_value_10km': 0.0,
        'cs_on_site_highest_tier': 'None'
    }
    updated_gdf.fillna(final_fill_values, inplace=True)
    
    # Set dtype for tier to allow for ordering/analysis later
    tier_order = pd.CategoricalDtype(['None', 'Other', 'Capital Grants', 'Mid Tier', 'Higher Tier'], ordered=True)
    updated_gdf['cs_on_site_highest_tier'] = updated_gdf['cs_on_site_highest_tier'].astype(tier_order)

    logging.info("[REDACTED_BY_SCRIPT]")
    return updated_gdf