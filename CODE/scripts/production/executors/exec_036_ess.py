import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
ES_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "geopackage", "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2000, 5000, 10000]
NULL_SENTINEL_FLOAT = 0.0
NULL_SENTINEL_INT = 0

# --- Module-level State for Performance ---
gdf_es, sindex_es = None, None
gdf_hls, sindex_hls = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_es, sindex_es, gdf_hls, sindex_hls
    if gdf_es is not None:
        return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load the sanitized L1 artifact
        gdf_es_local = gpd.read_file(ES_L1_ARTIFACT)
        assert gdf_es_local.crs.to_string() == PROJECT_CRS, "[REDACTED_BY_SCRIPT]"
        
        # Pre-process for performance: Ensure dates are datetime objects for fast filtering
        gdf_es_local['startdat'] = pd.to_datetime(gdf_es_local['startdat'], errors='coerce')
        gdf_es_local['enddate'] = pd.to_datetime(gdf_es_local['enddate'], errors='coerce')

        # Pre-create a dedicated, indexed subset for HLS-specific queries
        gdf_hls_local = gdf_es_local[gdf_es_local['es_tier'] == 'HLS'].copy()
        
        # Set final module state atomically
        gdf_es = gdf_es_local
        sindex_es = gdf_es.sindex
        gdf_hls = gdf_hls_local
        sindex_hls = gdf_hls.sindex

        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_es = "INIT_FAILED"

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates all temporally-aware Environmental Stewardship features for the entire master GeoDataFrame.
    """
    if sindex_es is None:
        _initialize_module_state()
    if sindex_es == "INIT_FAILED":
        logging.error("[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Temporal Guard ---
    if 'application_date' not in master_gdf.columns:
        logging.warning("EXEC_ES_WARN: 'application_date'[REDACTED_BY_SCRIPT]")
        master_gdf['application_date'] = pd.to_datetime("2025-11-15")
    else:
        master_gdf['application_date'] = pd.to_datetime(master_gdf['application_date'], errors='coerce')

    # --- Feature Initialization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf['es_hls_on_site_pct_area'] = 0.0
    master_gdf['es_on_site_highest_tier'] = 'None'
    for r in RADII_METERS:
        r_km = r // 1000
        master_gdf[f'es_count_{r_km}km'] = 0
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = 0
        master_gdf[f'es_total_area_ha_{r_km}km'] = 0.0
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = 0.0
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = 0.0

    # --- Row-wise Processing for Temporal Features ---
    logging.info("[REDACTED_BY_SCRIPT]")

    def calculate_es_features_for_row(row):
        app_date = row['application_date']
        geom = row['geometry']

        if pd.isnull(app_date):
            return row

        # Create point-in-time snapshots of active agreements
        active_es = gdf_es[(gdf_es['startdat'] <= app_date) & (gdf_es['enddate'] > app_date)]
        if active_es.empty:
            return row
        
        active_hls = gdf_hls[(gdf_hls['startdat'] <= app_date) & (gdf_hls['enddate'] > app_date)]

        # --- Direct Impact Features ---
        intersecting_es_indices = active_es.sindex.query(geom, predicate='intersects')
        intersecting_hls_indices = active_hls.sindex.query(geom, predicate='intersects')

        row['es_on_site_bool'] = 1 if intersecting_es_indices.size > 0 else 0
        row['es_hls_on_site_bool'] = 1 if intersecting_hls_indices.size > 0 else 0
        
        if row['es_hls_on_site_bool']:
            intersecting_hls_polys = active_hls.iloc[intersecting_hls_indices]
            hls_intersection_area = gpd.overlay(
                gpd.GeoDataFrame(geometry=[geom], crs=PROJECT_CRS), 
                intersecting_hls_polys, how='intersection'
            ).area.sum()
            row['es_hls_on_site_pct_area'] = (hls_intersection_area / geom.area * 100) if geom.area > 0 else 0
            row['es_on_site_highest_tier'] = 'HLS'
        elif row['es_on_site_bool']:
            row['es_on_site_highest_tier'] = 'ELS'
        
        # --- Risk Gradient Features ---
        for r in RADII_METERS:
            r_km = r // 1000
            buffer_geom = geom.buffer(r)
            
            es_in_buffer_indices = active_es.sindex.query(buffer_geom, predicate='intersects')
            hls_in_buffer_indices = active_hls.sindex.query(buffer_geom, predicate='intersects')
            
            row[f'es_count_{r_km}km'] = es_in_buffer_indices.size
            row[f'[REDACTED_BY_SCRIPT]'] = hls_in_buffer_indices.size
            
            if es_in_buffer_indices.size > 0:
                es_in_buffer = active_es.iloc[es_in_buffer_indices]
                intersected_area = gpd.overlay(gpd.GeoDataFrame(geometry=[buffer_geom], crs=PROJECT_CRS), es_in_buffer, how='intersection').area.sum()
                row[f'es_total_area_ha_{r_km}km'] = intersected_area / 10000
            
            if hls_in_buffer_indices.size > 0:
                hls_in_buffer = active_hls.iloc[hls_in_buffer_indices]
                intersected_hls_area = gpd.overlay(gpd.GeoDataFrame(geometry=[buffer_geom], crs=PROJECT_CRS), hls_in_buffer, how='intersection').area.sum()
                row[f'[REDACTED_BY_SCRIPT]'] = (intersected_hls_area / buffer_geom.area) if buffer_geom.area > 0 else 0
                row[f'[REDACTED_BY_SCRIPT]'] = hls_in_buffer['totcost'].mean()
        
        return row

    try:
        from tqdm import tqdm
        tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
        updated_gdf = master_gdf.progress_apply(calculate_es_features_for_row, axis=1)
    except ImportError:
        updated_gdf = master_gdf.apply(calculate_es_features_for_row, axis=1)

    # --- Finalization and Cleanup ---
    logging.info("[REDACTED_BY_SCRIPT]")
    final_fill_values = {
        'es_hls_avg_cost_2km': 0.0,
        'es_hls_avg_cost_5km': 0.0,
        '[REDACTED_BY_SCRIPT]': 0.0,
        'es_on_site_highest_tier': 'None'
    }
    updated_gdf.fillna(final_fill_values, inplace=True)
    
    tier_order = pd.CategoricalDtype(['None', 'ELS', 'HLS'], ordered=True)
    updated_gdf['es_on_site_highest_tier'] = updated_gdf['es_on_site_highest_tier'].astype(tier_order)

    logging.info("[REDACTED_BY_SCRIPT]")
    return updated_gdf