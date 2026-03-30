import logging
import numpy as np
import geopandas as gpd
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
NHLC_SOURCE_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2000, 5000, 10000]
URBAN_TYPES = ['SETTLEMENT', 'COMMERCE', 'INDUSTRY', 'CIVIC PROVISION']
NULL_SENTINEL_FLOAT = 0.0
NULL_SENTINEL_INT = 0

# --- Module-level State for Performance ---
gdf_urban, sindex_urban = None, None
gdf_settlement, sindex_settlement = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_urban, sindex_urban, gdf_settlement, sindex_settlement
    if gdf_urban is not None:
        return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Ingest raw data
        nhlc_gdf = gpd.read_file(NHLC_SOURCE_ARTIFACT)
        assert nhlc_gdf.crs.to_string() == PROJECT_CRS, "[REDACTED_BY_SCRIPT]"
        
        # Sanitize
        nhlc_gdf.columns = nhlc_gdf.columns.str.lower()
        nhlc_gdf.dropna(subset=['geometry'], inplace=True)
        nhlc_gdf = nhlc_gdf[nhlc_gdf.is_valid]

        # The Surgical Triage: Isolate Urban Proxies and discard the rest
        gdf_urban_local = nhlc_gdf[nhlc_gdf['dominantbroadtype'].isin(URBAN_TYPES)].copy()
        gdf_settlement_local = gdf_urban_local[gdf_urban_local['dominantbroadtype'] == 'SETTLEMENT'].copy()
        
        # Set final module state atomically
        gdf_urban = gdf_urban_local
        sindex_urban = gdf_urban.sindex
        gdf_settlement = gdf_settlement_local
        sindex_settlement = gdf_settlement.sindex

        logging.info(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_urban = "INIT_FAILED"

def execute(master_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_urban is None:
        _initialize_module_state()
    if sindex_urban == "INIT_FAILED":
        logging.error("[REDACTED_BY_SCRIPT]")
        return master_gdf

    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Feature Initialization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    master_gdf['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL_INT
    master_gdf['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL_FLOAT
    for r in RADII_METERS:
        r_km = r // 1000
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL_INT
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL_FLOAT
        master_gdf[f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL_FLOAT

    # --- Row-wise Processing ---
    logging.info("[REDACTED_BY_SCRIPT]")

    def calculate_nhlc_features_for_row(row):
        geom = row['geometry']

        # --- Direct Impact Features ---
        intersecting_indices = sindex_urban.query(geom, predicate='intersects')
        if intersecting_indices.size > 0:
            row['[REDACTED_BY_SCRIPT]'] = 1
            intersecting_polys = gdf_urban.iloc[intersecting_indices]
            
            intersection_geo = gpd.overlay(
                gpd.GeoDataFrame(geometry=[geom], crs=PROJECT_CRS),
                intersecting_polys, how='intersection'
            )
            row['[REDACTED_BY_SCRIPT]'] = (intersection_geo.area.sum() / geom.area * 100) if geom.area > 0 else 0
        else:
            row['[REDACTED_BY_SCRIPT]'] = 0
            row['[REDACTED_BY_SCRIPT]'] = 0.0

        # --- Risk Gradient Features ---
        for r in RADII_METERS:
            r_km = r // 1000
            buffer_geom = geom.buffer(r)
            buffer_area = buffer_geom.area

            # All urban types
            urban_candidates_idx = list(sindex_urban.intersection(buffer_geom.bounds))
            if urban_candidates_idx:
                urban_candidates = gdf_urban.iloc[urban_candidates_idx]
                urban_in_buffer = urban_candidates[urban_candidates.intersects(buffer_geom)]
                row[f'[REDACTED_BY_SCRIPT]'] = len(urban_in_buffer)
                if not urban_in_buffer.empty:
                    urban_area = gpd.overlay(gpd.GeoDataFrame(geometry=[buffer_geom], crs=PROJECT_CRS), urban_in_buffer, how='intersection').area.sum()
                    row[f'[REDACTED_BY_SCRIPT]'] = urban_area / buffer_area if buffer_area > 0 else 0.0

            # Settlement-only
            settlement_candidates_idx = list(sindex_settlement.intersection(buffer_geom.bounds))
            if settlement_candidates_idx:
                settlement_candidates = gdf_settlement.iloc[settlement_candidates_idx]
                settlement_in_buffer = settlement_candidates[settlement_candidates.intersects(buffer_geom)]
                if not settlement_in_buffer.empty:
                    settlement_area = gpd.overlay(gpd.GeoDataFrame(geometry=[buffer_geom], crs=PROJECT_CRS), settlement_in_buffer, how='intersection').area.sum()
                    row[f'[REDACTED_BY_SCRIPT]'] = settlement_area / buffer_area if buffer_area > 0 else 0.0
        
        return row

    try:
        from tqdm import tqdm
        tqdm.pandas(desc="NHLC Urban Analysis")
        updated_gdf = master_gdf.progress_apply(calculate_nhlc_features_for_row, axis=1)
    except ImportError:
        updated_gdf = master_gdf.apply(calculate_nhlc_features_for_row, axis=1)

    # --- Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    # The .apply method creates new columns, so we just need to ensure NaNs are filled if any logic path failed.
    # The initialization already handles the default cases.
    updated_gdf.fillna({col: 0 for col in updated_gdf.columns if 'nhlc_' in col}, inplace=True)

    logging.info("[REDACTED_BY_SCRIPT]")
    return updated_gdf