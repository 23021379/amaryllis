import logging
import numpy as np
import geopandas as gpd
import pandas as pd
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
NP_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "geopackage", "[REDACTED_BY_SCRIPT]")
LPA_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "geospatial", "boundaries", "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2000, 5000, 10000, 20000]
NULL_SENTINEL_FLOAT = 0.0
NULL_SENTINEL_INT = 0

# --- Module-level State for Performance ---
gdf_np, sindex_np = None, None
gdf_lpa, sindex_lpa = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_np, sindex_np, gdf_lpa, sindex_lpa
    if gdf_np is not None:
        return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # 1. Load National Parks data
        gdf_np_local = gpd.read_file(NP_L1_ARTIFACT)
        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        assert gdf_np_local.crs.to_string() == PROJECT_CRS, "[REDACTED_BY_SCRIPT]"
        gdf_np_local['np_area_sqkm'] = gdf_np_local.geometry.area / 1_000_000

        # 2. Load LPA data and perform one-time coverage calculation
        gdf_lpa_local = gpd.read_file(LPA_L1_ARTIFACT)
        if gdf_lpa_local.crs.to_string() != PROJECT_CRS:
            gdf_lpa_local = gdf_lpa_local.to_crs(PROJECT_CRS)
        
        gdf_lpa_local['lpa_area_sqkm'] = gdf_lpa_local.geometry.area / 1_000_000
        np_lpa_intersection = gpd.overlay(gdf_lpa_local, gdf_np_local, how='intersection', keep_geom_type=False)
        np_lpa_intersection['[REDACTED_BY_SCRIPT]'] = np_lpa_intersection.geometry.area / 1_000_000
        lpa_coverage = np_lpa_intersection.groupby('LPA23NM')['[REDACTED_BY_SCRIPT]'].sum()
        
        gdf_lpa_local = gdf_lpa_local.join(lpa_coverage.rename('np_coverage_sqkm'), on='LPA23NM')
        gdf_lpa_local['lpa_np_coverage_pct'] = (gdf_lpa_local['np_coverage_sqkm'] / gdf_lpa_local['lpa_area_sqkm'] * 100).fillna(0)
        
        # Set final module state atomically
        gdf_np = gdf_np_local.reset_index(drop=True)
        sindex_np = gdf_np.sindex
        gdf_lpa = gdf_lpa_local.reset_index(drop=True)
        sindex_lpa = gdf_lpa.sindex

        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        # Mark as failed to prevent execution
        sindex_np = "INIT_FAILED"

def execute(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    _initialize_module_state()
    if sindex_np == "INIT_FAILED":
        logging.error("[REDACTED_BY_SCRIPT]")
        # Add empty columns to prevent downstream errors
        gdf['np_dist_to_nearest_m'] = NULL_SENTINEL_FLOAT
        gdf['np_nearest_name'] = 'UNKNOWN'
        gdf['np_nearest_area_sqkm'] = NULL_SENTINEL_FLOAT
        gdf['np_is_within'] = NULL_SENTINEL_INT
        for r in RADII_METERS:
            r_km = r // 1000
            gdf[f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL_FLOAT
            gdf[f'np_count_in_{r_km}km'] = NULL_SENTINEL_INT
        gdf['lpa_np_coverage_pct'] = NULL_SENTINEL_FLOAT
        return gdf

    try:
        # --- Proximity & Direct Impact Features (Vectorized) ---
        nearest_indices, distances = sindex_np.nearest(gdf.geometry, return_distance=True)
        
        nearest_nps = gdf_np.iloc[nearest_indices]
        
        gdf['np_dist_to_nearest_m'] = distances
        gdf['np_nearest_name'] = nearest_nps['name'].values
        gdf['np_nearest_area_sqkm'] = nearest_nps['np_area_sqkm'].values
        gdf['np_is_within'] = (gdf['np_dist_to_nearest_m'] == 0).astype(int)

        # --- LPA Context Feature (Vectorized with sjoin) ---
        # Perform a spatial join to find which LPA each point falls into
        gdf_with_lpa = gpd.sjoin(gdf, gdf_lpa[['lpa_np_coverage_pct', 'geometry']], how='left', predicate='intersects')
        # The join might create duplicates if a point intersects multiple LPA boundaries (unlikely for points)
        # We drop duplicates, keeping the first match, and retain the original index.
        gdf_with_lpa = gdf_with_lpa[~gdf_with_lpa.index.duplicated(keep='first')]
        gdf['lpa_np_coverage_pct'] = gdf_with_lpa['lpa_np_coverage_pct'].fillna(NULL_SENTINEL_FLOAT)

        # --- Multi-Radii Density (Row-wise apply) ---
        def calculate_density(geom):
            results = {}
            max_radius = max(RADII_METERS)
            buffer = geom.buffer(max_radius)
            possible_matches_idx = list(sindex_np.intersection(buffer.bounds))
            
            if not possible_matches_idx:
                for r in RADII_METERS:
                    r_km = r // 1000
                    results[f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL_FLOAT
                    results[f'np_count_in_{r_km}km'] = NULL_SENTINEL_INT
                return pd.Series(results)

            candidates = gdf_np.iloc[possible_matches_idx]
            
            for r in RADII_METERS:
                r_km = r // 1000
                radius_buffer = geom.buffer(r)
                # Filter candidates that actually intersect the smaller radius buffer
                intersecting_nps = candidates[candidates.intersects(radius_buffer)]
                
                total_area_in_radius = 0
                if not intersecting_nps.empty:
                    # Use vectorized intersection
                    intersections = intersecting_nps.geometry.intersection(radius_buffer)
                    total_area_in_radius = intersections.area.sum()

                results[f'[REDACTED_BY_SCRIPT]'] = total_area_in_radius / 1_000_000
                results[f'np_count_in_{r_km}km'] = len(intersecting_nps)
            return pd.Series(results)

        density_features = gdf['geometry'].apply(calculate_density)
        gdf = pd.concat([gdf, density_features], axis=1)

        # --- Capstone Synthesis Update ---
        # This is better handled by a dedicated synthesis script after all individual
        # constraint features have been calculated. We will assume a later script
        # will calculate '[REDACTED_BY_SCRIPT]' and '[REDACTED_BY_SCRIPT]'.

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")

    return gdf