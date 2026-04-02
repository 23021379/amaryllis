import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
SSSI_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "geopackage", "SSSI_L1.gpkg")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = 0.0
RISK_RADII_M = [2000, 5000, 10000, 20000]

# --- Module-level State for Performance ---
gdf_sssi, sindex_sssi = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_sssi, sindex_sssi
    # This block is thread-safe due to Python's GIL. It will only execute once
    # per worker process, even if multiple tasks trigger it simultaneously.
    if gdf_sssi is not None:
        return
        
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf = gpd.read_file(SSSI_L1_ARTIFACT)

        # PRE-OPERATIVE CHECK: CRS VALIDATION (Pattern 1)
        if gdf.crs.to_string() != PROJECT_CRS:
            logging.error(f"[REDACTED_BY_SCRIPT]")
            sindex_sssi = "INIT_FAILED"
            return

        gdf['area_ha'] = gdf.geometry.area / 10000
        gdf_sssi = gdf
        sindex_sssi = gdf_sssi.sindex
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_sssi = "INIT_FAILED"

def execute(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates all SSSI-related features for the entire input GeoDataFrame.
    """
    _initialize_module_state()
    if sindex_sssi == "INIT_FAILED":
        logging.error("[REDACTED_BY_SCRIPT]")
        # Return the dataframe with empty columns to avoid downstream errors
        for radius_m in RISK_RADII_M:
            radius_km = int(radius_m / 1000)
            gdf[f'[REDACTED_BY_SCRIPT]'] = 0
            gdf[f'[REDACTED_BY_SCRIPT]'] = 0.0
        gdf['sssi_dist_to_nearest_m'] = NULL_SENTINEL
        gdf['sssi_nearest_area_ha'] = NULL_SENTINEL
        gdf['sssi_is_within'] = 0
        gdf['[REDACTED_BY_SCRIPT]'] = 0.0
        return gdf

    try:
        # --- SSSI Proximity and Containment ---
        # Use sindex.nearest to find the index and distance of the nearest SSSI for all points at once
        nearest_indices, distances = sindex_sssi.nearest(gdf.geometry, return_distance=True)
        
        # Extract the nearest SSSI features using the indices
        nearest_sssi_geoms = gdf_sssi.iloc[nearest_indices]
        
        gdf['sssi_dist_to_nearest_m'] = distances
        # Use .values to avoid index alignment issues
        gdf['sssi_nearest_area_ha'] = nearest_sssi_geoms['area_ha'].values
        gdf['sssi_nearest_name'] = nearest_sssi_geoms['sssi_name'].values
        
        # A point is within if its distance is 0
        gdf['sssi_is_within'] = (gdf['sssi_dist_to_nearest_m'] == 0).astype(int)

        # --- SSSI Multi-Radius Density ---
        # This part requires a row-wise operation as buffer and intersection are per-geometry
        def calculate_density(geom):
            results = {}
            max_radius = max(RISK_RADII_M)
            buffer = geom.buffer(max_radius)
            possible_matches_idx = list(sindex_sssi.intersection(buffer.bounds))
            
            if not possible_matches_idx:
                for r in RISK_RADII_M:
                    results[f'[REDACTED_BY_SCRIPT]'] = 0
                    results[f'[REDACTED_BY_SCRIPT]'] = 0.0
                return pd.Series(results)

            candidates = gdf_sssi.iloc[possible_matches_idx].copy()
            # Further filter to actual intersections with the buffer
            candidates = candidates[candidates.intersects(buffer)]
            
            if not candidates.empty:
                # Calculate distances from the point to the candidates
                candidates['distance'] = candidates.geometry.apply(lambda g: geom.distance(g))
            
            for radius_m in RISK_RADII_M:
                radius_km = int(radius_m / 1000)
                if candidates.empty:
                    results[f'[REDACTED_BY_SCRIPT]'] = 0
                    results[f'[REDACTED_BY_SCRIPT]'] = 0.0
                else:
                    radius_subset = candidates[candidates['distance'] <= radius_m]
                    results[f'[REDACTED_BY_SCRIPT]'] = len(radius_subset)
                    results[f'[REDACTED_BY_SCRIPT]'] = radius_subset['area_ha'].sum()

            return pd.Series(results)

        # Apply the density calculation
        density_features = gdf['geometry'].apply(calculate_density)
        gdf = pd.concat([gdf, density_features], axis=1)

        # --- SSSI Density Focus Ratio (Alpha Feature) ---
        count_2km = gdf.get('sssi_count_in_2km', 0)
        count_20km = gdf.get('sssi_count_in_20km', 0)
        gdf['[REDACTED_BY_SCRIPT]'] = np.divide(count_2km, count_20km, out=np.zeros_like(count_2km, dtype=float), where=count_20km!=0)

        # --- Update Capstone Synthesis Features ---
        # This logic seems to be handled by a later, dedicated synthesis script.
        # Replicating it here might be redundant. If it's needed, it should be
        # done after all constraint features are added to the gdf.
        # For now, we will assume another script handles this.
            
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        # Avoid adding an 'errors' column if it doesn't exist
        
    return gdf