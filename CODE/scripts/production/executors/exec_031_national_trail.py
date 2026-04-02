import logging
import numpy as np
import geopandas as gpd
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
NT_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "geopackage", "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
RADII_METERS = [2000, 5000, 10000, 20000]
NULL_SENTINEL_FLOAT = -1.0

# --- Module-level State for Performance ---
gdf_nt, sindex_nt = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_nt, sindex_nt
    if gdf_nt is not None:
        return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        gdf_nt_local = gpd.read_file(NT_L1_ARTIFACT)
        assert gdf_nt_local.crs.to_string() == PROJECT_CRS, "[REDACTED_BY_SCRIPT]"
        
        # Set final module state atomically
        gdf_nt = gdf_nt_local
        sindex_nt = gdf_nt.sindex

        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sindex_nt = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if sindex_nt is None:
        _initialize_module_state()
    if sindex_nt == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        geom = state['input_geom']
        features = state['features']

        # --- Proximity & Direct Impact Features ---
        nearest_idx = sindex_nt.nearest(geom, return_all=False)[1][0]
        nearest_trail = gdf_nt.iloc[nearest_idx]
        
        features['nt_dist_to_nearest_m'] = geom.distance(nearest_trail.geometry)
        features['nt_nearest_name'] = nearest_trail.get('nt_name', 'UNKNOWN')
        
        # Direct intersection check
        intersecting_indices = sindex_nt.query(geom, predicate='intersects')
        features['nt_intersects_site_bool'] = 1 if intersecting_indices.size > 0 else 0

        # --- Multi-Radii Density (Surgical Clip Method) ---
        max_radius = max(RADII_METERS)
        buffer = geom.buffer(max_radius)
        possible_matches_idx = list(sindex_nt.intersection(buffer.bounds))
        candidates = gdf_nt.iloc[possible_matches_idx]

        for r in RADII_METERS:
            r_km = r // 1000
            radius_buffer = geom.buffer(r)
            
            intersecting_trails = candidates[candidates.intersects(radius_buffer)]
            
            if not intersecting_trails.empty:
                # Clip intersecting trails to the buffer and calculate length
                clipped_trails = gpd.clip(intersecting_trails, radius_buffer)
                total_length_m = clipped_trails.geometry.length.sum()
                
                features[f'nt_length_in_{r_km}km'] = total_length_m / 1000
                features[f'[REDACTED_BY_SCRIPT]'] = len(intersecting_trails)
            else:
                features[f'nt_length_in_{r_km}km'] = 0.0
                features[f'[REDACTED_BY_SCRIPT]'] = 0

        # --- Capstone Synthesis Update ---
        constraints = {
            'AncientWoodland': features.get('aw_dist_to_nearest_m', np.inf),
            'AONB': features.get('aonb_dist_to_nearest_m', np.inf),
            'SSSI': features.get('sssi_dist_to_nearest_m', np.inf),
            'NationalPark': features.get('np_dist_to_nearest_m', np.inf),
            'SAC': features.get('sac_dist_to_nearest_m', np.inf),
            'SPA': features.get('spa_dist_to_nearest_m', np.inf),
            'NationalTrail': features.get('nt_dist_to_nearest_m', np.inf)
        }
        min_dist = min(constraints.values())
        features['[REDACTED_BY_SCRIPT]'] = min_dist if min_dist != np.inf else NULL_SENTINEL_FLOAT
        if min_dist == np.inf:
            features['[REDACTED_BY_SCRIPT]'] = 'None'
        else:
            features['[REDACTED_BY_SCRIPT]'] = min(constraints, key=constraints.get)

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)

    return state