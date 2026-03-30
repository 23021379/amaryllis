import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
DNOA_POINTS_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PRIMARY_SUB_DNOA_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
K_NEIGHBORS = 20
TEMPORAL_WINDOW_YEARS = 3
NULL_SENTINEL = 0.0 # Counts and ratios default to 0

# --- Module-level State for Performance ---
gdf_dnoa, gdf_primary_subs = None, None
sindex_primary_subs = None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_dnoa, gdf_primary_subs, sindex_primary_subs
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # Load the raw DNOA points. A full spatial index is not built here,
        # as it will be built on-the-fly for temporally-correct subsets.
        gdf_dnoa = gpd.read_file(DNOA_POINTS_L1_ARTIFACT, engine='pyarrow')
        assert gdf_dnoa.crs.to_string() == PROJECT_CRS
        
        # Load pre-aggregated primary substation DNOA profiles for fast lookups
        gdf_primary_subs = gpd.read_file(PRIMARY_SUB_DNOA_L1_ARTIFACT, engine='pyarrow')
        assert gdf_primary_subs.crs.to_string() == PROJECT_CRS
        sindex_primary_subs = gdf_primary_subs.sindex
        
        logging.info("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        gdf_dnoa = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if gdf_dnoa is None:
        _initialize_module_state()
    if isinstance(gdf_dnoa, str) and gdf_dnoa == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    knn_feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'dnoa_avg_dist_knn_m', 'dnoa_avg_deferred_kva_knn',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    primary_sub_feature_keys = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    all_feature_keys = knn_feature_keys + primary_sub_feature_keys
    
    try:
        geom = state['input_geom']
        submission_year = state['submission_year']

        # --- Point-in-Time k-NN Features ---
        # CRITICAL TEMPORAL GUARDRAIL (Pattern 3: Feature Leakage)
        dnoa_relevant = gdf_dnoa[
            (gdf_dnoa['constraint_year'] >= submission_year) &
            (gdf_dnoa['constraint_year'] <= submission_year + TEMPORAL_WINDOW_YEARS)
        ]

        if not dnoa_relevant.empty:
            # Build a temporary spatial index on the small, temporally-correct subset for performance
            relevant_coords = np.deg2rad(np.array(list(zip(dnoa_relevant.geometry.y, dnoa_relevant.geometry.x))))
            temp_tree = BallTree(relevant_coords, metric='haversine')
            
            app_coords_rad = np.deg2rad([[geom.y, geom.x]])
            k = min(K_NEIGHBORS, len(dnoa_relevant))
            distances_rad, indices = temp_tree.query(app_coords_rad, k=k)
            
            # Convert distances from radians to meters
            distances_m = distances_rad[0] * 6371000 # Earth radius in meters
            knn_cohort = dnoa_relevant.iloc[indices[0]]

            # Synthesize k-NN features
            nearest = knn_cohort.iloc[0]
            state['features']['[REDACTED_BY_SCRIPT]'] = distances_m[0]
            state['features']['[REDACTED_BY_SCRIPT]'] = nearest['constraint_year'] - submission_year
            state['features']['[REDACTED_BY_SCRIPT]'] = len(knn_cohort)
            state['features']['dnoa_avg_dist_knn_m'] = np.mean(distances_m)
            state['features']['dnoa_avg_deferred_kva_knn'] = knn_cohort['[REDACTED_BY_SCRIPT]'].mean()
            state['features']['[REDACTED_BY_SCRIPT]'] = knn_cohort['constraint_year'].min()
            # The one-hot encoded column name is derived from the L1 artifact creation spec
            flex_col = '[REDACTED_BY_SCRIPT]'
            state['features']['[REDACTED_BY_SCRIPT]'] = knn_cohort[flex_col].mean() if flex_col in knn_cohort else 0.0
        else:
            for key in knn_feature_keys: state['features'][key] = NULL_SENTINEL

        # --- Primary Substation Aggregation Features (Lookup) ---
        nearest_idx = list(sindex_primary_subs.nearest(geom))
        if nearest_idx:
            nearest_sub_profile = gdf_primary_subs.iloc[nearest_idx[0]]
            for key in primary_sub_feature_keys:
                state['features'][key] = nearest_sub_profile.get(key, NULL_SENTINEL)
        else:
            for key in primary_sub_feature_keys: state['features'][key] = NULL_SENTINEL
            
    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        for key in all_feature_keys:
            if key not in state['features']:
                state['features'][key] = NULL_SENTINEL
                
    return state