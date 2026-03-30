import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
import logging
import sys

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input/Output Artifacts
L3_DATA_PATH = '[REDACTED_BY_SCRIPT]'
L4_DATA_PATH = '[REDACTED_BY_SCRIPT]'

# Architectural Hyperparameters
CUTOFF_YEAR = 2012
SEARCH_RADIUS_METERS = 10000  # 10km
MAX_DISTANCE_KM = 999 # Sentinel value for non-existent nearest sites

def main():
    """
    Executes Directive 006: Temporal Cohort Stratification and Legacy Feature Synthesis.
    This script ingests the L3 dataset, partitions it into modern and legacy cohorts,
    and uses the legacy data to engineer historical and geospatial features for the
    modern cohort, producing a temporally-sound L4 training dataset.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    try:
        df_l3 = pd.read_csv(L3_DATA_PATH)
        # Basic schema validation
        required_cols = ['submission_year', 'easting', 'northing', 'permission_granted', '[REDACTED_BY_SCRIPT]']
        if not all(col in df_l3.columns for col in required_cols):
            raise ValueError(f"[REDACTED_BY_SCRIPT]")
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- MANDATE 6.1: Temporal Cohort Stratification ---
    modern_cohort = df_l3[df_l3['submission_year'] >= CUTOFF_YEAR].copy()
    legacy_cohort = df_l3[df_l3['submission_year'] < CUTOFF_YEAR].copy()

    if modern_cohort.empty:
        logging.error("[REDACTED_BY_SCRIPT]")
        sys.exit(1)
    if legacy_cohort.empty:
        logging.warning("[REDACTED_BY_SCRIPT]")

    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- MANDATE 6.2: Geospatial Integrity Enforcement ---
    gdf_modern = gpd.GeoDataFrame(
        modern_cohort, geometry=gpd.points_from_xy(modern_cohort.easting, modern_cohort.northing), crs="EPSG:27700"
    )
    gdf_legacy = gpd.GeoDataFrame(
        legacy_cohort, geometry=gpd.points_from_xy(legacy_cohort.easting, legacy_cohort.northing), crs="EPSG:27700"
    )
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- MANDATE 6.3: Legacy Aggregation by LPA ---
    # --- MANDATE 6.3: Legacy Aggregation by LPA ---
    logging.info("[REDACTED_BY_SCRIPT]")
    if 'planning_authority' in legacy_cohort.columns and not legacy_cohort.empty:
        # Filter for granted applications to correctly calculate average duration
        granted_legacy = legacy_cohort[legacy_cohort['permission_granted'] == 1]
        
        lpa_legacy_duration = granted_legacy.groupby('planning_authority').agg(
            lpa_legacy_avg_duration=('[REDACTED_BY_SCRIPT]', 'mean')
        ).reset_index()

        lpa_legacy_profile = legacy_cohort.groupby('planning_authority').agg(
            lpa_legacy_approval_rate=('permission_granted', 'mean'),
            lpa_legacy_application_count=('permission_granted', 'size')
        ).reset_index()
        
        # Merge duration separately to handle the filtering correctly
        lpa_legacy_profile = lpa_legacy_profile.merge(lpa_legacy_duration, on='planning_authority', how='left')

        gdf_modern = gdf_modern.merge(lpa_legacy_profile, on='planning_authority', how='left')
        logging.info("[REDACTED_BY_SCRIPT]")
    else:
        logging.critical("CRITICAL WARNING: 'planning_authority'[REDACTED_BY_SCRIPT]")
        gdf_modern['lpa_legacy_approval_rate'] = np.nan
        gdf_modern['[REDACTED_BY_SCRIPT]'] = np.nan
        gdf_modern['[REDACTED_BY_SCRIPT]'] = np.nan


    # --- MANDATE 6.4: Geospatial Proximity Feature Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    if not legacy_cohort.empty:
        # Prepare coordinate arrays for spatial indexing
        modern_points = np.array(list(zip(gdf_modern.geometry.x, gdf_modern.geometry.y)))
        
        # Build spatial trees for efficient querying
        legacy_success = gdf_legacy[gdf_legacy['permission_granted'] == 1]
        legacy_refusal = gdf_legacy[gdf_legacy['permission_granted'] == 0]

        tree_all = cKDTree(np.array(list(zip(gdf_legacy.geometry.x, gdf_legacy.geometry.y))))
        tree_success = cKDTree(np.array(list(zip(legacy_success.geometry.x, legacy_success.geometry.y)))) if not legacy_success.empty else None
        tree_refusal = cKDTree(np.array(list(zip(legacy_refusal.geometry.x, legacy_refusal.geometry.y)))) if not legacy_refusal.empty else None
        logging.info("[REDACTED_BY_SCRIPT]")

        # 1. Calculate count and approval rate within radius
        nearby_indices_list = tree_all.query_ball_point(modern_points, r=SEARCH_RADIUS_METERS)
        
        nearby_counts = [len(indices) for indices in nearby_indices_list]
        nearby_approval_rates = [
            gdf_legacy.iloc[indices]['permission_granted'].mean() if len(indices) > 0 else -1
            for indices in nearby_indices_list
        ]

        # 2. Calculate distances to nearest sites
        dist_nearest_all, _ = tree_all.query(modern_points, k=1)
        
        if tree_success:
            dist_nearest_success, _ = tree_success.query(modern_points, k=1)
        else:
            dist_nearest_success = np.full(len(modern_points), np.inf)

        if tree_refusal:
            dist_nearest_refusal, _ = tree_refusal.query(modern_points, k=1)
        else:
            dist_nearest_refusal = np.full(len(modern_points), np.inf)

        # Assemble features into a DataFrame
        df_proximity = pd.DataFrame({
            'nearby_legacy_count': nearby_counts,
            '[REDACTED_BY_SCRIPT]': nearby_approval_rates,
            '[REDACTED_BY_SCRIPT]': dist_nearest_all / 1000,
            '[REDACTED_BY_SCRIPT]': dist_nearest_success / 1000,
            '[REDACTED_BY_SCRIPT]': dist_nearest_refusal / 1000
        })
        
        # Replace infinite distances with the sentinel value
        df_proximity.replace([np.inf, -np.inf], MAX_DISTANCE_KM, inplace=True)
        
        # Join features back to the main modern GeoDataFrame
        gdf_modern = gdf_modern.reset_index(drop=True).join(df_proximity)
        logging.info("[REDACTED_BY_SCRIPT]")
    else:
        gdf_modern['nearby_legacy_count'] = 0
        gdf_modern['[REDACTED_BY_SCRIPT]'] = -1
        gdf_modern['[REDACTED_BY_SCRIPT]'] = MAX_DISTANCE_KM
        gdf_modern['[REDACTED_BY_SCRIPT]'] = MAX_DISTANCE_KM
        gdf_modern['[REDACTED_BY_SCRIPT]'] = MAX_DISTANCE_KM


    # --- MANDATE 6.5: Final Integration and Schema Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Drop intermediate geospatial column
    df_l4 = pd.DataFrame(gdf_modern.drop(columns='geometry'))
    
    # Final imputation to handle NaNs from left merges or calculations
    df_l4.fillna(-1, inplace=True)
    
    # Final check for non-numeric data before saving
    non_numeric_cols = df_l4.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        df_l4 = df_l4.drop(columns=non_numeric_cols)

    df_l4.to_csv(L4_DATA_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()