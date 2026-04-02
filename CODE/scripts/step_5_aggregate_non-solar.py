import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm
import logging
import sys

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input/Output Artifacts
L4_DATA_PATH = '[REDACTED_BY_SCRIPT]'
L5_DATA_PATH = '[REDACTED_BY_SCRIPT]'

# Architectural Hyperparameters
K_NEIGHBORS = 10
SOLAR_TECH_CODE = 21  # As per categorical mapping for 'Solar Photovoltaics'
NULL_SENTINEL = -1
MAX_DISTANCE_KM = 999 # Sentinel for cases where a nearest type doesn't exist among neighbors

def main():
    """
    Executes the revised Directive 007. Ingests the L4 dataset, engineers a proxy
    decision date, and performs a temporally-constrained KNN feature synthesis for
    solar applications to produce the final L5 training set.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    try:
        df_l4 = pd.read_csv(L4_DATA_PATH)
        # MANDATORY SCHEMA VALIDATION against L4 artifact
        required_cols = [
            'submission_year', 'submission_month_sin', 'submission_month_cos',
            '[REDACTED_BY_SCRIPT]', 'easting', 'northing', 'technology_type',
            'permission_granted'
        ]
        if not all(col in df_l4.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_l4.columns]
            raise ValueError(f"[REDACTED_BY_SCRIPT]")
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- MANDATE 7.2 (Revised): Temporal Integrity Engineering ---
    # Step 1: Re-engineer Submission Date from cyclical features
    logging.info("Re-engineering 'planning_application_submitted'[REDACTED_BY_SCRIPT]")
    # arctan2(y, x) -> arctan2(sin, cos)
    month_raw = np.arctan2(df_l4['submission_month_sin'], df_l4['submission_month_cos']) * (12 / (2 * np.pi))
    # Shift from (-6, 6) to (1, 12) range
    df_l4['submission_month'] = np.round(month_raw % 12) + 1
    
    df_l4['[REDACTED_BY_SCRIPT]'] = pd.to_datetime(
        df_l4['submission_year'].astype(int).astype(str) + '-' + df_l4['submission_month'].astype(int).astype(str) + '-01',
        errors='coerce'
    )

    # Step 2: Engineer Estimated Decision Date
    logging.info("Engineering 'estimated_decision_date'[REDACTED_BY_SCRIPT]")
    # A valid precedent MUST have a valid planning duration.
    source_df = df_l4[df_l4['[REDACTED_BY_SCRIPT]'] >= 0].copy()
    source_df['estimated_decision_date'] = source_df['[REDACTED_BY_SCRIPT]'] + pd.to_timedelta(source_df['[REDACTED_BY_SCRIPT]'], unit='D')

    # --- MANDATE 7.1 (Revised): Isolate Populations from L4 Data ---
    source_gdf = gpd.GeoDataFrame(
        source_df, geometry=gpd.points_from_xy(source_df.easting, source_df.northing), crs="EPSG:27700"
    )
    target_df = df_l4.copy()
    target_gdf = gpd.GeoDataFrame(
        target_df, geometry=gpd.points_from_xy(target_df.easting, target_df.northing), crs="EPSG:27700"
    )
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- MANDATE 7.3: Architect a Temporally-Aware Iterative KNN Engine ---
    results = []
    
    for index, target in tqdm(target_gdf.iterrows(), total=target_gdf.shape[0], desc="[REDACTED_BY_SCRIPT]"):
        submission_date = target['[REDACTED_BY_SCRIPT]']
        if pd.isna(submission_date):
            results.append({'original_index': index})
            continue
            
        # REVISED TEMPORAL GUARD
        source_candidates = source_gdf[source_gdf['estimated_decision_date'] < submission_date]
        
        if len(source_candidates) < K_NEIGHBORS:
            results.append({'original_index': index})
            continue
        
        tree = cKDTree(np.array(list(zip(source_candidates.geometry.x, source_candidates.geometry.y))))
        target_coord = [target.geometry.x, target.geometry.y]
        distances_m, indices = tree.query(target_coord, k=K_NEIGHBORS)
        neighbors = source_candidates.iloc[indices]
        
        # --- MANDATE 7.4: Synthesize Advanced Precedent Features ---
        features = {'original_index': index}
        distances_km = distances_m / 1000.0
        
        features['knn_avg_distance_km'] = np.mean(distances_km)
        features['knn_std_distance_km'] = np.std(distances_km)
        features['knn_dist_to_nearest_km'] = np.min(distances_km)
        
        features['knn_approval_rate'] = neighbors['permission_granted'].mean()
        weights = 1.0 / (distances_km + 1e-6)
        features['[REDACTED_BY_SCRIPT]'] = np.sum(neighbors['permission_granted'] * weights) / np.sum(weights)
        
        is_solar_mask = neighbors['technology_type'] == SOLAR_TECH_CODE
        is_refusal_mask = neighbors['permission_granted'] == 0

        features['knn_count_solar'] = is_solar_mask.sum()
        
        solar_distances_km = distances_km[is_solar_mask]
        features['[REDACTED_BY_SCRIPT]'] = np.min(solar_distances_km) if len(solar_distances_km) > 0 else MAX_DISTANCE_KM

        refusal_distances_km = distances_km[is_refusal_mask]
        features['[REDACTED_BY_SCRIPT]'] = np.min(refusal_distances_km) if len(refusal_distances_km) > 0 else MAX_DISTANCE_KM

        results.append(features)

    # --- FINAL INTEGRATION ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_knn_features = pd.DataFrame(results).set_index('original_index')
    df_l5 = target_gdf.merge(df_knn_features, left_index=True, right_index=True, how='left')
    
    # Drop intermediate columns
    cols_to_drop = ['geometry', 'submission_month', '[REDACTED_BY_SCRIPT]']
    df_l5 = pd.DataFrame(df_l5.drop(columns=cols_to_drop))
    
    df_l5.fillna(NULL_SENTINEL, inplace=True) 

    df_l5.to_csv(L5_DATA_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()