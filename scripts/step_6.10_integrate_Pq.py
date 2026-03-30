import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')


# Input for this directive
PQ_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
K_NEIGHBORS = 5

def load_and_aggregate_pq_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Loads, sanitizes, and aggregates the hostile long-form PQ data into a stable L1 artifact.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_pq = gpd.read_file(filepath)

    # MANDATE: Unconditional CRS Unification.
    gdf_pq = gdf_pq.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Schema Normalization and Cleaning.
    gdf_pq.columns = [col.lower().strip() for col in gdf_pq.columns]
    gdf_pq['highest'] = pd.to_numeric(gdf_pq['highest'], errors='coerce')
    gdf_pq.dropna(subset=['highest', 'sitefunctionallocation'], inplace=True)

    # Create a numeric harmonic order, handling the 'THD' special case.
    gdf_pq['harmonic_order'] = gdf_pq['harmonic'].replace('THD', '-1').str.replace('H', '').astype(int)

    # MANDATE: Intelligent Aggregation (The Anti-Pivot).
    def aggregate_site(group):
        # Use a temporary pivot within the group to easily access specific harmonics.
        pivot = group.pivot_table(index='sitefunctionallocation', columns='harmonic_order', values='highest')
        
        # Extract key harmonic features, filling with 0 if not present.
        res = {
            'pq_thd_highest': pivot.get(-1, pd.Series(0)).iloc[0],
            'pq_h3_highest': pivot.get(3, pd.Series(0)).iloc[0],
            'pq_h5_highest': pivot.get(5, pd.Series(0)).iloc[0],
            'pq_h7_highest': pivot.get(7, pd.Series(0)).iloc[0],
            'pq_h11_highest': pivot.get(11, pd.Series(0)).iloc[0],
            'pq_h13_highest': pivot.get(13, pd.Series(0)).iloc[0],
        }
        
        # Calculate aggregate characterization features.
        odd_harmonics = group[group['harmonic_order'] > 0 & (group['harmonic_order'] % 2 != 0)]
        res['[REDACTED_BY_SCRIPT]'] = odd_harmonics['highest'].mean() if not odd_harmonics.empty else 0
        res['[REDACTED_BY_SCRIPT]'] = group['highest'].max()
        res['[REDACTED_BY_SCRIPT]'] = group.loc[group['highest'].idxmax(), 'harmonic_order']
        
        return pd.Series(res)

    logging.info("[REDACTED_BY_SCRIPT]")
    aggregated_data = gdf_pq.groupby('sitefunctionallocation').apply(aggregate_site)
    
    # Re-join with geometry to create the final GeoDataFrame L1 artifact.
    sites_geometry = gdf_pq[['sitefunctionallocation', 'geometry']].drop_duplicates(subset='sitefunctionallocation').set_index('sitefunctionallocation')
    gdf_pq_l1 = sites_geometry.join(aggregated_data).reset_index()
    
    logging.info("[REDACTED_BY_SCRIPT]")
    return gdf_pq_l1


def calculate_spatial_pq_features(gdf_solar: gpd.GeoDataFrame, gdf_pq_l1: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Generates spatially interpolated PQ features for each solar application using IDW k-NN.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    if gdf_pq_l1.empty:
        logging.warning("[REDACTED_BY_SCRIPT]")
        return pd.DataFrame()

    # Prepare PQ data for fast spatial queries
    pq_coords = np.array(list(zip(gdf_pq_l1.geometry.x, gdf_pq_l1.geometry.y)))
    pq_tree = BallTree(pq_coords, metric='euclidean')
    
    # Query for all solar apps at once for performance
    app_coords = np.array(list(zip(gdf_solar.geometry.x, gdf_solar.geometry.y)))
    distances, indices = pq_tree.query(app_coords, k=min(K_NEIGHBORS, len(pq_coords)))

    results = []
    for i in range(len(gdf_solar)):
        cohort_indices = indices[i]
        cohort_distances = distances[i]
        
        # Handle cases where distance is zero to avoid division by zero in weights
        cohort_distances[cohort_distances == 0] = 1e-6
        
        cohort = gdf_pq_l1.iloc[cohort_indices]
        
        # MANDATE: Inverse Distance Weighting
        weights = 1 / (cohort_distances ** 2)
        
        res = {
            'solar_farm_id': gdf_solar.iloc[i]['solar_farm_id'],
            'pq_idw_thd_knn5': np.average(cohort['pq_thd_highest'], weights=weights),
            '[REDACTED_BY_SCRIPT]': np.average(cohort['pq_h5_highest'], weights=weights),
            '[REDACTED_BY_SCRIPT]': np.average(cohort['[REDACTED_BY_SCRIPT]'], weights=weights),
            '[REDACTED_BY_SCRIPT]': cohort_distances[0] / 1000,
            'pq_max_thd_in_knn5': cohort['pq_thd_highest'].max(),
            'pq_std_thd_in_knn5': cohort['pq_thd_highest'].std()
        }
        results.append(res)
        
    logging.info("[REDACTED_BY_SCRIPT]")
    return pd.DataFrame(results)


def main():
    """
    Main function to execute the Power Quality feature enrichment pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Ingest and Aggregate ---
    gdf_pq_l1 = load_and_aggregate_pq_data(PQ_INPUT)
    
    # --- Load Solar Data ---
    df_solar_l15 = pd.read_csv(SOLAR_DATA_INPUT)
    gdf_solar = gpd.GeoDataFrame(
        df_solar_l15, geometry=gpd.points_from_xy(df_solar_l15.easting, df_solar_l15.northing), crs="EPSG:27700"
    )
    # Create a stable ID for the join
    gdf_solar['solar_farm_id'] = gdf_solar.index

    # --- Phase 2: Spatial Feature Generation ---
    pq_features = calculate_spatial_pq_features(gdf_solar, gdf_pq_l1)
    
    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_enriched = df_solar_l15.merge(pq_features, left_index=True, right_on='solar_farm_id', how='left')
    
    # Clean up and fill NaNs
    df_enriched.drop(columns=['solar_farm_id'], inplace=True, errors='ignore')
    new_cols = [col for col in df_enriched.columns if 'pq_' in col]
    df_enriched[new_cols] = df_enriched[new_cols].fillna(0)
    
    # --- Persist Final Artifact ---
    output_path = OUTPUT_ARTIFACT
    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()