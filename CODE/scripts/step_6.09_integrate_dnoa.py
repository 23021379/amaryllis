import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import os
from sklearn.neighbors import BallTree

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')


# Input for this directive
DNOA_LV_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_DATA_INPUT = '[REDACTED_BY_SCRIPT]'
# Required for linking to primary substations
GRANULAR_LCT_INPUT =  r"[REDACTED_BY_SCRIPT]' Secondary Sites.geojson"

# Output of this directive
OUTPUT_ARTIFACT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
K_NEIGHBORS = 20
TEMPORAL_WINDOW_YEARS = 3


def load_and_prepare_dnoa_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Loads, re-projects, and sanitizes the DNOA LV constraint data.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_dnoa = gpd.read_file(filepath)

    # MANDATE: Unconditional CRS Unification to mitigate Pattern 1.
    gdf_dnoa = gdf_dnoa.to_crs("EPSG:27700")
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE: Proactive Schema Normalization and cleaning.
    gdf_dnoa.columns = [col.lower().strip().replace(' ', '_') for col in gdf_dnoa.columns]
    
    # MANDATE: Paranoid Type Casting.
    numeric_cols = ['constraint_year', '[REDACTED_BY_SCRIPT]']
    for col in numeric_cols:
        gdf_dnoa[col] = pd.to_numeric(gdf_dnoa[col], errors='coerce')
    
    initial_rows = len(gdf_dnoa)
    gdf_dnoa.dropna(subset=numeric_cols, inplace=True)
    if len(gdf_dnoa) < initial_rows:
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    # MANDATE: One-hot encode categorical results for feature generation.
    gdf_dnoa['dnoa_result'] = gdf_dnoa['dnoa_result'].str.strip()
    dnoa_dummies = pd.get_dummies(gdf_dnoa['dnoa_result'], prefix='dnoa_result_is', dtype=int)
    gdf_dnoa = pd.concat([gdf_dnoa, dnoa_dummies], axis=1)

    return gdf_dnoa

def calculate_knn_features(gdf_solar: gpd.GeoDataFrame, gdf_dnoa: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculates k-NN based features for each solar app, ensuring temporal correctness.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Prepare DNOA data for fast spatial queries
    dnoa_coords = np.array(list(zip(gdf_dnoa.geometry.x, gdf_dnoa.geometry.y)))
    dnoa_tree = BallTree(dnoa_coords, metric='euclidean')

    results = []
    for index, solar_app in gdf_solar.iterrows():
        # MANDATE: The Temporal Guard. Filter DNOA data for each application's point-in-time.
        submission_year = solar_app['submission_year']
        
        dnoa_temporally_relevant = gdf_dnoa[
            (gdf_dnoa['constraint_year'] >= submission_year) &
            (gdf_dnoa['constraint_year'] <= submission_year + TEMPORAL_WINDOW_YEARS)
        ].copy()

        if dnoa_temporally_relevant.empty:
            results.append({'solar_farm_id': solar_app['solar_farm_id']})
            continue

        # MANDATE: The Spatial Guard. Perform k-NN search on the *temporally filtered* subset.
        app_coords = [[solar_app.geometry.x, solar_app.geometry.y]]
        distances, indices = dnoa_tree.query(app_coords, k=min(K_NEIGHBORS, len(dnoa_coords)))
        
        # We queried the full tree, now we need to filter these results to only include temporally relevant ones
        relevant_indices = [idx for idx in indices[0] if gdf_dnoa.index[idx] in dnoa_temporally_relevant.index]
        
        if not relevant_indices:
            results.append({'solar_farm_id': solar_app['solar_farm_id']})
            continue

        knn_cohort = gdf_dnoa.iloc[relevant_indices].copy()
        # Calculate distances for the actual cohort
        cohort_coords = np.array(list(zip(knn_cohort.geometry.x, knn_cohort.geometry.y)))
        distances_to_cohort = np.linalg.norm(cohort_coords - app_coords, axis=1)
        knn_cohort['distance_m'] = distances_to_cohort

        # Feature Synthesis
        nearest = knn_cohort.iloc[0]
        res = {
            'solar_farm_id': solar_app['solar_farm_id'],
            '[REDACTED_BY_SCRIPT]': nearest['distance_m'],
            '[REDACTED_BY_SCRIPT]': nearest['constraint_year'] - submission_year,
            '[REDACTED_BY_SCRIPT]': len(knn_cohort),
            'dnoa_avg_dist_knn_m': knn_cohort['distance_m'].mean(),
            'dnoa_avg_deferred_kva_knn': knn_cohort['[REDACTED_BY_SCRIPT]'].mean(),
            '[REDACTED_BY_SCRIPT]': knn_cohort['constraint_year'].min(),
            '[REDACTED_BY_SCRIPT]': knn_cohort['[REDACTED_BY_SCRIPT]'].mean() if '[REDACTED_BY_SCRIPT]' in knn_cohort else 0
        }
        results.append(res)
        
    logging.info("[REDACTED_BY_SCRIPT]")
    return pd.DataFrame(results)


def aggregate_dnoa_to_primary(gdf_dnoa: gpd.GeoDataFrame, granular_lct_path: str) -> pd.DataFrame:
    """
    Links DNOA transformers to primary substations and aggregates constraint data.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    # Load primary substation locations from our trusted granular LCT source
    gdf_granular_lct = gpd.read_file(granular_lct_path)
    gdf_granular_lct = gdf_granular_lct.to_crs("EPSG:27700")
    gdf_primary_subs = gdf_granular_lct[['[REDACTED_BY_SCRIPT]', 'geometry']].drop_duplicates(subset='[REDACTED_BY_SCRIPT]').rename(columns={'[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]'})

    # Spatially join to find the nearest primary for each DNOA transformer
    gdf_dnoa_linked = gpd.sjoin_nearest(gdf_dnoa, gdf_primary_subs, how='left')

    logging.info("[REDACTED_BY_SCRIPT]")
    primary_summary = gdf_dnoa_linked.groupby('[REDACTED_BY_SCRIPT]').agg(
        primary_sub_downstream_constrained_tx_count=('geometry', 'count'),
        primary_sub_downstream_total_deferred_kva=('[REDACTED_BY_SCRIPT]', 'sum'),
        primary_sub_downstream_earliest_constraint_year=('constraint_year', 'min')
    ).reset_index()
    
    return primary_summary


def main():
    """
    Main function to execute the DNOA LV feature enrichment pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Load Data ---
    gdf_dnoa_l1 = load_and_prepare_dnoa_data(DNOA_LV_INPUT)

    df_solar_l14 = pd.read_csv(SOLAR_DATA_INPUT)
    gdf_solar = gpd.GeoDataFrame(
        df_solar_l14, geometry=gpd.points_from_xy(df_solar_l14.easting, df_solar_l14.northing), crs="EPSG:27700"
    )
    # Create a stable ID for the loop and final join
    gdf_solar['solar_farm_id'] = gdf_solar.index
    
    # --- Execute Phases ---
    # Phase 2: Point-in-Time k-NN Features
    knn_features = calculate_knn_features(gdf_solar, gdf_dnoa_l1)
    
    # Phase 3: Primary Substation Aggregation
    primary_sub_features = aggregate_dnoa_to_primary(gdf_dnoa_l1, GRANULAR_LCT_INPUT)
    
    # --- Final Integration ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_enriched = df_solar_l14.merge(knn_features, left_index=True, right_on='solar_farm_id', how='left')
    df_enriched = df_enriched.merge(primary_sub_features, on='[REDACTED_BY_SCRIPT]', how='left')
    
    # Clean up and fill NaNs
    df_enriched.drop(columns=['solar_farm_id'], inplace=True, errors='ignore')
    new_cols = [col for col in df_enriched.columns if 'dnoa_' in col or '[REDACTED_BY_SCRIPT]' in col]
    df_enriched[new_cols] = df_enriched[new_cols].fillna(0)
    
    # --- Persist Final Artifact ---
    output_path = OUTPUT_ARTIFACT
    df_enriched.to_csv(output_path, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()