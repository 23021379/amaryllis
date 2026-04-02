import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from scipy.spatial import cKDTree
from shapely.geometry import Point
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
# Input files
L9_SOLAR_APPLICATIONS_PATH = '[REDACTED_BY_SCRIPT]'
TRANSFORMERS_PATH = r"[REDACTED_BY_SCRIPT]"
DFES_LOCATIONS_PATH = r"[REDACTED_BY_SCRIPT]"

# Output file
L10_OUTPUT_PATH = '[REDACTED_BY_SCRIPT]'

# Geospatial constants
CRS_PROJECT = 'EPSG:27700' # British National Grid

# Analysis parameters
K_NEIGHBORS = 5
RADIUS_METERS = 10000

def clean_col_names(df):
    """[REDACTED_BY_SCRIPT]"""
    cols = [col.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for col in df.columns]
    df.columns = cols
    return df

def find_column_by_pattern(df, pattern):
    """[REDACTED_BY_SCRIPT]"""
    for col in df.columns:
        if pattern in col:
            return col
    raise KeyError(f"Pattern '{pattern}'[REDACTED_BY_SCRIPT]")

def prepare_substation_capacity_data():
    """
    Implements Mandates 12.1 & 12.2: Ingests, cleans, and aggregates native
    transformer GeoJSON data to create a substation-level capacity profile.
    The Geospatial Bridge is not required as the source is already geolocated.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    # Step 1: Ingestion, CRS Unification, and Normalization
    gdf_transformers = gpd.read_file(TRANSFORMERS_PATH)
    if gdf_transformers.crs != CRS_PROJECT:
        gdf_transformers = gdf_transformers.to_crs(CRS_PROJECT)
    gdf_transformers = clean_col_names(gdf_transformers)
    initial_count = len(gdf_transformers)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # Step 2: Paranoid Type Casting
    gdf_transformers['onanrating_kva'] = pd.to_numeric(gdf_transformers['onanrating_kva'], errors='coerce')
    gdf_transformers.dropna(subset=['onanrating_kva'], inplace=True)
    valid_kva_count = len(gdf_transformers)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # Mandate 12.2: Aggregation by Summation
    # Identify the correct grouping key for a substation
    substation_key = find_column_by_pattern(gdf_transformers, 'sitefunctional')
    
    agg_data = gdf_transformers.groupby(substation_key).agg(
        total_onan_rating_kva=('onanrating_kva', 'sum'),
        geometry=('geometry', 'first')
    ).reset_index()

    # Mandate 12.2: Geospatial State Restoration
    gdf_substation_capacity = gpd.GeoDataFrame(agg_data, geometry='geometry', crs=CRS_PROJECT)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return gdf_substation_capacity

def calculate_transformer_features(solar_site_coords, tree, substation_data):
    """
    For a single solar site, calculates KNN and radius-based transformer capacity features.
    """
    features = {}
    
    # KNN Query (k=5)
    distances, indices = tree.query(solar_site_coords, k=K_NEIGHBORS)
    
    # Handle cases where fewer than k neighbors are found
    valid_indices = indices[np.isfinite(distances)]
    
    if len(valid_indices) > 0:
        nearest_substations = substation_data.iloc[valid_indices]
        # KNN Feature Synthesis
        features['[REDACTED_BY_SCRIPT]'] = nearest_substations.iloc[0]['total_onan_rating_kva']
        features['avg_total_kva_5nn'] = nearest_substations['total_onan_rating_kva'].mean()
    else:
        features['[REDACTED_BY_SCRIPT]'] = 0
        features['avg_total_kva_5nn'] = 0

    # Radius Query (10km)
    radius_indices = tree.query_ball_point(solar_site_coords, r=RADIUS_METERS)
    
    if len(radius_indices) > 0:
        substations_in_radius = substation_data.iloc[radius_indices]
        # Radius Feature Synthesis
        features['[REDACTED_BY_SCRIPT]'] = substations_in_radius['total_onan_rating_kva'].sum()
    else:
        features['[REDACTED_BY_SCRIPT]'] = 0
        
    return pd.Series(features)


# --- Main Script ---
def main():
    """
    Main function to execute the transformer capacity feature synthesis pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Mandates 12.1 & 12.2
    gdf_substation_capacity = prepare_substation_capacity_data()

    # Load L9 Data
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df_solar = pd.read_csv(L9_SOLAR_APPLICATIONS_PATH)
    # Create a numpy array of coordinates for efficient processing
    solar_coords = df_solar[['easting', 'northing']].values

    # Mandate 12.3: Build Spatial Index
    logging.info("[REDACTED_BY_SCRIPT]")
    substation_coords = np.array([geom.coords[0] for geom in gdf_substation_capacity.geometry])
    tree = cKDTree(substation_coords)

    # Mandate 12.3: Apply Feature Synthesis Engine
    logging.info(f"[REDACTED_BY_SCRIPT]")
    # Using a direct apply on the numpy array for performance
    results = [calculate_transformer_features(coords, tree, gdf_substation_capacity) for coords in tqdm(solar_coords, desc="[REDACTED_BY_SCRIPT]")]
    transformer_features = pd.DataFrame(results)

    # Final Integration Protocol
    logging.info("[REDACTED_BY_SCRIPT]")
    df_solar_l10 = pd.concat([df_solar, transformer_features], axis=1)
    
    # Final Cleanup
    new_cols = ['[REDACTED_BY_SCRIPT]', 'avg_total_kva_5nn', '[REDACTED_BY_SCRIPT]']
    df_solar_l10[new_cols] = df_solar_l10[new_cols].fillna(-1.0)

    df_solar_l10.to_csv(L10_OUTPUT_PATH, index=False)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()