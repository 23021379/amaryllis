import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from shapely.geometry import Point
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
# Input files
L7_SOLAR_APPLICATIONS_PATH = r"[REDACTED_BY_SCRIPT]"
ECR_SUB_1MW_PATH = r"[REDACTED_BY_SCRIPT]"
ECR_OVER_1MW_PATH = r"[REDACTED_BY_SCRIPT]"

# Output file
L9_OUTPUT_PATH = r"[REDACTED_BY_SCRIPT]"

# Geospatial constants
CRS_PROJECT = 'EPSG:27700' # British National Grid

# Analysis parameters
RADII_METERS = [2000, 5000, 10000]

def clean_col_names(df):
    """
    Cleans and standardizes DataFrame column names.
    - Converts to lowercase
    - Replaces non-alphanumeric characters with underscores
    - Strips leading/trailing underscores
    """
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.lower()
        new_col = ''.join(c if c.isalnum() else '_' for c in new_col)
        new_col = new_col.strip('_')
        new_col = new_col.replace('__', '_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def load_and_unify_ecr_data():
    """
    Implements Mandate 11.1: Ingests, harmonizes, and unifies the sub-1MW and
    over-1MW ECR GeoJSON datasets into a single master GeoDataFrame.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Ingest sub-1MW data
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_sub1mw = gpd.read_file(ECR_SUB_1MW_PATH)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # Ingest over-1MW data
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_over1mw = gpd.read_file(ECR_OVER_1MW_PATH)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # Unify CRS immediately - The primary gate for geospatial integrity.
    if gdf_sub1mw.crs != CRS_PROJECT:
        gdf_sub1mw = gdf_sub1mw.to_crs(CRS_PROJECT)
    if gdf_over1mw.crs != CRS_PROJECT:
        gdf_over1mw = gdf_over1mw.to_crs(CRS_PROJECT)
    
    # Normalize schemas
    gdf_sub1mw = clean_col_names(gdf_sub1mw)
    gdf_over1mw = clean_col_names(gdf_over1mw)

    # Source Tagging (Critical Step)
    gdf_sub1mw['capacity_scale'] = 'sub_1mw'
    gdf_over1mw['capacity_scale'] = 'over_1mw'

    # Concatenate into a single GeoDataFrame
    ecr_master_gdf = pd.concat([gdf_sub1mw, gdf_over1mw], ignore_index=True)
    ecr_master_gdf = gpd.GeoDataFrame(ecr_master_gdf, geometry='geometry', crs=CRS_PROJECT)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return ecr_master_gdf

def distill_and_filter_ecr(gdf):
    """
    Implements Mandate 11.2: Distills the master GDF down to essential features,
    aggregates capacity, and applies status/temporal filters.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    # Coerce capacity and storage columns to numeric
    capacity_cols = [c for c in gdf.columns if 'capacity_mw' in c or 'storage_mwh' in c]
    for col in capacity_cols:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

    # Aggregate total capacity
    gdf['[REDACTED_BY_SCRIPT]'] = gdf.filter(like='capacity_mw').sum(axis=1)
    gdf['total_storage_mwh'] = gdf.filter(like='storage_mwh').sum(axis=1)

    # Feature Selection
    essential_cols = [
        'geometry', 'capacity_scale', 'date_accepted', 'connection_status',
        'energy_source_1', '[REDACTED_BY_SCRIPT]', 'total_storage_mwh'
    ]
    gdf = gdf[essential_cols]

    # Status & Temporal Filtering
    gdf = gdf[gdf['connection_status'].isin(['Connected', 'Accepted to Connect'])].copy()
    gdf['date_accepted'] = pd.to_datetime(gdf['date_accepted'], errors='coerce')
    gdf.dropna(subset=['date_accepted'], inplace=True)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf

def calculate_stratified_density_features(solar_site, ecr_sindex, gdf_ecr_final):
    """
    For a single solar site, queries nearby ECR precedents, stratifies them,
    and calculates density features for each stratum and radius.
    """
    features = {}
    site_geom = solar_site.geometry
    submission_date = solar_site.submission_date

    # The temporal filter is now applied *after* the spatial query.

    for r in RADII_METERS:
        r_km = r // 1000
        # Step 1: Broad spatial query on the full dataset using the index
        possible_matches_index = list(ecr_sindex.intersection(site_geom.buffer(r).bounds))
        possible_matches = gdf_ecr_final.iloc[possible_matches_index]
        
        # Step 2: Precise spatial query on the smaller candidate set
        precise_spatial_matches = possible_matches[possible_matches.intersects(site_geom.buffer(r))]

        # Step 3: NOW apply the temporal filter to the spatially-valid precedents
        actual_matches = precise_spatial_matches[precise_spatial_matches['date_accepted'] < submission_date]
        
        # Stratify the results
        precedents_sub1mw = actual_matches[actual_matches['capacity_scale'] == 'sub_1mw']
        precedents_over1mw = actual_matches[actual_matches['capacity_scale'] == 'over_1mw']
        
        # --- Calculate features for each stratum ---
        strata = {
            'sub1mw': precedents_sub1mw,
            'over1mw': precedents_over1mw,
        }
        
        for name, df_stratum in strata.items():
            count = len(df_stratum)
            capacity_mw = df_stratum['[REDACTED_BY_SCRIPT]'].sum()
            storage_mwh = df_stratum['total_storage_mwh'].sum()
            solar_count = df_stratum[df_stratum['energy_source_1'] == 'Solar Photovoltaics'].shape[0]
            
            features[f'[REDACTED_BY_SCRIPT]'] = count
            features[f'[REDACTED_BY_SCRIPT]'] = capacity_mw
            features[f'[REDACTED_BY_SCRIPT]'] = storage_mwh
            features[f'[REDACTED_BY_SCRIPT]'] = solar_count / count if count > 0 else 0

        # --- Calculate combined total features ---
        features[f'[REDACTED_BY_SCRIPT]'] = len(actual_matches)
        features[f'[REDACTED_BY_SCRIPT]'] = actual_matches['[REDACTED_BY_SCRIPT]'].sum()

    return pd.Series(features)

def main():
    """
    Main function to execute the unified DER feature synthesis pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Mandate 11.1 & 11.2: Unify and distill ECR data in a GDF-native pipeline
    ecr_master_gdf = load_and_unify_ecr_data()
    gdf_ecr_final = distill_and_filter_ecr(ecr_master_gdf)

    # Load L7 solar application data
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df_solar = pd.read_csv(L7_SOLAR_APPLICATIONS_PATH)
    
    # Reconstruct the submission date from its components to restore temporal integrity
    logging.info("[REDACTED_BY_SCRIPT]")
    # Use arctan2 to robustly reverse the cyclical encoding of the month
    angle = np.mod(np.arctan2(df_solar['submission_month_sin'], df_solar['submission_month_cos']), 2*np.pi)
    # Convert angle back to month (1-12)
    df_solar['submission_month'] = np.round((angle * 12 / (2 * np.pi)) + 1).astype(int)
    # Handle the edge case where rounding 12.x results in 13
    df_solar['submission_month'] = df_solar['submission_month'].replace(13, 1)

    # Combine components into a single datetime column
    date_cols = {'year': df_solar['submission_year'], 'month': df_solar['submission_month'], 'day': df_solar['submission_day']}
    df_solar['submission_date'] = pd.to_datetime(date_cols, errors='coerce')

    geometry = [Point(xy) for xy in zip(df_solar['easting'], df_solar['northing'])]
    gdf_solar = gpd.GeoDataFrame(df_solar, geometry=geometry, crs=CRS_PROJECT)
    gdf_solar.dropna(subset=['submission_date'], inplace=True)

    # Mandate 11.3: Build spatial index and apply the density engine
    logging.info("[REDACTED_BY_SCRIPT]")
    ecr_sindex = gdf_ecr_final.sindex

    logging.info(f"[REDACTED_BY_SCRIPT]")
    tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
    der_features = gdf_solar.progress_apply(
        calculate_stratified_density_features, 
        axis=1, 
        ecr_sindex=ecr_sindex, 
        gdf_ecr_final=gdf_ecr_final
    )

    # Final Integration Protocol
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_solar_l9 = gdf_solar.join(der_features)
    
    # Drop geometry for final CSV artifact
    df_solar_l9 = pd.DataFrame(gdf_solar_l9.drop(columns='geometry'))
    df_solar_l9.to_csv(L9_OUTPUT_PATH, index=False)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()