import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import glob
from shapely.geometry import Point
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
# Input files
L11_SOLAR_APPLICATIONS_PATH = '[REDACTED_BY_SCRIPT]'

LTDS_SOURCE_FILES = {
    r"[REDACTED_BY_SCRIPT]": "2018-10-13"
    }

# Output file
L12_OUTPUT_PATH = '[REDACTED_BY_SCRIPT]'

# Geospatial constants
CRS_PROJECT = 'EPSG:27700' # British National Grid

# Analysis parameters
RADIUS_METERS = 10000
NULL_SENTINEL = -1.0


def load_ltds_data():
    """
    Loads all historical LTDS GeoJSON files, assigns their mandatory publication
    date, unifies CRS, and concatenates them into a single master artifact.
    This function correctly implements the Temporal Stamping Protocol.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    all_ltds_gdfs = []
    for path, pub_date in LTDS_SOURCE_FILES.items():
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf = gpd.read_file(path)
        
        # MANDATE: Assign Publication Date
        gdf['[REDACTED_BY_SCRIPT]'] = pd.to_datetime(pub_date)
        
        # DEFENSIVE RE-PROJECTION: The previous conditional check was insufficient and
        # allowed corrupted CRS metadata to pass through. We now enforce our knowledge
        # that the source coordinates are WGS84, overriding any file metadata, and
        # then transform to the project standard CRS. This is non-negotiable.
        gdf = gdf.set_crs('EPSG:4326', allow_override=True).to_crs(CRS_PROJECT)

        all_ltds_gdfs.append(gdf)

    if not all_ltds_gdfs:
        raise ValueError("[REDACTED_BY_SCRIPT]")

    # After concatenation, the CRS state can be lost. We must explicitly restore it.
    concatenated_df = pd.concat(all_ltds_gdfs, ignore_index=True)
    gdf_ltds_master = gpd.GeoDataFrame(concatenated_df, geometry='geometry', crs=CRS_PROJECT)
    
    # Normalize column names for consistency AFTER state restoration
    gdf_ltds_master.rename(columns=lambda x: x.lower(), inplace=True)

    # Paranoid Type Casting for temporal features using the correct normalized column name
    gdf_ltds_master['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(gdf_ltds_master['[REDACTED_BY_SCRIPT]'], errors='coerce')
    gdf_ltds_master.dropna(subset=['[REDACTED_BY_SCRIPT]'], inplace=True)
    gdf_ltds_master['[REDACTED_BY_SCRIPT]'] = gdf_ltds_master['[REDACTED_BY_SCRIPT]'].astype(int)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_ltds_master
def calculate_features_for_application(solar_app_row, gdf_ltds_master):
    """
    Calculates all LTDS features for a SINGLE solar application, respecting
    the temporal guard protocol.
    """
    features = {
        '[REDACTED_BY_SCRIPT]': 0.001,
        '[REDACTED_BY_SCRIPT]': 0.001,
        '[REDACTED_BY_SCRIPT]': 0.001,
        'ltds_count_in_10km': 0.001,
        '[REDACTED_BY_SCRIPT]': 0.001,
        '[REDACTED_BY_SCRIPT]': 0.001,
        '[REDACTED_BY_SCRIPT]': 0.001
    }
    
    # MANDATE: The Temporal Filter
    app_date = solar_app_row['submission_date']
    valid_ltds = gdf_ltds_master[gdf_ltds_master['[REDACTED_BY_SCRIPT]'] <= app_date]
    
    if valid_ltds.empty:
        # This is correct behavior if the application date precedes all known LTDS publications.
        logging.debug(f"[REDACTED_BY_SCRIPT]")
        return features # No LTDS data existed at this time.
        
    latest_pub_date = valid_ltds['[REDACTED_BY_SCRIPT]'].max()
    point_in_time_ltds = valid_ltds[valid_ltds['[REDACTED_BY_SCRIPT]'] == latest_pub_date].copy()

    if point_in_time_ltds.empty:
        return features # Should not happen if valid_ltds is not empty, but a safeguard.

    # --- Proximity & Imminence Features ---
    app_geom = solar_app_row.geometry
    point_in_time_ltds['distance'] = point_in_time_ltds.geometry.distance(app_geom)
    nearest = point_in_time_ltds.loc[point_in_time_ltds['distance'].idxmin()]
    
    features['[REDACTED_BY_SCRIPT]'] = nearest['distance']
    features['[REDACTED_BY_SCRIPT]'] = nearest['[REDACTED_BY_SCRIPT]']
    features['[REDACTED_BY_SCRIPT]'] = nearest['[REDACTED_BY_SCRIPT]'] - app_date.year

    # --- Density & Character Features (10km radius) ---
    buffer = app_geom.buffer(RADIUS_METERS)
    upgrades_in_10km = point_in_time_ltds[point_in_time_ltds.intersects(buffer)]
    
    features['ltds_count_in_10km'] = len(upgrades_in_10km)
    
    if not upgrades_in_10km.empty:
        asset_text = upgrades_in_10km['[REDACTED_BY_SCRIPT]'].str.lower().fillna('')
        features['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('[REDACTED_BY_SCRIPT]').sum()
        features['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('transformer|tx').sum()
        features['[REDACTED_BY_SCRIPT]'] = asset_text.str.contains('[REDACTED_BY_SCRIPT]').sum()
        
    return features

def main():
    """
    Main function to execute the LTDS feature synthesis pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    gdf_ltds_master = load_ltds_data()
    
    df_solar_l11 = pd.read_csv(L11_SOLAR_APPLICATIONS_PATH)
    df_solar_l11['submission_date'] = pd.to_datetime(df_solar_l11['submission_date'], errors='coerce')
    df_solar_l11.dropna(subset=['submission_date', 'easting', 'northing'], inplace=True)
    
    # The 'easting' and 'northing' columns from our L-stage artifacts are, by
    # project mandate, already in the British National Grid CRS. We must directly
    # assign this CRS without performing a transformation. This corrects the
    # faulty projection that was corrupting the solar application coordinates.
    gdf_solar = gpd.GeoDataFrame(
        df_solar_l11,
        geometry=gpd.points_from_xy(df_solar_l11['easting'], df_solar_l11['northing']),
        crs=CRS_PROJECT # Directly assign the correct project CRS.
    )
    
    gdf_solar = gdf_solar.rename(columns=lambda x: x.lower())

    logging.info(f"[REDACTED_BY_SCRIPT]")
    tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
    
    # Apply the temporally-safe function to each row
    new_features_df = gdf_solar.progress_apply(
        lambda row: calculate_features_for_application(row, gdf_ltds_master), 
        axis=1, 
        result_type='expand'
    )

    # Final Integration
    logging.info("[REDACTED_BY_SCRIPT]")
    df_solar_l12 = pd.concat([gdf_solar.drop(columns='geometry'), new_features_df], axis=1)
    
    # Save final artifact
    df_solar_l12.to_csv(L12_OUTPUT_PATH, index=False)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    

if __name__ == '__main__':
    main()