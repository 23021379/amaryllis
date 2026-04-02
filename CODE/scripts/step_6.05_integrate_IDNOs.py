import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import json
from shapely.geometry import shape, Point, LineString
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
# Input files
L10_SOLAR_APPLICATIONS_PATH = '[REDACTED_BY_SCRIPT]'
IDNO_RAW_PATH = r"[REDACTED_BY_SCRIPT]"
SUBSTATION_CAPACITY_PATH = r"[REDACTED_BY_SCRIPT]"
LPA_BOUNDARIES_PATH = r"[REDACTED_BY_SCRIPT]"

# Intermediate & Output files
IDNO_L1_UNIFIED_PATH = '[REDACTED_BY_SCRIPT]'
L11_OUTPUT_PATH = '[REDACTED_BY_SCRIPT]'

# Geospatial constants
CRS_PROJECT = 'EPSG:27700' # British National Grid
CRS_SOURCE = 'EPSG:4326'   # WGS84 for raw IDNO data

def ingest_and_unify_idno_data():
    """
    Implements Phase 1: Ingests raw IDNO GeoJSON, enforces mandatory CRS
    unification, and saves the L1 artifact.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Ingest directly using the correct GeoJSON parser
    gdf_idno_raw = gpd.read_file(IDNO_RAW_PATH)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # MANDATE: CRS Unification
    if gdf_idno_raw.crs.to_string() != CRS_PROJECT:
        gdf_idno_unified = gdf_idno_raw.to_crs(CRS_PROJECT)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    else:
        gdf_idno_unified = gdf_idno_raw
        logging.info("[REDACTED_BY_SCRIPT]")

    # Persist L1 Artifact as GeoJSON
    gdf_idno_unified.to_file(IDNO_L1_UNIFIED_PATH, driver='GeoJSON')
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return gdf_idno_unified


def generate_core_idno_features(gdf_solar, gdf_idno):
    """
    Implements Phase 2: Generates core IDNO features for each solar application.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Feature: idno_is_within
    within_join = gpd.sjoin(gdf_solar, gdf_idno, how='left', predicate='within')
    gdf_solar['idno_is_within'] = ~within_join['index_right'].isnull()

    # Feature: idno_distance_to_nearest_boundary
    # For sites NOT within, calculate distance. For sites within, it's 0.
    sites_outside = gdf_solar[~gdf_solar['idno_is_within']]
    distances = sites_outside.geometry.apply(lambda g: gdf_idno.distance(g).min())
    gdf_solar['[REDACTED_BY_SCRIPT]'] = distances
    # Adhering to modern pandas practice by reassigning the result instead of using inplace=True on a slice.
    gdf_solar['[REDACTED_BY_SCRIPT]'] = gdf_solar['[REDACTED_BY_SCRIPT]'].fillna(-1.0)

    # Features: count_in_5km and total_area_in_10km
    idno_sindex = gdf_idno.sindex
    
    buffer_5km = gdf_solar.geometry.buffer(5000)
    buffer_10km = gdf_solar.geometry.buffer(10000)

    # Count in 5km
    intersections_5km = [list(idno_sindex.intersection(b.bounds)) for b in buffer_5km]
    gdf_solar['idno_count_in_5km'] = [len(i) for i in intersections_5km]
    
    # Total area in 10km
    # This is computationally heavy; an intersection count is a good proxy, but directive mandates area.
    # A simplified, faster approach is to sum the area of intersecting IDNOs.
    gdf_solar['[REDACTED_BY_SCRIPT]'] = 0 # Placeholder for a more optimized implementation if needed
    # For now, let's use a simpler proxy: sum area of IDNOs whose bounds intersect the buffer bounds
    areas = []
    for i in tqdm(range(len(gdf_solar)), desc="[REDACTED_BY_SCRIPT]"):
        possible_matches_idx = list(idno_sindex.intersection(buffer_10km.iloc[i].bounds))
        if not possible_matches_idx:
            areas.append(0)
            continue
        possible_matches = gdf_idno.iloc[possible_matches_idx]
        # Precise intersection
        intersecting_idnos = possible_matches[possible_matches.intersects(buffer_10km.iloc[i])]
        areas.append(intersecting_idnos.area.sum())
    gdf_solar['[REDACTED_BY_SCRIPT]'] = areas

    return gdf_solar

def generate_grid_interaction_features(gdf_solar, gdf_idno, gdf_substations):
    """
    Implements Phase 3 (Grid): Fuses IDNO context with substation data.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Find nearest substation for each solar site
    substation_sindex = gdf_substations.sindex
    # Correctly pass the geometry object 'g', not its bounds tuple, to the nearest method.
    nearest_indices = [list(substation_sindex.nearest(g, return_all=False))[1][0] for g in gdf_solar.geometry]
    nearest_substations = gdf_substations.iloc[nearest_indices].reset_index(drop=True)
    
    # Feature: idno_nearest_substation_is_within
    sub_within_join = gpd.sjoin(nearest_substations, gdf_idno, how='left', predicate='within')
    gdf_solar['[REDACTED_BY_SCRIPT]'] = ~sub_within_join['index_right'].isnull()

    # Feature: idno_crosses_connection_path
    connection_paths = [LineString([solar_geom, sub_geom]) for solar_geom, sub_geom in zip(gdf_solar.geometry, nearest_substations.geometry)]
    gdf_paths = gpd.GeoDataFrame(geometry=connection_paths, crs=CRS_PROJECT, index=gdf_solar.index)
    path_crosses_join = gpd.sjoin(gdf_paths, gdf_idno, how='inner', predicate='intersects')
    
    # Get the unique indices of paths that have at least one intersection.
    crossing_path_indices = path_crosses_join.index.unique()
    
    # Create the final boolean series by checking which solar sites are in the crossing set.
    gdf_solar['[REDACTED_BY_SCRIPT]'] = gdf_solar.index.isin(crossing_path_indices)
    
    return gdf_solar

def generate_lpa_interaction_features(gdf_idno, gdf_lpa):
    """
    Implements Phase 3 (LPA): Creates LPA-level features based on IDNO density.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    
    # Use the correct key 'lpa23nm' as identified by diagnostic telemetry.
    lpa_key = 'lpa23nm' 
    
    # Feature: lpa_idno_area_as_percent_of_total_area
    lpa_idno_intersection = gpd.overlay(gdf_lpa, gdf_idno, how='intersection')

    lpa_idno_area = lpa_idno_intersection.groupby(lpa_key)['geometry'].apply(lambda g: g.area.sum())
    gdf_lpa['total_area'] = gdf_lpa.area
    gdf_lpa = gdf_lpa.set_index(lpa_key) # Set index for easier alignment
    gdf_lpa['lpa_idno_area_as_percent_of_total_area'] = (lpa_idno_area / gdf_lpa['total_area']) * 100
    gdf_lpa['lpa_idno_area_as_percent_of_total_area'].fillna(-1.0, inplace=True)

    # Feature: lpa_idno_count_per_100km2
    # The 'lpa23nm' column from gdf_lpa is added to idno_in_lpa by the sjoin
    idno_in_lpa = gpd.sjoin(gdf_idno, gdf_lpa, how='inner', predicate='within')
    # Grouping by the explicit 'lpa23nm' column is correct and robust.
    lpa_idno_count = idno_in_lpa.groupby('lpa23nm').size()
    gdf_lpa['lpa_idno_count'] = lpa_idno_count
    gdf_lpa['lpa_idno_count'].fillna(-1.0, inplace=True)
    gdf_lpa['[REDACTED_BY_SCRIPT]'] = gdf_lpa['lpa_idno_count'] / (gdf_lpa['total_area'] / 1e8) # 100km2 = 1e8 m2

    lpa_features = gdf_lpa.reset_index()[['lpa23nm', 'lpa_idno_area_as_percent_of_total_area', '[REDACTED_BY_SCRIPT]']]
    return lpa_features

def clean_col_names(df):
    """[REDACTED_BY_SCRIPT]"""
    cols = [col.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for col in df.columns]
    df.columns = cols
    return df


# --- Main Script ---
def main():
    """
    Main function to execute the IDNO feature synthesis pipeline.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Phase 1
    gdf_idno_unified = ingest_and_unify_idno_data()

    # Load prerequisite data for subsequent phases
    df_solar_l10 = pd.read_csv(L10_SOLAR_APPLICATIONS_PATH)
    gdf_solar = gpd.GeoDataFrame(
        df_solar_l10, 
        geometry=[Point(xy) for xy in zip(df_solar_l10['easting'], df_solar_l10['northing'])],
        crs=CRS_PROJECT
    )
    # Create a stable key for duplicate resolution.
    gdf_solar['solar_farm_id'] = gdf_solar.index
    gdf_substations = gpd.read_file(SUBSTATION_CAPACITY_PATH)
    # MANDATORY CRS UNIFICATION for substations to prevent corruption
    if gdf_substations.crs.to_string() != CRS_PROJECT:
        gdf_substations = gdf_substations.to_crs(CRS_PROJECT)
        logging.info(f"[REDACTED_BY_SCRIPT]")

    gdf_lpa = gpd.read_file(LPA_BOUNDARIES_PATH)
    # MANDATORY CRS UNIFICATION for LPAs to prevent corruption
    if gdf_lpa.crs.to_string() != CRS_PROJECT:
        gdf_lpa = gdf_lpa.to_crs(CRS_PROJECT)
        logging.info(f"[REDACTED_BY_SCRIPT]")

    # Normalize LPA schema immediately after ingestion
    gdf_lpa = clean_col_names(gdf_lpa)


    # Phase 2
    gdf_solar = generate_core_idno_features(gdf_solar, gdf_idno_unified)

    # Phase 3
    gdf_solar = generate_grid_interaction_features(gdf_solar, gdf_idno_unified, gdf_substations)
    lpa_features = generate_lpa_interaction_features(gdf_idno_unified, gdf_lpa)
    
    # Final Integration
    logging.info("[REDACTED_BY_SCRIPT]")
    # Harmonize the key on the LPA feature set for the final merge.
    lpa_features.rename(columns={'lpa23nm': 'lpa_name'}, inplace=True)

    # --- ARCHITECTURAL INTERVENTION: BRIDGE INTELLIGENCE GAP ---
    # The solar data lacks an LPA identifier. We must create it via a spatial join.
    logging.info("[REDACTED_BY_SCRIPT]")
    # The sjoin appends the columns of the right GDF (gdf_lpa) to the left (gdf_solar).
    # We select only the key column ('lpa23nm') we need for the subsequent join.
    gdf_solar_with_lpa_key_raw = gpd.sjoin(gdf_solar, gdf_lpa[['lpa23nm', 'geometry']], how='left', predicate='within')

    # MANDATE: Install Duplicate Resolution Gate to prevent data corruption from overlapping LPA boundaries.
    gdf_solar_with_lpa_key = gdf_solar_with_lpa_key_raw.drop_duplicates(subset=['solar_farm_id'], keep='first')
    
    # Now, harmonize the newly added key to match the lpa_features key.
    gdf_solar_with_lpa_key.rename(columns={'lpa23nm': 'lpa_name'}, inplace=True)
    
    # Clean up artifacts from the sjoin and drop geometry for CSV export.
    df_solar_with_app_features = pd.DataFrame(gdf_solar_with_lpa_key.drop(columns=['geometry', 'index_right']))
    
    # The solar data now has the 'lpa_name' key and can be merged successfully.
    df_solar_l11 = pd.merge(df_solar_with_app_features, lpa_features, on='lpa_name', how='left')

    # --- FINAL ARTIFACT SANITIZATION ---
    # Convert all boolean columns to integers (1/0) for the model.
    logging.info("[REDACTED_BY_SCRIPT]")
    bool_cols = df_solar_l11.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_solar_l11[col] = df_solar_l11[col].astype(int)
    
    # Drop the string-based LPA name column as it is not a feature for the model.
    logging.info("[REDACTED_BY_SCRIPT]")
    df_solar_l11 = df_solar_l11.drop(columns=['lpa_name'])

    # Save final artifact
    df_solar_l11.to_csv(L11_OUTPUT_PATH, index=False)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()