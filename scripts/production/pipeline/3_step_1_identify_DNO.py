import pandas as pd
import geopandas as gpd
import os

# --- Artifact & Project Constants ---
# Mandate: Ingest both pipeline outputs, not a single merged file.
UKPN_LATEST_PATH = r"[REDACTED_BY_SCRIPT]" # ASSUMED PATH
NGED_LATEST_PATH = r"[REDACTED_BY_SCRIPT]" # ASSUMED PATH

DNO_SOURCE_PATH = r"[REDACTED_BY_SCRIPT]"

# Mandate: Create two distinct, stratified output artifacts.
UKPN_STRATIFIED_PATH = r"[REDACTED_BY_SCRIPT]"
NGED_STRATIFIED_PATH = r"[REDACTED_BY_SCRIPT]"

PROJECT_CRS = "EPSG:27700"

def phase_1_load_and_prepare_solar_data(path, crs, is_nged=False):
    """
    Loads a single solar dataset, converts it to a GeoDataFrame, and
    performs necessary schema normalization.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(path)

    # --- Proactive Schema Unification ---
    if is_nged and 'easting' in df.columns:
        print("Normalizing 'easting' -> 'easting_x' in NGED data.")
        df.rename(columns={'easting': 'easting_x', 'northing': 'northing_x'}, inplace=True)

    # Create the GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.easting_x, df.northing_x),
        crs=crs
    )
    print(f"[REDACTED_BY_SCRIPT]")
    return gdf

def phase_2_prepare_dno_boundaries(dno_path, crs, target_dnos=['UKPN', 'NGED']):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    dno_gdf = gpd.read_file(dno_path)
    # IMMEDIATE CRS Unification on hostile data, preventing Pattern 1 (Geospatial Corruption).
    dno_gdf = dno_gdf.to_crs(crs)
    
    # Proactive Schema Normalization
    dno_gdf.columns = dno_gdf.columns.str.lower()
    
    # Isolate polygons for all target DNOs
    dno_boundaries = dno_gdf[dno_gdf['dno'].isin(target_dnos)].copy()
    
    print(f"[REDACTED_BY_SCRIPT]'dno'[REDACTED_BY_SCRIPT]")
    return dno_boundaries[['dno', 'geometry']] # Return only essential columns


def phase_3_filter_by_dno(solar_gdf, dno_boundaries, target_dno):
    """
    Filters the solar GeoDataFrame to keep only the sites that are
    geographically within the specified target DNO boundary.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Isolate the specific DNO boundary for this operation
    target_boundary = dno_boundaries[dno_boundaries['dno'] == target_dno]
    
    # Use an INNER join to keep only points that fall WITHIN the target boundary
    filtered_gdf = gpd.sjoin(solar_gdf, target_boundary, how='inner', predicate='within')

    # --- ARCHITECTURAL MANDATE: ANOMALY REPORTING & DEDUPLICATION ---
    # Identify sites duplicated by the join (i.e., existing in multiple DNO boundaries).
    if 'application_id' in filtered_gdf.columns:
        duplicated_mask = filtered_gdf.duplicated(subset=['application_id'], keep=False)
        if duplicated_mask.any():
            duplicated_ids = filtered_gdf[duplicated_mask]['application_id'].unique()
            print(f"[REDACTED_BY_SCRIPT]")
            # print(f"[REDACTED_BY_SCRIPT]") # Optional: for deep diagnostics

        # Enforce a one-to-one relationship by keeping only the first DNO match for any given site.
        initial_count = len(filtered_gdf)
        filtered_gdf.drop_duplicates(subset=['application_id'], keep='first', inplace=True)
        if len(filtered_gdf) < initial_count:
            print(f"[REDACTED_BY_SCRIPT]")

    # Clean up columns from the join
    filtered_gdf.drop(columns=['index_right', 'dno'], inplace=True, errors='ignore')
    
    print(f"[REDACTED_BY_SCRIPT]")
    return filtered_gdf

def phase_4_export_artifact(filtered_gdf, output_path):
    """[REDACTED_BY_SCRIPT]"""
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Drop the geometry column before saving to CSV
    df_to_save = filtered_gdf.drop(columns=['geometry'], errors='ignore')
    
    df_to_save.to_csv(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    return df_to_save

if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    
    # Phase 2: Prepare the authoritative DNO boundaries (do this once)
    dno_boundaries = phase_2_prepare_dno_boundaries(DNO_SOURCE_PATH, PROJECT_CRS)
    
    # --- Process UKPN Data ---
    print("[REDACTED_BY_SCRIPT]")
    ukpn_solar_gdf = phase_1_load_and_prepare_solar_data(UKPN_LATEST_PATH, PROJECT_CRS)
    ukpn_solar_gdf['dno_source'] = 'ukpn'
    ukpn_filtered_gdf = phase_3_filter_by_dno(ukpn_solar_gdf, dno_boundaries, 'UKPN')
    df_ukpn = phase_4_export_artifact(ukpn_filtered_gdf, UKPN_STRATIFIED_PATH)
    
    # --- Process NGED Data ---
    print("[REDACTED_BY_SCRIPT]")
    nged_solar_gdf = phase_1_load_and_prepare_solar_data(NGED_LATEST_PATH, PROJECT_CRS, is_nged=True)
    nged_solar_gdf['dno_source'] = 'nged'
    nged_filtered_gdf = phase_3_filter_by_dno(nged_solar_gdf, dno_boundaries, 'NGED')
    df_nged = phase_4_export_artifact(nged_filtered_gdf, NGED_STRATIFIED_PATH)
    
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]", df_ukpn.shape)
    print("[REDACTED_BY_SCRIPT]", df_nged.shape)