import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import stats
from pathlib import Path
import re
import warnings

# Suppress SettingWithCopyWarning, as our chained operations are controlled and intentional.
pd.options.mode.chained_assignment = None

def sanitize_col_name(col_name):
    """[REDACTED_BY_SCRIPT]"""
    s = str(col_name).strip().lower()
    s = re.sub(r'\s+', '_', s)  # Replace spaces with underscores
    s = re.sub(r'[^0-9a-zA-Z_]', '', s)  # Remove invalid characters
    s = re.sub(r'^(\d)', r'_\1', s) # Prepend underscore if starts with a digit
    return s

def harmonize_oa_data(data_dir: Path, output_path: Path):
    """
    Executes Phase 1: Ingests, decontaminates, pivots, and merges all OA CSVs
    into a single, clean, non-spatial master table.
    """
    print("[REDACTED_BY_SCRIPT]")
    data_dir = Path(data_dir)

    # Establish Backbone with population data
    pop_df = pd.read_csv(data_dir / 'OA - population.csv', index_col='2021 output area')
    pop_df.columns = [f"pop_{c}" for c in pop_df.columns]
    pop_df.index.name = 'oa21cd'
    master_df = pop_df

    # Define categorical datasets to process
    categorical_files = {
        'dep': ('ons-deprivation.csv', '[REDACTED_BY_SCRIPT]'),
        'edu': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'emp': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'hhd_comp': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'hhd_stu': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'hhd_size': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'hhd_adults': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'hhd_adults_emp': ('[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'),
        'car': ('ons-vehicles.csv', '[REDACTED_BY_SCRIPT]'),
    }

    for prefix, (filename, desc_col) in categorical_files.items():
        print(f"[REDACTED_BY_SCRIPT]")
        df = pd.read_csv(data_dir / filename)
        
        # CRITICAL MANDATE: Decontamination
        code_col = [c for c in df.columns if 'Code' in c and 'Area' not in c][0]
        df = df[df[code_col] != -8]

        # Mandated Pivot
        pivot_df = df.pivot_table(
            index='Output Areas Code',
            columns=desc_col,
            values='Observation',
            fill_value=0
        )
        
        # Mandated Schema Sanitization
        pivot_df.columns = [f"[REDACTED_BY_SCRIPT]" for c in pivot_df.columns]
        pivot_df.index.name = 'oa21cd'
        
        master_df = master_df.join(pivot_df, how='left')

    # Ingest & Join OAC Data
    print("[REDACTED_BY_SCRIPT]")
    oac_df = pd.read_csv(data_dir / '[REDACTED_BY_SCRIPT]', index_col='oa21cd')
    oac_df.columns = [sanitize_col_name(c) for c in oac_df.columns]
    master_df = master_df.join(oac_df, how='left')

    # Final Audit & Persistence
    print("[REDACTED_BY_SCRIPT]")
    hhd_comp_cols = [c for c in master_df.columns if c.startswith('hhd_comp_')]
    master_df['total_households'] = master_df[hhd_comp_cols].sum(axis=1)

    # Paranoid Type Casting
    for col in master_df.columns:
        if 'oac' not in col and 'group' not in col: # OAC codes are strings
             # MANDATE: Use a sentinel value (-1.0) for missing data, not an ambiguous zero.
             master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(-1.0).astype('float64')

    master_df.to_csv(output_path)
    print(f"[REDACTED_BY_SCRIPT]")
    return output_path

def create_geospatial_master(l1_artifact_path: Path, oa_boundaries_path: Path, output_path: Path):
    """
    Executes Phase 2: Fuses the L1 master table with OA boundaries, enforces
    CRS, and creates the L2 master geospatial artifact.
    """
    print("[REDACTED_BY_SCRIPT]")
    # Load non-spatial data
    l1_df = pd.read_csv(l1_artifact_path, index_col='oa21cd')

    # Load spatial data
    oa_gdf = gpd.read_file(oa_boundaries_path)

    # CRITICAL MANDATE: Verify and enforce CRS
    if oa_gdf.crs != "EPSG:27700":
        print(f"[REDACTED_BY_SCRIPT]")
        oa_gdf = oa_gdf.to_crs("EPSG:27700")
    
    # Standardize join key
    oa_gdf = oa_gdf.rename(columns={'OA21CD': 'oa21cd'})

    # Perform attribute join
    merged_gdf = oa_gdf[['oa21cd', 'geometry']].merge(l1_df, on='oa21cd', how='inner')

    merged_gdf.to_file(output_path, driver='GPKG', layer='oa_socioeconomic')
    print(f"[REDACTED_BY_SCRIPT]")
    return output_path

def synthesize_strategic_indices(l2_artifact_path: Path, output_path: Path):
    """
    Executes Phase 3: Enriches the L2 master artifact with the approved,
    hypothesis-driven strategic indices at the individual OA level.
    """
    print("[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(l2_artifact_path, layer='oa_socioeconomic')

    # Handle potential division by zero
    gdf['total_households'] = gdf['total_households'].replace(0, np.nan)

    # Synthesize oa_dep_multi_dim_idx
    gdf['oa_dep_multi_dim_idx'] = (
        gdf['[REDACTED_BY_SCRIPT]'] * 2 +
        gdf['[REDACTED_BY_SCRIPT]'] * 3 +
        gdf['[REDACTED_BY_SCRIPT]'] * 4
    ) / gdf['total_households']

    # Synthesize oa_household_stability_idx
    gdf['[REDACTED_BY_SCRIPT]'] = (
        gdf['[REDACTED_BY_SCRIPT]'] +
        gdf['[REDACTED_BY_SCRIPT]']
    ) / gdf['total_households']

    # Synthesize oa_pop_trend
    pop_cols = [f'pop_{year}' for year in range(2018, 2023)]
    time_axis = np.arange(len(pop_cols))
    
    def get_trend(row):
        # Convert to numpy array to ensure robust, type-safe operations
        pop_values = row[pop_cols].to_numpy()
        
        # Create a pure numpy boolean mask for valid (non-zero) population values
        valid_mask = pop_values > 0
        num_valid_points = np.sum(valid_mask)
        
        # A regression requires at least two points
        if num_valid_points < 2:
            return 0.0
        
        # Filter both the population values and the time axis using the same numpy mask
        # This guarantees dimensional consistency for linregress
        valid_pops = pop_values[valid_mask]
        valid_time = time_axis[valid_mask]
        
        # Handle a rare case where all valid values are identical (slope is 0, but stderr is NaN)
        if np.all(valid_pops == valid_pops[0]):
            return 0.0

        # FINAL SAFEGUARD: Aggressively cast inputs to numpy arrays of a consistent float dtype.
        # This acts as a firewall against upstream data type degradation from the .apply() method,
        # guaranteeing that linregress receives valid, array-like inputs and preventing the AttributeError.
        x_in = np.array(valid_time, dtype=np.float64)
        y_in = np.array(valid_pops, dtype=np.float64)

        slope, _, _, _, _ = stats.linregress(x_in, y_in)
        return slope

    gdf['oa_pop_trend'] = gdf.apply(get_trend, axis=1)

    # Synthesize oa_pop_density_per_sqkm
    gdf['[REDACTED_BY_SCRIPT]'] = gdf['pop_2022'] / (gdf.geometry.area / 1_000_000)

    # One-Hot Encode OAC (Defensive Implementation)
    oac_cols_to_encode = ['supergroup', 'group', 'subgroup']
    prefix_map = {'supergroup': 'oac_sg', 'group': 'oac_g', 'subgroup': 'oac_subg'}

    # Dynamically find which of the expected OAC columns actually exist in the dataframe
    existing_oac_cols = [col for col in oac_cols_to_encode if col in gdf.columns]

    if existing_oac_cols:
        print(f"[REDACTED_BY_SCRIPT]")
        dynamic_prefix = {k: v for k, v in prefix_map.items() if k in existing_oac_cols}
        
        # Bypass the fragile pre-slicing by using the 'columns' argument.
        # This delegates the column selection to the robust internal logic of get_dummies,
        # making the operation resilient to schema corruptions that cause slicing to fail.
        gdf = pd.get_dummies(gdf, columns=existing_oac_cols, prefix=dynamic_prefix, dtype=int)
    else:
        print("[REDACTED_BY_SCRIPT]'supergroup', 'group', 'subgroup') found to encode.")

    # Clean up and fill NaNs created by division
    index_cols = ['oa_dep_multi_dim_idx', '[REDACTED_BY_SCRIPT]', 'oa_pop_trend', '[REDACTED_BY_SCRIPT]']
    # MANDATE: Eradicate ambiguous zeros; use a sentinel value for missing data.
    gdf[index_cols] = gdf[index_cols].fillna(-1.0)

    gdf.to_file(output_path, driver='GPKG', layer='oa_socioeconomic_enriched')
    print(f"[REDACTED_BY_SCRIPT]")
    return output_path

def perform_dual_context_aggregation(solar_sites_path: Path, l2_enriched_artifact_path: Path):
    """
    Executes Phase 4: Engineers final features for each solar application using
    the two mandated aggregation protocols.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # Harden ingestion gate against CRS corruption from non-standard file formats
    if str(solar_sites_path).lower().endswith('.csv'):
        print("[REDACTED_BY_SCRIPT]")
        sites_df = pd.read_csv(solar_sites_path)
        
        # HIERARCHICAL FALLBACK PROTOCOL
        # Step 1: Attempt to find a WKT geometry column.
        possible_wkt_cols = ['geometry', 'WKT', 'wkt', 'geom']
        geom_col_found = next((col for col in possible_wkt_cols if col in sites_df.columns), None)

        if geom_col_found:
            print(f"[REDACTED_BY_SCRIPT]'{geom_col_found}'[REDACTED_BY_SCRIPT]")
            sites_gdf = gpd.GeoDataFrame(
                sites_df, 
                geometry=gpd.GeoSeries.from_wkt(sites_df[geom_col_found]), 
                crs="EPSG:4326" # WKT from generic CSVs often defaults to WGS84
            )
        else:
            # Step 2: If WKT fails, attempt to construct geometry from coordinate pairs.
            print("[REDACTED_BY_SCRIPT]")
            possible_easting_cols = ['easting', 'east', 'x', 'BNG_E']
            possible_northing_cols = ['northing', 'north', 'y', 'BNG_N']
            
            easting_col = next((col for col in possible_easting_cols if col in sites_df.columns), None)
            northing_col = next((col for col in possible_northing_cols if col in sites_df.columns), None)

            if easting_col and northing_col:
                print(f"[REDACTED_BY_SCRIPT]'{easting_col}', '{northing_col}'[REDACTED_BY_SCRIPT]")
                # CRITICAL: Assign the correct CRS for British National Grid at point of creation.
                sites_gdf = gpd.GeoDataFrame(
                    sites_df,
                    geometry=gpd.points_from_xy(sites_df[easting_col], sites_df[northing_col]),
                    crs="EPSG:27700" 
                )
            else:
                # Step 3: If both methods fail, raise a comprehensive fatal error.
                raise ValueError(
                    "[REDACTED_BY_SCRIPT]"
                    f"[REDACTED_BY_SCRIPT]"
                    f"[REDACTED_BY_SCRIPT]"
                )
    else:
        sites_gdf = gpd.read_file(solar_sites_path)

    # Dynamic, resilient search for the site primary key to prevent schema brittleness.
    possible_id_cols = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'app_ref', 'application_number', 'planning_ref']
    site_id_col = next((col for col in possible_id_cols if col in sites_gdf.columns), None)

    if site_id_col is None:
        print("[REDACTED_BY_SCRIPT]")
        # PROVISIONAL KEY SYNTHESIS PROTOCOL
        # This is a tactical compromise to handle a deficient data artifact.
        # It creates a deterministic key from unique row data instead of relying on an unstable index.
        key_vector_cols = [easting_col, northing_col, '[REDACTED_BY_SCRIPT]', 'submission_year']
        
        if all(col in sites_gdf.columns for col in key_vector_cols):
            print(f"[REDACTED_BY_SCRIPT]")
            sites_gdf['prototype_uid'] = sites_gdf[key_vector_cols].astype(str).agg('_'.join, axis=1)
            
            # Add the new provisional key to the search list and re-run the check.
            possible_id_cols.append('prototype_uid')
            site_id_col = next((col for col in possible_id_cols if col in sites_gdf.columns), None)
        
        if site_id_col is None:
            # If synthesis also fails, the error is now more comprehensive.
            raise ValueError(
                "[REDACTED_BY_SCRIPT]"
                f"[REDACTED_BY_SCRIPT]"
                f"[REDACTED_BY_SCRIPT]"
            )

    print(f"[REDACTED_BY_SCRIPT]'{site_id_col}'")

    oa_gdf = gpd.read_file(l2_enriched_artifact_path, layer='oa_socioeconomic_enriched')

    # Ensure both are in the mandated CRS
    if sites_gdf.crs != "EPSG:27700": sites_gdf = sites_gdf.to_crs("EPSG:27700")
    if oa_gdf.crs != "EPSG:27700": oa_gdf = oa_gdf.to_crs("EPSG:27700")

    # --- Part A: Buffered Area-Weighted Average (Primary Method) ---
    print("[REDACTED_BY_SCRIPT]")
    site_buffers = sites_gdf.copy()
    site_buffers['geometry'] = site_buffers.geometry.buffer(1000)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        intersection = gpd.overlay(site_buffers, oa_gdf, how='intersection')

    # Calculate Weights
    intersection['fragment_area'] = intersection.geometry.area
    buffer_total_areas = intersection.groupby(site_id_col)['fragment_area'].sum().to_dict()
    intersection['buffer_total_area'] = intersection[site_id_col].map(buffer_total_areas)
    intersection['weight'] = intersection['fragment_area'] / intersection['buffer_total_area']

    # Calculate Weighted Averages
    feature_cols = [c for c in oa_gdf.columns if c.startswith(('oa_', 'oac_'))]
    for col in feature_cols:
        intersection[col] = intersection[col] * intersection['weight']

    buffered_features = intersection.groupby(site_id_col)[feature_cols].sum()
    buffered_features.columns = [f"oa_buffered_{c}" for c in buffered_features.columns]

    # --- Part B: Dominant Character (Secondary Method) ---
    print("[REDACTED_BY_SCRIPT]")
    site_centroids = sites_gdf.copy()
    site_centroids['geometry'] = site_centroids.geometry.centroid

    centroid_join = gpd.sjoin(site_centroids, oa_gdf[feature_cols + ['geometry']], how='left', predicate='within')
    
    # MANDATE: Apply de-duplication guard rail immediately after the join for architectural consistency.
    centroid_features = centroid_join.drop_duplicates(subset=site_id_col, keep='first')
    centroid_features = centroid_features.drop(columns=['index_right', 'geometry'])
    centroid_features = centroid_features.set_index(site_id_col)
    centroid_features.columns = [f"oa_centroid_{c}" for c in centroid_features.columns]
    
    # --- Merge features back to main solar farm DataFrame ---
    final_df = sites_gdf.set_index(site_id_col)
    final_df = final_df.join(buffered_features)
    final_df = final_df.join(centroid_features)
    
    print("[REDACTED_BY_SCRIPT]")
    return final_df.reset_index()

def synthesize_final_sics(enriched_solar_sites_gdf: gpd.GeoDataFrame, output_path: Path):
    """
    Executes Phase 5: Creates the final, surgical interaction features
    using the newly aggregated data.
    """
    print("[REDACTED_BY_SCRIPT]")
    df = enriched_solar_sites_gdf

    # SIC 7: "[REDACTED_BY_SCRIPT]"
    df['[REDACTED_BY_SCRIPT]'] = df['oa_buffered_oa_pop_trend'] * df['[REDACTED_BY_SCRIPT]']

    # SIC 8: "[REDACTED_BY_SCRIPT]"
    # Per blueprint, assuming Supergroup 1 represents the target demographic.
    # This must be verified against OAC documentation.
    affluent_col = '[REDACTED_BY_SCRIPT]'
    if affluent_col not in df.columns:
        print(f"  Warning: Column '{affluent_col}'[REDACTED_BY_SCRIPT]")
        df[affluent_col] = 0
        
    df['[REDACTED_BY_SCRIPT]'] = df[affluent_col] * df['[REDACTED_BY_SCRIPT]']

    df.to_file(output_path, driver='GPKG', layer='[REDACTED_BY_SCRIPT]')
    print(f"[REDACTED_BY_SCRIPT]")
    return output_path


def main():
    """[REDACTED_BY_SCRIPT]"""
    # Define project paths - these must be configured for the execution environment
    INPUT_DIR = r"[REDACTED_BY_SCRIPT]"

    # Input file paths
    OA_BOUNDARIES_PATH = r"[REDACTED_BY_SCRIPT]"
    SOLAR_SITES_PATH = r"[REDACTED_BY_SCRIPT]"

    # Artifact paths
    L1_ARTIFACT_PATH = r"[REDACTED_BY_SCRIPT]"
    L2_ARTIFACT_PATH = r"[REDACTED_BY_SCRIPT]"
    L2_ENRICHED_PATH = r"[REDACTED_BY_SCRIPT]"
    FINAL_OUTPUT_PATH = r"[REDACTED_BY_SCRIPT]"

    # Execute pipeline
    harmonize_oa_data(data_dir=INPUT_DIR, output_path=L1_ARTIFACT_PATH)
    create_geospatial_master(l1_artifact_path=L1_ARTIFACT_PATH, oa_boundaries_path=OA_BOUNDARIES_PATH, output_path=L2_ARTIFACT_PATH)
    synthesize_strategic_indices(l2_artifact_path=L2_ARTIFACT_PATH, output_path=L2_ENRICHED_PATH)
    enriched_solar_sites_gdf = perform_dual_context_aggregation(solar_sites_path=SOLAR_SITES_PATH, l2_enriched_artifact_path=L2_ENRICHED_PATH)
    synthesize_final_sics(enriched_solar_sites_gdf, output_path=FINAL_OUTPUT_PATH)

    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()