import geopandas as gpd
import pandas as pd
import logging
import os

# --- Executor Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
ECR_SUB_1MW_PATH = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
ECR_OVER_1MW_PATH = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
NULL_SENTINEL = 0.0 # For counts and sums, 0 is a more appropriate sentinel
RADII_METERS = [2000, 5000, 10000]

# --- Module-level State for Performance ---
gdf_ecr = None
ecr_sindex = None

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


def _initialize_module_state():
    """
    Loads, unifies, and distills the two ECR GeoJSON sources into an
    analysis-ready, indexed GeoDataFrame. This process is performed once per worker.
    """
    global gdf_ecr, ecr_sindex
    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        
        # 1. Ingest raw data
        gdf_sub1mw = gpd.read_file(ECR_SUB_1MW_PATH)
        gdf_over1mw = gpd.read_file(ECR_OVER_1MW_PATH)

        # 2. MANDATE: Enforce project CRS immediately upon ingestion (Pattern 1 Defense)
        gdf_sub1mw = gdf_sub1mw.to_crs(PROJECT_CRS)
        gdf_over1mw = gdf_over1mw.to_crs(PROJECT_CRS)

        # 3. Normalize schemas and tag sources
        gdf_sub1mw = clean_col_names(gdf_sub1mw)
        gdf_over1mw = clean_col_names(gdf_over1mw)
        gdf_sub1mw['capacity_scale'] = 'sub_1mw'
        gdf_over1mw['capacity_scale'] = 'over_1mw'

        # 4. Unify into a single GeoDataFrame
        unified_gdf = pd.concat([gdf_sub1mw, gdf_over1mw], ignore_index=True)
        unified_gdf = gpd.GeoDataFrame(unified_gdf, geometry='geometry', crs=PROJECT_CRS)
        
        # 5. Distill and filter the unified data
        # Coerce capacity columns to numeric, handling errors
        capacity_cols = [c for c in unified_gdf.columns if 'capacity_mw' in c or 'storage_mwh' in c]
        for col in capacity_cols:
            unified_gdf[col] = pd.to_numeric(unified_gdf[col], errors='coerce')

        # Aggregate capacities
        unified_gdf['[REDACTED_BY_SCRIPT]'] = unified_gdf.filter(like='capacity_mw').sum(axis=1)
        unified_gdf['total_storage_mwh'] = unified_gdf.filter(like='storage_mwh').sum(axis=1)

        # Select essential features
        essential_cols = [
            'geometry', 'capacity_scale', 'date_accepted', 'connection_status',
            'energy_source_1', '[REDACTED_BY_SCRIPT]', 'total_storage_mwh'
        ]
        # Ensure all essential columns exist, adding missing ones if necessary
        for col in essential_cols:
            if col not in unified_gdf.columns:
                unified_gdf[col] = None

        distilled_gdf = unified_gdf[essential_cols]

        # Apply status and temporal filters
        distilled_gdf = distilled_gdf[distilled_gdf['connection_status'].isin(['Connected', 'Accepted to Connect'])].copy()
        distilled_gdf['date_accepted'] = pd.to_datetime(distilled_gdf['date_accepted'], errors='coerce')
        distilled_gdf.dropna(subset=['date_accepted', 'geometry'], inplace=True)
        
        # PRE-OPERATIVE CHECK: Final CRS Validation
        assert distilled_gdf.crs.to_string() == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"

        # 6. Set final state and build spatial index
        gdf_ecr = distilled_gdf
        ecr_sindex = gdf_ecr.sindex
        logging.info(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
        gdf_ecr = gpd.GeoDataFrame()
        ecr_sindex = None

def execute(state: dict) -> dict:
    """
    Calculates spatially and temporally aware DER density features.
    """
    if ecr_sindex is None:
        _initialize_module_state()
        if ecr_sindex is None:
            state['errors'].append("[REDACTED_BY_SCRIPT]")
            return state
            
    try:
        site_geom = state['input_geom']
        # The temporal context is critical to prevent feature leakage (Pattern 3)
        submission_date = pd.to_datetime(f"{state['submission_year']}-01-01")

        for r in RADII_METERS:
            r_km = r // 1000
            
            # Spatial query using the index
            possible_matches_idx = list(ecr_sindex.intersection(site_geom.buffer(r).bounds))
            possible_matches = gdf_ecr.iloc[possible_matches_idx]
            
            # Precise spatial filter
            precise_matches = possible_matches[possible_matches.intersects(site_geom.buffer(r))]
            
            # CRITICAL TEMPORAL FILTER: Only consider precedents accepted *before* the submission.
            actual_precedents = precise_matches[precise_matches['date_accepted'] < submission_date]
            
            # --- Stratify and Calculate Features ---
            strata = {
                'sub1mw': actual_precedents[actual_precedents['capacity_scale'] == 'sub_1mw'],
                'over1mw': actual_precedents[actual_precedents['capacity_scale'] == 'over_1mw'],
            }
            
            for name, df_stratum in strata.items():
                count = len(df_stratum)
                capacity_mw = df_stratum['[REDACTED_BY_SCRIPT]'].sum()
                storage_mwh = df_stratum['total_storage_mwh'].sum()
                solar_count = df_stratum[df_stratum['energy_source_1'] == 'Solar Photovoltaics'].shape[0]
                
                state['features'][f'[REDACTED_BY_SCRIPT]'] = count
                state['features'][f'[REDACTED_BY_SCRIPT]'] = capacity_mw
                state['features'][f'[REDACTED_BY_SCRIPT]'] = storage_mwh
                state['features'][f'[REDACTED_BY_SCRIPT]'] = solar_count / count if count > 0 else 0.0

            state['features'][f'[REDACTED_BY_SCRIPT]'] = len(actual_precedents)
            state['features'][f'[REDACTED_BY_SCRIPT]'] = actual_precedents['[REDACTED_BY_SCRIPT]'].sum()

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        # Ensure schema consistency on failure
        for r in RADII_METERS:
            r_km = r // 1000
            for prefix in ['der_sub1mw', 'der_over1mw']:
                state['features'][f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
                state['features'][f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
                state['features'][f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
                state['features'][f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            state['features'][f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            state['features'][f'[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL

    return state