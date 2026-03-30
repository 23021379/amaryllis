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
L5_SOLAR_FARMS_PATH = r"[REDACTED_BY_SCRIPT]"
# Mandated Input: This MUST be the LPA-level energy forecast data, not the national technology profiles.
NG_DFES_LPA_FORECAST_CSV_PATH = r"[REDACTED_BY_SCRIPT]"
LPA_BOUNDARIES_PATH = r"[REDACTED_BY_SCRIPT]"
L1_DFES_LOOKUP_ARTIFACT_PATH = r"[REDACTED_BY_SCRIPT]"
L6_OUTPUT_PATH = r"[REDACTED_BY_SCRIPT]"

# Architectural Hyperparameters
TARGET_CRS = "EPSG:27700"
TARGET_SCENARIO = 'Holistic Transition'
HORIZON_YEARS = 5
NULL_SENTINEL = -1
EPSILON = 1e-6 # Mandated for numerical stability

def reconcile_authority_name(name):
    """
    Applies the mandated, aggressive name reconciliation protocol to create a
    standardized join key from disparate LPA and LAD naming conventions.
    """
    if not isinstance(name, str):
        return None
    
    # Convert to lowercase and remove common suffixes and punctuation
    name = name.lower()
    suffixes_to_remove = [' lpa', ' council', ' unitary authority', ' borough', ' district']
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            
    # Final alphanumeric sanitization
    return ''.join(filter(str.isalnum, name))


def create_l1_dfes_lookup_artifact():
    """
    Phase I (Revised): Decontamination & Reconciliation Key Generation.
    Ingests and processes the raw NGED DFES data to create a clean, non-spatial
    lookup table with a reconciled join key.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # 1. Ingest NGED DFES Data with Schema Enforcement
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df_ng = pd.read_csv(NG_DFES_LPA_FORECAST_CSV_PATH)
    required_cols = ['local_authority', 'year', 'scenario', 'energy_type', 'TWh_total', '[REDACTED_BY_SCRIPT]']
    if not all(col in df_ng.columns for col in required_cols):
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # 2. Mandated Filtering and Type Coercion
    df_ng = df_ng[df_ng['scenario'] == TARGET_SCENARIO].copy()
    logging.info(f"[REDACTED_BY_SCRIPT]'{TARGET_SCENARIO}'[REDACTED_BY_SCRIPT]")
    for col in ['TWh_total', '[REDACTED_BY_SCRIPT]']:
        df_ng[col] = pd.to_numeric(df_ng[col], errors='coerce')
    df_ng.dropna(subset=['TWh_total', '[REDACTED_BY_SCRIPT]'], inplace=True)
    df_ng['year'] = df_ng['year'].astype(int)

    # 3. Mandated Data Pivot
    logging.info("[REDACTED_BY_SCRIPT]")
    df_pivot = df_ng.pivot_table(
        index=['local_authority', 'year'],
        columns='energy_type',
        values='TWh_total'
    ).reset_index()
    df_pivot.columns.name = None
    df_pivot.rename(columns={
        'Generation': 'generation_twh',
        'Demand': 'demand_twh',
        'Storage': 'storage_twh'
    }, inplace=True)

    # Merge back the renewables percentage (it's constant across energy_type for a given year/LPA)
    df_renewables = df_ng[['local_authority', 'year', '[REDACTED_BY_SCRIPT]']].drop_duplicates()
    df_analysis = pd.merge(df_pivot, df_renewables, on=['local_authority', 'year'])
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # 4. NEW: Apply Reconciliation Protocol to create the join key
    logging.info("[REDACTED_BY_SCRIPT]")
    df_analysis['reconciled_key'] = df_analysis['local_authority'].apply(reconcile_authority_name)
    
    # 5. Persist Non-Spatial Artifact
    # Note: We are now saving a CSV, not a GPKG, as there is no geometry.
    df_analysis.to_csv(L1_DFES_LOOKUP_ARTIFACT_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return df_analysis

def calculate_lpa_context_features(solar_farm_row, dfes_lookup):
    """
    Phase II: The Temporal Guard & Feature Synthesis.
    For a single solar farm, this function executes the fault-tolerant temporal
    search and synthesizes the three strategic context features.
    """
    submission_year = solar_farm_row['submission_year']
    reconciled_key = solar_farm_row['reconciled_key'] # Use the reconciled key
    target_future_year = submission_year + HORIZON_YEARS

    if pd.isna(reconciled_key):
        # Handle cases where the solar farm had no matching LPA geometry
        return pd.Series({
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL,
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL,
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL
        })
        
    try:
        lpa_timeseries = dfes_lookup.loc[reconciled_key]
    except KeyError:
        # This LPA had no data in the DFES file at all.
        return pd.Series({
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL,
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL,
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL
        })

    # Step B & C: Fault-Tolerant Temporal Search
    present_data = lpa_timeseries[lpa_timeseries.index <= submission_year]
    if present_data.empty:
        present_row = None
    else:
        present_row = present_data.loc[present_data.index.max()]

    future_data = lpa_timeseries[lpa_timeseries.index >= target_future_year]
    if future_data.empty:
        future_row = None
    else:
        future_row = future_data.loc[future_data.index.min()]

    # Step D: Handle Catastrophic Data Gaps
    if present_row is None or future_row is None:
        return pd.Series({
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL,
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL,
            '[REDACTED_BY_SCRIPT]': NULL_SENTINEL
        })

    # Feature Synthesis Mandate
    gen_growth = future_row['generation_twh'] - present_row['generation_twh']
    dem_growth = future_row['demand_twh'] - present_row['demand_twh']

    gen_growth_pct = (gen_growth / (present_row['generation_twh'] + EPSILON)) * 100
    gen_dem_ratio = gen_growth / (dem_growth + EPSILON)
    renewables_target = future_row['[REDACTED_BY_SCRIPT]']

    return pd.Series({
        '[REDACTED_BY_SCRIPT]': gen_growth_pct,
        '[REDACTED_BY_SCRIPT]': gen_dem_ratio,
        '[REDACTED_BY_SCRIPT]': renewables_target
    })

def main():
    """
    Orchestrates the full pipeline as per Directive AD-GRID-15 (Rev 1.1).
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    
    # 1. Load Solar Farm Data and Establish Geospatial State
    logging.info(f"[REDACTED_BY_SCRIPT]")
    try:
        df_solar = pd.read_csv(L5_SOLAR_FARMS_PATH)
        # Mandated Geospatial State Construction from CSV
        gdf_solar = gpd.GeoDataFrame(
            df_solar,
            geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing),
            crs=TARGET_CRS
        )
        logging.info("[REDACTED_BY_SCRIPT]")
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)
    except KeyError:
        logging.error(f"[REDACTED_BY_SCRIPT]'easting' or 'northing'[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # 2. Load LPA Geometries and Assign to Solar Farms
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf_lpa = gpd.read_file(LPA_BOUNDARIES_PATH).to_crs(TARGET_CRS)
    logging.info("[REDACTED_BY_SCRIPT]")
    # Using LPA23NM as per the provided file structure
    gdf_solar_with_lpa = gpd.sjoin(gdf_solar, gdf_lpa[['LPA23NM', 'geometry']], how='left', predicate='within')

    # MANDATED DE-DUPLICATION GUARD RAIL: Resolve boundary-case duplicates.
    if gdf_solar_with_lpa.index.has_duplicates:
        duplicate_count = gdf_solar_with_lpa.index.duplicated().sum()
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf_solar_with_lpa = gdf_solar_with_lpa[~gdf_solar_with_lpa.index.duplicated(keep='first')]

    # 3. Create the Reconciled Join Key on the Solar Farm Data
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_solar_with_lpa['reconciled_key'] = gdf_solar_with_lpa['LPA23NM'].apply(reconcile_authority_name)

    # 4. Prepare DFES lookup table for high-performance access
    dfes_lookup_table = create_l1_dfes_lookup_artifact()
    logging.info("[REDACTED_BY_SCRIPT]")
    dfes_lookup = dfes_lookup_table.set_index(['reconciled_key', 'year'])

    # 5. Feature Integration via .apply()
    tqdm.pandas(desc="[REDACTED_BY_SCRIPT]")
    df_new_features = gdf_solar_with_lpa.progress_apply(
        calculate_lpa_context_features, axis=1, dfes_lookup=dfes_lookup
    )

    # 5. Final Integration and Persistence
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_l6 = gdf_solar_with_lpa.join(df_new_features)
    
    # Clean up join artifacts and fill nulls
    if 'index_right' in gdf_l6.columns:
        gdf_l6.drop(columns=['index_right'], inplace=True)
    
    # Isolate non-geometry columns for safe filling to prevent corruption
    geometry_col_name = gdf_l6.geometry.name
    cols_to_fill = [col for col in gdf_l6.columns if col != geometry_col_name]
    
    # Apply fillna with surgical precision
    gdf_l6[cols_to_fill] = gdf_l6[cols_to_fill].fillna(NULL_SENTINEL)

    df_l6 = pd.DataFrame(gdf_l6.drop(columns='geometry', errors='ignore'))
    df_l6.drop(columns=['LPA23NM', 'reconciled_key'], inplace=True, errors='ignore')
    df_l6.to_csv(L6_OUTPUT_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()
