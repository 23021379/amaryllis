import pandas as pd
import geopandas as gpd
import numpy as np
import re

# --- ARCHITECTURAL CONSTANTS ---
TARGET_CRS = "EPSG:27700"

def _normalize_columns(df):
    """[REDACTED_BY_SCRIPT]"""
    df.columns = [col.lower().strip() for col in df.columns]
    return df

def forge_l1_non_spatial_artifact(config):
    """
    Implements Phase 1 of AD-SOC-01: Decontamination & Unification.
    Conquers hostile CSV sources to forge a single, clean, non-spatial master table.
    This mitigates data contamination and schema mismatch risks at the source.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # 1. Establish Backbone (AHAH)
    df_master = pd.read_csv(config['ahah_path'])
    df_master = _normalize_columns(df_master)
    df_master.set_index('lsoa21cd', inplace=True)

    # 2. Ingest & Join OAC Data
    df_oac = pd.read_csv(config['oac_path'])
    df_oac = _normalize_columns(df_oac)
    # MANDATE: Proceeding with join on LSOA11CD, as per directive. High data loss here would
    # trigger an architectural review for an LSOA11-LSOA21 crosswalk.
    initial_lsoa_count = len(df_master)
    df_master = df_master.join(df_oac.set_index('lsoa11_cd'), how='left')
    
    # ARCHITECTURAL LOGGING: Quantify data loss from LSOA11->LSOA21 direct join.
    oac_na_count = df_master['grp_label'].isna().sum()
    if oac_na_count > 0:
        print(f"[REDACTED_BY_SCRIPT]'lsoa11_cd'[REDACTED_BY_SCRIPT]")

    # 3. Ingest, Parse & Join Council Tax Data
    df_ct = pd.read_csv(config['council_tax_path'], engine='python', skip_blank_lines=True)
    df_ct = _normalize_columns(df_ct)
    # MANDATE: Parse LSOA code from malformed source column to prevent data corruption.
    df_ct['lsoa21cd'] = df_ct['[REDACTED_BY_SCRIPT]'].str.extract(r'(E\d{8}|W\d{8})')
    df_ct = df_ct.dropna(subset=['lsoa21cd'])
    # MANDATE: Sanitize column headers to a predictable, prefixed format.
    df_ct = df_ct.rename(columns={
        'total': 'ct_total', 'band a': 'ct_band_a', 'band b': 'ct_band_b',
        'band c': 'ct_band_c', 'band d': 'ct_band_d', 'band e': 'ct_band_e',
        'band f': 'ct_band_f', 'band g': 'ct_band_g'
    })
    ct_cols = ['lsoa21cd'] + [col for col in df_ct.columns if col.startswith('ct_')]
    df_master = df_master.join(df_ct[ct_cols].set_index('lsoa21cd'), how='left')

    # 4. Ingest & Join NDVI/EVI Data
    df_ndvi = pd.read_csv(config['ndvi_path'])
    df_ndvi = _normalize_columns(df_ndvi)
    df_master = df_master.join(df_ndvi.set_index('lsoa21cd'), how='left')

    # 5. Ingest & Join RUC Data
    df_ruc = pd.read_csv(config['ruc_path'])
    df_ruc = _normalize_columns(df_ruc)
    df_master = df_master.join(df_ruc.set_index('lsoa21cd'), how='left')

    # 6. Final Audit & Type Casting
    numeric_cols = [col for col in df_master.columns if df_master[col].dtype in ['int64', 'float64']]
    for col in numeric_cols:
        df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

    df_master.to_csv(config['l1_output_path'])
    print(f"[REDACTED_BY_SCRIPT]'l1_output_path']}")
    return df_master


def forge_l2_geospatial_artifact(config):
    """
    Implements Phase 2 of AD-SOC-01: Geospatial Fusion.
    Bridges the non-spatial L1 master table into the physical world, creating the
    authoritative, spatially-aware source of truth for all LSOA intelligence.
    """
    print("[REDACTED_BY_SCRIPT]")
    df_l1 = pd.read_csv(config['l1_output_path'])
    gdf_lsoa_boundaries = gpd.read_file(config['[REDACTED_BY_SCRIPT]'])

    # MANDATE: Uncompromising CRS Unification. The first and most critical gate.
    if gdf_lsoa_boundaries.crs.to_string() != TARGET_CRS:
        gdf_lsoa_boundaries = gdf_lsoa_boundaries.to_crs(TARGET_CRS)

    # Join non-spatial data to LSOA geometries
    gdf_l2_master = gdf_lsoa_boundaries.merge(df_l1, left_on='LSOA21CD', right_on='lsoa21cd', how='left')
    # MANDATE: Prevent field name collision by dropping the redundant join key.
    gdf_l2_master = gdf_l2_master.drop(columns=['lsoa21cd', 'lsoa21nm', 'lsoa21nmw'])
    
    gdf_l2_master.to_file(config['l2_output_path'], driver='GPKG', layer='lsoa_socio_economic')
    print(f"[REDACTED_BY_SCRIPT]'l2_output_path']}")
    return gdf_l2_master

def add_site_level_features(gdf_solar, gdf_l2_master):
    """
    Implements Phase 3 of AD-SOC-01: Site-Level Linkage.
    Executes the first half of the Granularity Mandate by linking each solar farm
    to its specific host LSOA.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # MANDATE: Ensure CRS integrity before any spatial operation.
    if gdf_solar.crs.to_string() != TARGET_CRS:
        gdf_solar = gdf_solar.to_crs(TARGET_CRS)

    gdf_solar_enriched = gpd.sjoin(gdf_solar, gdf_l2_master, how='left', predicate='intersects')
    
    # MANDATE: Enforce a one-to-one join relationship to prevent record duplication from boundary overlaps.
    initial_rows = len(gdf_solar_enriched)
    # The primary key for solar sites is 'application_reference'.
    gdf_solar_enriched.drop_duplicates(subset=['application_reference'], keep='first', inplace=True)
    if len(gdf_solar_enriched) < initial_rows:
        print(f"[REDACTED_BY_SCRIPT]")

    # MANDATE: Purge the metadata contaminant column immediately after the join.
    gdf_solar_enriched.drop(columns=['index_right'], inplace=True, errors='ignore')

    # MANDATE: Enforce namespace discipline to prevent data collision.
    lsoa_cols = gdf_l2_master.columns.drop('geometry')
    rename_dict = {col: f'[REDACTED_BY_SCRIPT]' for col in lsoa_cols}
    gdf_solar_enriched.rename(columns=rename_dict, inplace=True)
    
    print("[REDACTED_BY_SCRIPT]")
    return gdf_solar_enriched


def add_lpa_level_features(gdf_solar_enriched, gdf_l2_master, gdf_lpa_boundaries):
    """
    Implements the definitive blueprint from AD-SOC-01.4.
    Synthesizes the complete L6 LPA profile, including all mandated indices,
    cohesion metrics (stddev), and political texture (pct distributions).
    """
    print("[REDACTED_BY_SCRIPT]")

    # --- PREREQUISITES ---
    if gdf_lpa_boundaries.crs.to_string() != TARGET_CRS:
        gdf_lpa_boundaries = gdf_lpa_boundaries.to_crs(TARGET_CRS)
    lpa_id_col = _find_lpa_identifier_column(gdf_lpa_boundaries)

    # --- BLUEPRINT STEP 1: FOUNDATION - ONE-HOT ENCODE SOURCE DATA ---
    # Create a clean copy for this complex operation to avoid side-effects.
    gdf_lsoa_processed = gdf_l2_master.copy()
    
    # Mandated one-hot encoding of categorical data *before* aggregation.
    # MANDATE: Explicitly cast to integer type for architectural consistency.
    oac_dummies = pd.get_dummies(gdf_lsoa_processed['grp_label'], prefix='oac_pct', dtype=int)
    ruc_dummies = pd.get_dummies(gdf_lsoa_processed['ruc21nm'], prefix='ruc_pct', dtype=int)
    gdf_lsoa_processed = gdf_lsoa_processed.join([oac_dummies, ruc_dummies])

    # --- BLUEPRINT STEP 2: LINKAGE - ASSIGN LPA ID TO EVERY LSOA ---
    gdf_lsoa_with_lpa = gpd.sjoin(gdf_lsoa_processed, gdf_lpa_boundaries[[lpa_id_col, 'geometry']], how='inner', predicate='intersects')

    # MANDATE: Enforce a one-LSOA-to-one-LPA relationship to prevent double-counting in aggregations.
    initial_lsoa_joins = len(gdf_lsoa_with_lpa)
    # 'LSOA21CD' is the unique identifier for LSOAs in gdf_lsoa_processed.
    gdf_lsoa_with_lpa.drop_duplicates(subset=['LSOA21CD'], keep='first', inplace=True)
    if len(gdf_lsoa_with_lpa) < initial_lsoa_joins:
        print(f"[REDACTED_BY_SCRIPT]")

    # --- BLUEPRINT STEP 3: AGGREGATION - THE CORE CALCULATION ---
    # Pre-calculate all strategic indices at the LSOA level before aggregation.
    gdf_lsoa_with_lpa['property_value_idx'] = gdf_lsoa_with_lpa.apply(lambda row: _calculate_property_value_idx(row, ''), axis=1)
    gdf_lsoa_with_lpa['[REDACTED_BY_SCRIPT]'] = (gdf_lsoa_with_lpa['ah4no2_pct'] + gdf_lsoa_with_lpa['ah4so2_pct'] + gdf_lsoa_with_lpa['ah4pm10_pct']) - gdf_lsoa_with_lpa['ah4g_pct']
    gdf_lsoa_with_lpa['[REDACTED_BY_SCRIPT]'] = (gdf_lsoa_with_lpa['ah4ffood_pct'] + gdf_lsoa_with_lpa['ah4pubs_pct'] + gdf_lsoa_with_lpa['ah4gamb_pct'])
    ruc_is_rural_flag = gdf_lsoa_with_lpa['ruc21nm'].str.contains('Rural', na=False).astype(int)
    gdf_lsoa_with_lpa['[REDACTED_BY_SCRIPT]'] = (gdf_lsoa_with_lpa['ndvi_mean'] * (1 - gdf_lsoa_with_lpa['ndvi_std'])) * ruc_is_rural_flag

    # Define the complete aggregation specification for all mandated features.
    index_cols = ['property_value_idx', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    pct_cols = list(oac_dummies.columns) + list(ruc_dummies.columns)
    
    agg_spec = {col: ['mean', 'std'] for col in index_cols}
    for col in pct_cols:
        agg_spec[col] = 'mean' # The mean of a dummy IS the percentage.

    df_lpa_agg = gdf_lsoa_with_lpa.groupby(lpa_id_col).agg(agg_spec)
    
    # Flatten the multi-level column index and apply mandated naming conventions.
    df_lpa_agg.columns = ['_'.join(col).strip() for col in df_lpa_agg.columns.values]
    df_lpa_agg = df_lpa_agg.rename(columns=lambda c: c.replace('_mean', ''))
    df_lpa_agg = df_lpa_agg.rename(columns=lambda c: c.replace('_std', '_stddev'))

    # --- BLUEPRINT STEP 4: FINAL MERGE ---
    # Ensure the solar data has the correct LPA identifier.
    if lpa_id_col in gdf_solar_enriched.columns:
        gdf_solar_enriched = gdf_solar_enriched.drop(columns=[lpa_id_col])
    gdf_solar_with_lpa_id = gpd.sjoin(gdf_solar_enriched, gdf_lpa_boundaries[[lpa_id_col, 'geometry']], how='left', predicate='intersects')
    gdf_solar_with_lpa_id.drop(columns=['index_right'], inplace=True, errors='ignore')

    # Merge the full strategic context back to the solar data.
    gdf_solar_final = gdf_solar_with_lpa_id.merge(df_lpa_agg, on=lpa_id_col, how='left')

    # Enforce the final, non-negotiable namespace discipline.
    rename_dict = {col: f'lpa_lsoa_agg_{col}' for col in df_lpa_agg.columns}
    gdf_solar_final.rename(columns=rename_dict, inplace=True)

    print("[REDACTED_BY_SCRIPT]")
    return gdf_solar_final



def _calculate_property_value_idx(row, prefix):
    """[REDACTED_BY_SCRIPT]"""
    bands = { 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7 }
    numerator = 0
    denominator = row.get(f'{prefix}ct_total', 0)
    if denominator == 0: return np.nan
    
    for band, weight in bands.items():
        count = row.get(f'[REDACTED_BY_SCRIPT]', 0)
        numerator += weight * count
    return numerator / denominator

def create_strategic_features(df):
    """
    Implements Phase 5 of AD-SOC-01.1: Remediated & Expanded Hypothesis Engine.
    Synthesizes V1 and V2 strategic features and corrects all identified flaws.
    """
    print("[REDACTED_BY_SCRIPT]")

    # V1 Indices (Site Level) - LPA level indices are now created in Phase 4
    df['[REDACTED_BY_SCRIPT]'] = df.apply(lambda row: _calculate_property_value_idx(row, 'site_lsoa_'), axis=1)
    df['[REDACTED_BY_SCRIPT]'] = (df['[REDACTED_BY_SCRIPT]'] + df['[REDACTED_BY_SCRIPT]'] + df['[REDACTED_BY_SCRIPT]']) - df['site_lsoa_ah4g_pct']
    df['[REDACTED_BY_SCRIPT]'] = (df['[REDACTED_BY_SCRIPT]'] + df['[REDACTED_BY_SCRIPT]'] + df['[REDACTED_BY_SCRIPT]'])
    # The `site_lsoa_ruc_is_rural` flag is now created ephemerally for the index calculation and not persisted.
    ruc_is_rural_flag = df['site_lsoa_ruc21nm'].str.contains('Rural', na=False).astype(int)
    df['[REDACTED_BY_SCRIPT]'] = (df['site_lsoa_ndvi_mean'] * (1 - df['site_lsoa_ndvi_std'])) * ruc_is_rural_flag

    # REMEDIATION: Correctly implement OAC & RUC one-hot encoding with namespace discipline
    # ERADICATION: `dummy_na=False` removes the `oac_nan` contamination vector.
    # MANDATE: Explicitly cast to integer type to ensure a fully numeric output.
    df = pd.get_dummies(df, columns=['site_lsoa_grp_label'], prefix='site_lsoa_oac', dummy_na=False, dtype=int)
    df = pd.get_dummies(df, columns=['site_lsoa_ruc21nm'], prefix='site_lsoa_ruc', dummy_na=False, dtype=int)

    # V1 SICs (Surgical Interactions)
    df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * df['[REDACTED_BY_SCRIPT]']
    df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] / (df['[REDACTED_BY_SCRIPT]'] + 0.01)
    
    oac_offline_col = '[REDACTED_BY_SCRIPT]'
    if oac_offline_col in df.columns:
        rural_mapping = {'[REDACTED_BY_SCRIPT]': 4, '[REDACTED_BY_SCRIPT]': 3, '[REDACTED_BY_SCRIPT]': 5, '[REDACTED_BY_SCRIPT]': 2, '[REDACTED_BY_SCRIPT]': 3}
        # Note: We use the original ruc21nm column for mapping before it's dropped by get_dummies
        df['site_lsoa_ruc_rural_score'] = df['[REDACTED_BY_SCRIPT]'].map(rural_mapping).fillna(0) if '[REDACTED_BY_SCRIPT]' in df.columns else 0
        df['[REDACTED_BY_SCRIPT]'] = df[oac_offline_col] * df['site_lsoa_ruc_rural_score']
    else:
        df['[REDACTED_BY_SCRIPT]'] = 0

    # IMPLEMENT V2.0: "Site-LPA Delta" Indices
    # Add stabilization factor to denominator to prevent division by zero/infinity.
    df['delta_property_value'] = df['[REDACTED_BY_SCRIPT]'] / (df['[REDACTED_BY_SCRIPT]'] + 0.01)
    df['delta_env_health_disadvantage'] = df['[REDACTED_BY_SCRIPT]'] / (df['[REDACTED_BY_SCRIPT]'] + 0.01)

    # IMPLEMENT V2.0: Activate SIC-06 "Greenbelt Pressure Cooker"
    df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] / (df['site_lsoa_ah4g_pct'] + 0.01)

    print("[REDACTED_BY_SCRIPT]")
    return df

def _find_lpa_identifier_column(gdf):
    """
    MANDATE: Semantic, Not Syntactic, Logic.
    Dynamically finds the LPA identifier column in a GeoDataFrame by searching for
    common patterns. This prevents catastrophic KeyError failures due to minor
    schema variations in source files.
    """
    patterns_to_check = [r'LAD\d*CD', r'LPA\d*CD', r'LAD\d*NM']
    for pattern in patterns_to_check:
        for col in gdf.columns:
            if re.fullmatch(pattern, col, re.IGNORECASE):
                print(f"[REDACTED_BY_SCRIPT]'{col}'")
                return col
    # Fail-safe: If no column is found, raise a clear, actionable error.
    raise ValueError(
        "[REDACTED_BY_SCRIPT]'LAD23CD') "
        f"[REDACTED_BY_SCRIPT]"
    )

import os

def main():
    """
    Main orchestration function for Directive AD-SOC-01.
    Executes the full 5-phase pipeline for socio-economic and landscape synthesis.
    """
    print("[REDACTED_BY_SCRIPT]")

    # --- CONFIGURATION ---
    # MANDATE: All file paths are managed here for maintainability.
    # Users should update these paths to match their local environment.
    BASE_DIR = r"[REDACTED_BY_SCRIPT]"
    BOUNDARY_PATH = r"[REDACTED_BY_SCRIPT]"
    config = {
        # --- INPUTS: LSOA-level raw data ---
        'ahah_path': os.path.join(BASE_DIR, 'AHAH_V4.csv'),
        'oac_path': os.path.join(BASE_DIR, 'iuc2018.csv'),
        'council_tax_path': os.path.join(BASE_DIR, '[REDACTED_BY_SCRIPT]'),
        'ndvi_path': os.path.join(BASE_DIR, 'LSOA veg.csv'),
        'ruc_path': os.path.join(BASE_DIR, '[REDACTED_BY_SCRIPT]'),

        # --- INPUTS: Geospatial & Project Data ---
        '[REDACTED_BY_SCRIPT]': os.path.join(BOUNDARY_PATH, '[REDACTED_BY_SCRIPT]'),
        'lpa_boundaries_path': os.path.join(BOUNDARY_PATH, '[REDACTED_BY_SCRIPT]'),
        '[REDACTED_BY_SCRIPT]': r"[REDACTED_BY_SCRIPT]", 

        # --- OUTPUTS: Artifacts ---
        'l1_output_path': os.path.join(BASE_DIR, '[REDACTED_BY_SCRIPT]'),
        'l2_output_path': os.path.join(BASE_DIR, '[REDACTED_BY_SCRIPT]'),
        'final_output_path': r'[REDACTED_BY_SCRIPT]'
    }

    # --- EXECUTION ---

    # Phase 1 & 2: Forge master artifacts. Skip if they already exist for efficiency.
    if not os.path.exists(config['l1_output_path']):
        forge_l1_non_spatial_artifact(config)
    else:
        print("[REDACTED_BY_SCRIPT]")

    if not os.path.exists(config['l2_output_path']):
        forge_l2_geospatial_artifact(config)
    else:
        print("[REDACTED_BY_SCRIPT]")

    # Load core datasets for the main enrichment pipeline
    print("[REDACTED_BY_SCRIPT]")
    df_solar = pd.read_csv(config['[REDACTED_BY_SCRIPT]'])
    gdf_solar = gpd.GeoDataFrame(
        df_solar, 
        geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), 
        crs=TARGET_CRS
    )
    
    # MANDATE: Defensive Key Synthesis. The pipeline must not trust the incoming schema.
    # If the primary key is missing, synthesize it to fulfill the data contract.
    if 'application_reference' not in gdf_solar.columns:
        print("    WARNING: 'application_reference'[REDACTED_BY_SCRIPT]")
        gdf_solar['application_reference'] = gdf_solar.index

    gdf_l2_master = gpd.read_file(config['l2_output_path'], layer='lsoa_socio_economic')
    gdf_lpa_boundaries = gpd.read_file(config['lpa_boundaries_path'])

    # Phase 3: Add Site-Level Features
    gdf_solar_enriched = add_site_level_features(gdf_solar, gdf_l2_master)

    # Phase 4: Add LPA-Level Aggregate Features
    gdf_solar_enriched = add_lpa_level_features(gdf_solar_enriched, gdf_l2_master, gdf_lpa_boundaries)

    # Phase 5: Create Strategic Indices and SICs
    df_final = create_strategic_features(gdf_solar_enriched)

    # Finalization: Implement Re-Architected Intelligence Distillation Protocol for L6 Artifact
    print("[REDACTED_BY_SCRIPT]")

    # MANDATE: Define the patterns for all features to be preserved.
    initial_cols = df_solar.columns.tolist()
    
    # Define the patterns for all classes of synthesized intelligence.
    # This is architecturally superior to a long, brittle list of conditions.
    patterns_to_keep = [
        '_idx',                 # All strategic indices
        'sic_',                 # All Surgical Interactions
        'delta_',               # All Site-LPA Delta features
        'site_lsoa_oac_',       # Site-level OAC one-hot features
        'site_lsoa_ruc_',       # Site-level RUC one-hot features
        'lpa_lsoa_agg_pct_',    # LPA-level Political Texture features
        'lpa_lsoa_agg_stddev_'  # LPA-level Cohesion features
    ]

    synthesized_cols = [
        col for col in df_final.columns 
        if any(pattern in col for pattern in patterns_to_keep)
    ]
    
    # The final, L6-compliant set of columns.
    cols_to_keep = initial_cols + synthesized_cols
    
    # Create the final dataframe, ensuring no duplicates and preserving order.
    df_purified = df_final[list(dict.fromkeys(cols_to_keep))]

    # Persist the purified L6 artifact.
    df_purified.to_csv(config['final_output_path'], index=False)

    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]'final_output_path']}")


if __name__ == "__main__":
    main()