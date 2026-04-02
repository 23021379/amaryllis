# amaryllis_pipeline.py
# Monolithic execution script for Project Amaryllis data processing.
# ARCHITECTURAL NOTE: This script consolidates multiple sequential steps into a single file.
# For production, refactoring into a modular workflow (e.g., using Prefect, Airflow, or Dagster) is strongly recommended.

# =================================================================================================
# I. GLOBAL DEPENDENCIES
# =================================================================================================
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
import os
import re
import json
import hashlib
from functools import reduce
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from shapely.geometry import Point, LineString, shape
from tqdm import tqdm

# =================================================================================================
# II. CENTRALIZED CONFIGURATION & LOGGING
# =================================================================================================

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Project-wide Architectural Parameters ---
TARGET_CRS = "EPSG:27700"
NULL_SENTINEL = -1
AMARYLLIS_SALT = "[REDACTED_BY_SCRIPT]"

# --- Raw Data Paths (Inputs) ---
# NOTE: These paths must be correctly set for the environment.
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
LPA_DATA_DIR = os.path.join(BASE_DATA_DIR, 'LPA')
GEOSPATIAL_DATA_DIR = os.path.join(BASE_DATA_DIR, 'geospatial', 'boundaries')
UKPN_DATA_DIR = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')

# Step 1
SOURCE_APPLICATIONS_CSV = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
# Step 3
PS1_PATH = os.path.join(LPA_DATA_DIR, '[REDACTED_BY_SCRIPT]')
PS2_PATH = os.path.join(LPA_DATA_DIR, '[REDACTED_BY_SCRIPT]')
CPS1_PATH = os.path.join(LPA_DATA_DIR, '[REDACTED_BY_SCRIPT]')
CPS2_PATH = os.path.join(LPA_DATA_DIR, '[REDACTED_BY_SCRIPT]')
# Step 6.01
DFES_GEOJSON_PATH = os.path.join(UKPN_DATA_DIR, 'DFES NSHR.geojson')
# Step 6.02
FAULT_LEVEL_CSV_PATH = os.path.join(UKPN_DATA_DIR, '[REDACTED_BY_SCRIPT]')
# Step 6.04
ECR_SUB_1MW_PATH = os.path.join(UKPN_DATA_DIR, '[REDACTED_BY_SCRIPT]')
ECR_OVER_1MW_PATH = os.path.join(UKPN_DATA_DIR, '[REDACTED_BY_SCRIPT]')
# Step 6.04_transformers
TRANSFORMERS_GRID_SITE_PATH = os.path.join(UKPN_DATA_DIR, '[REDACTED_BY_SCRIPT]')
# Step 6.05
IDNO_RAW_PATH = os.path.join(UKPN_DATA_DIR, '[REDACTED_BY_SCRIPT]')
LPA_BOUNDARIES_PATH = os.path.join(GEOSPATIAL_DATA_DIR, '[REDACTED_BY_SCRIPT]')
# Step 6.06
LTDS_SOURCE_FILES = {os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]"): "2018-10-13"}
# Step 6.08
LCT_SECONDARY_SITES_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]' Secondary Sites.geojson")
# Step 6.08_2
LCT_PRIMARY_AGGREGATE_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.09
DNOA_LV_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.10
PQ_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.11
TRANSFORMERS_PRIMARY_SITE_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.12
SUBSTATION_BIO_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.13
OHL_33KV_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.14
OHL_33KV_POINTS_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.15
OHL_132KV_LINES_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
OHL_132KV_TOWERS_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.16
HV_LINES_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
HV_POLES_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.17
LV_LINES_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
LV_POLES_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.18
SERVICE_AREA_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.19
SECONDARY_SUB_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")
# Step 6.20
SEC_SUB_AREA_INPUT_PATH = os.path.join(UKPN_DATA_DIR, "[REDACTED_BY_SCRIPT]")


# --- Artifact Paths (Intermediate and Final Outputs) ---
# NOTE: These define the data dependency chain.
L1_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L2_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L3_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L4_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L5_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L6_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L7_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
# L8 is skipped in the user-provided files.
L9_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L10_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L11_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L12_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L13_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L14_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L15_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L16_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L17_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L18_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L19_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L20_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L21_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L22_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L23_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L24_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L25_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
L26_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
PIPELINE_FINAL_ARTIFACT = L26_ARTIFACT # Alias for the final output

# Supporting Artifacts
CATEGORICAL_MAPPINGS_JSON = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
SUBSTATION_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')
IDNO_L1_UNIFIED_PATH = os.path.join(BASE_DATA_DIR, '[REDACTED_BY_SCRIPT]')


# =================================================================================================
# III. CONSOLIDATED HELPER FUNCTIONS
# =================================================================================================

def clean_lpa_name(series: pd.Series) -> pd.Series:
    """[REDACTED_BY_SCRIPT]"""
    return series.astype(str).str.lower().str.strip().str.replace(r'\s*\(.*\)\s*', '', regex=True).str.strip()

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    new_cols = []
    for col in df.columns:
        new_col = col.lower().strip()
        new_col = re.sub(r'[^a-z0-9_]+', '_', new_col)
        new_col = new_col.strip('_')
        new_col = re.sub(r'__+', '_', new_col)
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def find_column_by_pattern(df: pd.DataFrame, pattern: str) -> str:
    """[REDACTED_BY_SCRIPT]"""
    matches = []
    for col in df.columns:
        normalized_col = re.sub(r'[^a-z0-9]+', '_', col.lower()).strip('_')
        if re.search(pattern, normalized_col):
            matches.append(col)
    if not matches:
        raise KeyError(f"[REDACTED_BY_SCRIPT]'{pattern}'.")
    if len(matches) > 1:
        raise ValueError(f"[REDACTED_BY_SCRIPT]'{pattern}': {matches}")
    return matches[0]

def safe_division(numerator, denominator):
    """[REDACTED_BY_SCRIPT]"""
    return np.divide(numerator.values, denominator.values, out=np.full_like(numerator.values, np.nan, dtype=float), where=(denominator.values != 0))


# =================================================================================================
# IV. PIPELINE STEP DEFINITIONS
# =================================================================================================

def run_step_1_ingest_repd():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")

    COLUMN_MAPPING = {
        'Planning Authority': 'planning_authority', 'Technology Type': 'technology_type',
        'Storage Type': 'storage_type', '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        'CHP Enabled': 'chp_enabled', '[REDACTED_BY_SCRIPT]': 'ro_banding_roc_mwh',
        'FiT Tariff (p/kWh)': 'fit_tariff_p_kwh', 'CfD Capacity (MW)': 'cfd_capacity_mw',
        'Turbine Capacity': 'turbine_capacity', 'No. of Turbines': 'no_of_turbines',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]', 'X-coordinate': 'easting',
        'Y-coordinate': 'northing', 'Secretary of State Reference': 'secretary_of_state_reference',
        '[REDACTED_BY_SCRIPT]': 'type_of_sos_intervention',
        'Judicial Review': 'judicial_review', 'Offshore Wind Round': 'offshore_wind_round',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]', 'Appeal Lodged': 'appeal_lodged',
        'Appeal Withdrawn': 'appeal_withdrawn', 'Appeal Refused': 'appeal_refused',
        'Appeal Granted': 'appeal_granted', '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]': 'sos_intervened', '[REDACTED_BY_SCRIPT]': 'sos_refusal',
        '[REDACTED_BY_SCRIPT]': 'sos_granted', '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
        'Under Construction': 'under_construction', 'Operational': 'operational',
        'Heat Network Ref': 'heat_network_ref', '[REDACTED_BY_SCRIPT]': 'solar_site_area_sqm'
    }
    DATE_COLS = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'appeal_lodged', 'appeal_withdrawn', 'appeal_refused', 'appeal_granted',
        '[REDACTED_BY_SCRIPT]', 'sos_intervened', 'sos_refusal', 'sos_granted',
        '[REDACTED_BY_SCRIPT]', 'under_construction', 'operational'
    ]
    NUMERIC_COLS = [
        '[REDACTED_BY_SCRIPT]', 'ro_banding_roc_mwh', 'fit_tariff_p_kwh', 'cfd_capacity_mw',
        'turbine_capacity', 'no_of_turbines', '[REDACTED_BY_SCRIPT]', 'easting', 'northing',
        'solar_site_area_sqm'
    ]

    try:
        df = pd.read_csv(SOURCE_APPLICATIONS_CSV, usecols=COLUMN_MAPPING.keys(), dtype=str, encoding='latin-1')
        df = df.rename(columns=COLUMN_MAPPING)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip().str.replace(r'\r?\n', ' ', regex=True)
        
        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        for col in DATE_COLS:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            
        df.to_csv(L1_ARTIFACT, index=False, encoding='utf-8')
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_2_standardise_and_encode():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        df = pd.read_csv(L1_ARTIFACT, low_memory=False)
        date_cols = [col for col in df.columns if 'date' in col or 'submitted' in col or 'granted' in col or 'refused' in col or 'lodged' in col or 'withdrawn' in col or 'expired' in col or 'operational' in col]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        df['[REDACTED_BY_SCRIPT]'] = df['appeal_granted'].fillna(df['[REDACTED_BY_SCRIPT]'])
        df['permission_granted'] = np.where(df['[REDACTED_BY_SCRIPT]'].notna(), 1, 0)
        time_delta = df['[REDACTED_BY_SCRIPT]'] - df['[REDACTED_BY_SCRIPT]']
        df['[REDACTED_BY_SCRIPT]'] = time_delta.dt.days

        submission_date = df['[REDACTED_BY_SCRIPT]']
        df['submission_year'] = submission_date.dt.year
        df['submission_month'] = submission_date.dt.month
        df['submission_day'] = submission_date.dt.day
        df['submission_month_sin'] = np.sin(2 * np.pi * df['submission_month'] / 12)
        df['submission_month_cos'] = np.cos(2 * np.pi * df['submission_month'] / 12)
        
        granted_date = df['[REDACTED_BY_SCRIPT]']
        df['[REDACTED_BY_SCRIPT]'] = granted_date.dt.year

        null_percentages = df.isnull().mean()
        cols_to_drop = null_percentages[null_percentages > 0.5].index.tolist()
        if 'solar_site_area_sqm' in cols_to_drop: cols_to_drop.remove('solar_site_area_sqm')
        if '[REDACTED_BY_SCRIPT]' in cols_to_drop: cols_to_drop.remove('[REDACTED_BY_SCRIPT]')
        df.drop(columns=cols_to_drop, inplace=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if '[REDACTED_BY_SCRIPT]' in numeric_cols: numeric_cols.remove('[REDACTED_BY_SCRIPT]')
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(-1, inplace=True)

        categorical_cols = df.select_dtypes(include=['object']).columns.drop('planning_authority', errors='ignore')
        master_mappings = {}
        for col in categorical_cols:
            df[col] = df[col].fillna('missing').astype('category')
            master_mappings[col] = {cat: code for code, cat in enumerate(df[col].cat.categories)}
            df[col] = df[col].cat.codes
        
        with open(CATEGORICAL_MAPPINGS_JSON, 'w') as f:
            json.dump(master_mappings, f, indent=4)

        df.drop(columns=[col for col in date_cols if col in df.columns], errors='ignore', inplace=True)
        df.drop(columns=['[REDACTED_BY_SCRIPT]'], errors='ignore', inplace=True)
        
        df['planning_authority'] = clean_lpa_name(df['planning_authority'])
        
        df.fillna(-1, inplace=True)

        df.to_csv(L2_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_3_process_lpa_csvs():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    def calculate_trend(group):
        if len(group) < 3: return np.nan
        group = group.dropna(subset=['year', 'workload'])
        if len(group) < 3: return np.nan
        x, y = group['year'], group['workload']
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope

    try:
        # PS1/CPS1 Workload Processing
        df_ps1 = pd.read_csv(PS1_PATH, encoding='latin-1', dtype=str).replace('..', np.nan)
        df_ps1.columns = df_ps1.columns.str.lower().str.replace(' ', '_', regex=False)
        df_ps1['lpanm_clean'] = clean_lpa_name(df_ps1['lpanm'])
        enforcement_cols_ps1 = [c for c in df_ps1.columns if 'enforcement' in c]
        df_ps1_out = df_ps1[['lpanm_clean', 'f_year', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'] + enforcement_cols_ps1].rename(columns={'f_year': 'year', '[REDACTED_BY_SCRIPT]': 'received', '[REDACTED_BY_SCRIPT]': 'withdrawn'})

        df_cps1 = pd.read_csv(CPS1_PATH, encoding='latin-1', dtype=str).replace('..', np.nan)
        df_cps1.columns = df_cps1.columns.str.lower().str.replace(' ', '_', regex=False)
        df_cps1['lpanm_clean'] = clean_lpa_name(df_cps1['lpanm'])
        df_cps1['year'] = df_cps1['quarter'].str.split(' ').str[0]
        df_cps1_out = df_cps1[['lpanm_clean', 'year', '[REDACTED_BY_SCRIPT]']].rename(columns={'[REDACTED_BY_SCRIPT]': 'received'})
        
        df_workload = pd.concat([df_ps1_out, df_cps1_out], ignore_index=True)
        numeric_cols = [c for c in df_workload.columns if c not in ['lpanm_clean', 'year']]
        for col in numeric_cols: df_workload[col] = pd.to_numeric(df_workload[col], errors='coerce')
        df_workload['[REDACTED_BY_SCRIPT]'] = df_workload[[c for c in df_workload.columns if 'enforcement' in c]].sum(axis=1)

        workload_base_agg = df_workload.groupby('lpanm_clean').agg(total_applications_received=('received', 'sum'), total_applications_withdrawn=('withdrawn', 'sum'), [REDACTED_BY_SCRIPT]=('[REDACTED_BY_SCRIPT]', 'sum'))
        workload_base_agg['lpa_withdrawal_rate'] = safe_division(workload_base_agg['[REDACTED_BY_SCRIPT]'], workload_base_agg['[REDACTED_BY_SCRIPT]'])
        
        yearly_workload = df_workload.groupby(['lpanm_clean', 'year'])['received'].sum().reset_index().rename(columns={'received': 'workload'})
        workload_dynamics_agg = yearly_workload.groupby('lpanm_clean')['workload'].agg(lpa_avg_yearly_workload='mean', lpa_workload_volatility=lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan, lpa_total_experience='sum')
        workload_trend = yearly_workload.groupby('lpanm_clean').apply(calculate_trend).rename('lpa_workload_trend')
        
        workload_profile = reduce(lambda left, right: pd.merge(left, right, on='lpanm_clean', how='outer'), [workload_base_agg, workload_dynamics_agg, workload_trend.reset_index()])
        workload_profile['[REDACTED_BY_SCRIPT]'] = safe_division(workload_profile['[REDACTED_BY_SCRIPT]'], workload_profile['lpa_total_experience'])

        # PS2 Decision Data
        df_ps2 = pd.read_csv(PS2_PATH, encoding='latin-1', dtype=str).replace('..', np.nan)
        df_ps2.columns = clean_col_names(df_ps2).columns
        df_ps2['lpanm_clean'] = clean_lpa_name(df_ps2['lpanm'])
        ps2_numeric_cols = [c for c in df_ps2.columns if 'total' in c or 'granted' in c]
        for col in ps2_numeric_cols: df_ps2[col] = pd.to_numeric(df_ps2[col], errors='coerce')
        df_ps2['[REDACTED_BY_SCRIPT]'] = df_ps2.filter(like='total_granted_major').sum(axis=1)
        df_ps2['[REDACTED_BY_SCRIPT]'] = df_ps2.filter(like='[REDACTED_BY_SCRIPT]').sum(axis=1)
        lpa_profile_ps2 = df_ps2.groupby('lpanm_clean').agg([REDACTED_BY_SCRIPT]=('[REDACTED_BY_SCRIPT]', 'sum'), [REDACTED_BY_SCRIPT]=('[REDACTED_BY_SCRIPT]', 'sum'), total_major_in_time=('[REDACTED_BY_SCRIPT]', 'sum'), total_major_decisions_for_compliance=('[REDACTED_BY_SCRIPT]', 'sum'))
        lpa_profile_ps2['lpa_major_commercial_approval_rate'] = safe_division(lpa_profile_ps2['[REDACTED_BY_SCRIPT]'], lpa_profile_ps2['[REDACTED_BY_SCRIPT]'])
        lpa_profile_ps2['[REDACTED_BY_SCRIPT]'] = safe_division(lpa_profile_ps2['total_major_in_time'], lpa_profile_ps2['[REDACTED_BY_SCRIPT]'])
        
        # CPS2 Transactional Data
        df_cps2 = pd.read_csv(CPS2_PATH, encoding='latin-1', dtype=str).replace('..', np.nan)
        df_cps2.columns = clean_col_names(df_cps2).columns
        df_cps2['lpanm_clean'] = clean_lpa_name(df_cps2['lpanm'])
        df_cps2['is_granted'] = df_cps2['granted_or_refused'].map({'Granted': 1, 'Refused': 0})
        df_cps2['date_received'] = pd.to_datetime(df_cps2['date_received'], errors='coerce', dayfirst=True)
        df_cps2['date_dispatched'] = pd.to_datetime(df_cps2['date_dispatched'], errors='coerce', dayfirst=True)
        df_cps2['decision_days'] = (df_cps2['date_dispatched'] - df_cps2['date_received']).dt.days
        df_cps2 = df_cps2[(df_cps2['decision_days'] >= 0) & (df_cps2['is_granted'].notna())]
        lpa_profile_cps2 = df_cps2.groupby('lpanm_clean').agg(lpa_approval_rate_cps2=('is_granted', 'mean'), lpa_avg_decision_days=('decision_days', 'mean'), lpa_decision_speed_variance=('decision_days', 'std'), lpa_p90_decision_days=('decision_days', lambda x: x.quantile(0.9)))
        
        # Merge all profiles
        lpa_master_profile = reduce(lambda left, right: pd.merge(left, right, on='lpanm_clean', how='outer'), [workload_profile, lpa_profile_ps2.reset_index(), lpa_profile_cps2.reset_index()])
        df_main = pd.read_csv(L2_ARTIFACT)
        df_merged = pd.merge(df_main, lpa_master_profile, left_on='planning_authority', right_on='lpanm_clean', how='left')
        df_merged.drop(columns=['lpanm_clean', 'planning_authority'], inplace=True)
        df_merged.fillna(-1, inplace=True)

        df_merged.to_csv(L3_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_4_aggregate_years():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        df_l3 = pd.read_csv(L3_ARTIFACT)
        CUTOFF_YEAR = 2012
        modern_cohort = df_l3[df_l3['submission_year'] >= CUTOFF_YEAR].copy()
        legacy_cohort = df_l3[df_l3['submission_year'] < CUTOFF_YEAR].copy()

        if not legacy_cohort.empty:
            granted_legacy = legacy_cohort[legacy_cohort['permission_granted'] == 1]
            lpa_legacy_duration = granted_legacy.groupby('planning_authority').agg(lpa_legacy_avg_duration=('[REDACTED_BY_SCRIPT]', 'mean')).reset_index()
            lpa_legacy_profile = legacy_cohort.groupby('planning_authority').agg(lpa_legacy_approval_rate=('permission_granted', 'mean'), lpa_legacy_application_count=('permission_granted', 'size')).reset_index()
            lpa_legacy_profile = lpa_legacy_profile.merge(lpa_legacy_duration, on='planning_authority', how='left')
            modern_cohort = modern_cohort.merge(lpa_legacy_profile, on='planning_authority', how='left')

            gdf_modern = gpd.GeoDataFrame(modern_cohort, geometry=gpd.points_from_xy(modern_cohort.easting, modern_cohort.northing), crs=TARGET_CRS)
            gdf_legacy = gpd.GeoDataFrame(legacy_cohort, geometry=gpd.points_from_xy(legacy_cohort.easting, legacy_cohort.northing), crs=TARGET_CRS)
            
            modern_points = np.array(list(zip(gdf_modern.geometry.x, gdf_modern.geometry.y)))
            tree_all = cKDTree(np.array(list(zip(gdf_legacy.geometry.x, gdf_legacy.geometry.y))))
            
            nearby_indices_list = tree_all.query_ball_point(modern_points, r=10000)
            gdf_modern['nearby_legacy_count'] = [len(indices) for indices in nearby_indices_list]
            gdf_modern['[REDACTED_BY_SCRIPT]'] = [gdf_legacy.iloc[indices]['permission_granted'].mean() if len(indices) > 0 else -1 for indices in nearby_indices_list]
            dist_nearest_all, _ = tree_all.query(modern_points, k=1)
            gdf_modern['[REDACTED_BY_SCRIPT]'] = dist_nearest_all / 1000
        else:
            gdf_modern = gpd.GeoDataFrame(modern_cohort, geometry=gpd.points_from_xy(modern_cohort.easting, modern_cohort.northing), crs=TARGET_CRS)
            gdf_modern['lpa_legacy_approval_rate'] = NULL_SENTINEL
            gdf_modern['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            gdf_modern['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            gdf_modern['nearby_legacy_count'] = 0
            gdf_modern['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
            gdf_modern['[REDACTED_BY_SCRIPT]'] = 999
        
        df_l4 = pd.DataFrame(gdf_modern.drop(columns='geometry'))
        df_l4.fillna(NULL_SENTINEL, inplace=True)
        df_l4.to_csv(L4_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_4_5_uid():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        df = pd.read_csv(L4_ARTIFACT)
        
        composite_key_cols = ['easting', 'northing', '[REDACTED_BY_SCRIPT]', 'submission_year', 'submission_month', 'submission_day']
        df.drop_duplicates(subset=composite_key_cols, keep='first', inplace=True)
        
        df_keys = pd.DataFrame(index=df.index)
        df_keys['easting_str'] = df['easting'].map('{:.2f}'.format)
        df_keys['northing_str'] = df['northing'].map('{:.2f}'.format)
        df_keys['capacity_str'] = df['[REDACTED_BY_SCRIPT]'].map('{:.3f}'.format)
        df_keys['year_str'] = df['submission_year'].astype(int).astype(str)
        df_keys['month_str'] = df['submission_month'].astype(int).astype(str)
        df_keys['day_str'] = df['submission_day'].astype(int).astype(str)

        canonical_string = (df_keys['easting_str'] + '|' + df_keys['northing_str'] + '|' + df_keys['capacity_str'] + '|' + df_keys['year_str'] + '|' + df_keys['month_str'] + '|' + df_keys['day_str'])

        def create_hash(value: str) -> str:
            salted_string = AMARYLLIS_SALT + value
            return hashlib.sha256(salted_string.encode('utf-8')).hexdigest()

        df['amaryllis_id'] = canonical_string.apply(create_hash)
        
        if not df['amaryllis_id'].is_unique:
            raise AssertionError("[REDACTED_BY_SCRIPT]")
            
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('amaryllis_id')))
        df = df[cols]
        
        df.to_csv(L4_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_5_aggregate_non_solar():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        df_l4 = pd.read_csv(L4_ARTIFACT)
        K_NEIGHBORS = 10
        SOLAR_TECH_CODE = 21 # from categorical_mappings.json
        MAX_DISTANCE_KM = 999

        # Re-engineer submission date
        month_raw = np.arctan2(df_l4['submission_month_sin'], df_l4['submission_month_cos']) * (12 / (2 * np.pi))
        df_l4['submission_month'] = np.round(month_raw % 12) + 1
        df_l4['[REDACTED_BY_SCRIPT]'] = pd.to_datetime(df_l4['submission_year'].astype(int).astype(str) + '-' + df_l4['submission_month'].astype(int).astype(str) + '-01', errors='coerce')

        source_df = df_l4[df_l4['[REDACTED_BY_SCRIPT]'] >= 0].copy()
        source_df['estimated_decision_date'] = source_df['[REDACTED_BY_SCRIPT]'] + pd.to_timedelta(source_df['[REDACTED_BY_SCRIPT]'], unit='D')
        
        source_gdf = gpd.GeoDataFrame(source_df, geometry=gpd.points_from_xy(source_df.easting, source_df.northing), crs=TARGET_CRS)
        target_gdf = gpd.GeoDataFrame(df_l4, geometry=gpd.points_from_xy(df_l4.easting, df_l4.northing), crs=TARGET_CRS)

        results = []
        for index, target in tqdm(target_gdf.iterrows(), total=target_gdf.shape[0], desc="[REDACTED_BY_SCRIPT]"):
            submission_date = target['[REDACTED_BY_SCRIPT]']
            if pd.isna(submission_date):
                results.append({'original_index': index})
                continue
            
            source_candidates = source_gdf[source_gdf['estimated_decision_date'] < submission_date]
            if len(source_candidates) < K_NEIGHBORS:
                results.append({'original_index': index})
                continue
            
            tree = cKDTree(np.array(list(zip(source_candidates.geometry.x, source_candidates.geometry.y))))
            distances_m, indices = tree.query([target.geometry.x, target.geometry.y], k=K_NEIGHBORS)
            neighbors = source_candidates.iloc[indices]
            
            features = {'original_index': index}
            distances_km = distances_m / 1000.0
            features['knn_avg_distance_km'] = np.mean(distances_km)
            features['knn_approval_rate'] = neighbors['permission_granted'].mean()
            features['knn_count_solar'] = (neighbors['technology_type'] == SOLAR_TECH_CODE).sum()
            solar_distances_km = distances_km[neighbors['technology_type'] == SOLAR_TECH_CODE]
            features['[REDACTED_BY_SCRIPT]'] = np.min(solar_distances_km) if len(solar_distances_km) > 0 else MAX_DISTANCE_KM
            
            results.append(features)

        df_knn_features = pd.DataFrame(results).set_index('original_index')
        df_l5 = target_gdf.merge(df_knn_features, left_index=True, right_index=True, how='left')
        
        df_l5 = pd.DataFrame(df_l5.drop(columns=['geometry', 'submission_month', '[REDACTED_BY_SCRIPT]']))
        df_l5.fillna(NULL_SENTINEL, inplace=True)
        df_l5.to_csv(L5_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_01_integrate_dfes():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        K_NEIGHBORS = 5
        gdf_dfes = gpd.read_file(DFES_GEOJSON_PATH).to_crs(TARGET_CRS)
        
        dfes_filtered = gdf_dfes[gdf_dfes['scenario'] == 'Holistic Transition'].copy()
        substation_profile = dfes_filtered.pivot_table(index=['sitefunctionallocation', 'substation_name', 'voltage_kv', 'geometry'], columns='year', values='headroom_mw').reset_index()
        substation_profile.columns = [f'headroom_mw_{col}' if str(col).isdigit() else col for col in substation_profile.columns]
        substation_profile = gpd.GeoDataFrame(substation_profile, geometry='geometry', crs=TARGET_CRS)
        
        authoritative_substations = substation_profile[['sitefunctionallocation', 'substation_name', 'voltage_kv', 'geometry']].rename(columns={'sitefunctionallocation': 'substation_id', 'substation_name': 'name', 'voltage_kv': 'voltage'})
        authoritative_substations.to_file(SUBSTATION_L1_ARTIFACT, driver='GPKG')

        df_l5 = pd.read_csv(L5_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_l5, geometry=gpd.points_from_xy(df_l5.easting, df_l5.northing), crs=TARGET_CRS)

        forecast_years = sorted([int(c.split('_')[-1]) for c in substation_profile.columns if 'headroom_mw_' in c])
        def get_relevant_headroom_col(year):
            for fy in forecast_years:
                if year < fy: return f'headroom_mw_{fy}'
            return f'[REDACTED_BY_SCRIPT]'

        substation_coords = np.array(list(zip(substation_profile.geometry.x, substation_profile.geometry.y)))
        substation_tree = cKDTree(substation_coords)
        
        results = []
        for _, site in gdf_solar.iterrows():
            dist, indices = substation_tree.query([site.geometry.x, site.geometry.y], k=K_NEIGHBORS)
            neighbors = substation_profile.iloc[indices]
            headroom_col = get_relevant_headroom_col(site['submission_year'])
            
            features = {}
            features['[REDACTED_BY_SCRIPT]'] = dist[0] / 1000.0
            features['[REDACTED_BY_SCRIPT]'] = neighbors.iloc[0]['sitefunctionallocation']
            features['[REDACTED_BY_SCRIPT]'] = neighbors.iloc[0].get(headroom_col, NULL_SENTINEL)
            results.append(features)
        
        df_grid_features = pd.DataFrame(results, index=gdf_solar.index)
        df_l6 = pd.concat([df_l5, df_grid_features], axis=1)
        df_l6.fillna(NULL_SENTINEL, inplace=True)
        df_l6.to_csv(L6_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_02_integrate_earthing():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        K_NEIGHBORS = 5
        df_fault = pd.read_csv(FAULT_LEVEL_CSV_PATH)
        df_fault.columns = [c.lower().strip() for c in df_fault.columns]
        
        gdf_locations = gpd.read_file(DFES_GEOJSON_PATH).to_crs(TARGET_CRS)
        gdf_locations.columns = [c.lower().strip() for c in gdf_locations.columns]
        
        location_lookup = gdf_locations[['sitefunctionallocation', 'geometry']].drop_duplicates(subset='sitefunctionallocation')
        gdf_fault_located = pd.merge(df_fault, location_lookup, on='sitefunctionallocation', how='inner')
        gdf_fault_located = gpd.GeoDataFrame(gdf_fault_located, geometry='geometry', crs=TARGET_CRS)

        fault_cols = ['threephasermsbreak', 'earthfaultrmsbreak']
        for col in fault_cols: gdf_fault_located[col] = pd.to_numeric(gdf_fault_located[col], errors='coerce')
        
        substation_fault_profile = gdf_fault_located.groupby('sitefunctionallocation').agg(
            threephasermsbreak=('threephasermsbreak', 'max'),
            earthfaultrmsbreak=('earthfaultrmsbreak', 'max'),
            geometry=('geometry', 'first')
        ).reset_index()
        substation_fault_profile = gpd.GeoDataFrame(substation_fault_profile, geometry='geometry', crs=TARGET_CRS)

        df_l6 = pd.read_csv(L6_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_l6, geometry=gpd.points_from_xy(df_l6.easting, df_l6.northing), crs=TARGET_CRS)
        
        substation_tree = cKDTree(np.array(list(zip(substation_fault_profile.geometry.x, substation_fault_profile.geometry.y))))

        results = []
        for _, site in gdf_solar.iterrows():
            _, indices = substation_tree.query([site.geometry.x, site.geometry.y], k=K_NEIGHBORS)
            neighbors = substation_fault_profile.iloc[indices]
            features = {}
            features['[REDACTED_BY_SCRIPT]'] = neighbors.iloc[0]['threephasermsbreak']
            features['[REDACTED_BY_SCRIPT]'] = neighbors['threephasermsbreak'].mean()
            results.append(features)
        
        df_stability_features = pd.DataFrame(results, index=gdf_solar.index)
        df_l7 = pd.concat([df_l6, df_stability_features], axis=1)
        df_l7.fillna(NULL_SENTINEL, inplace=True)
        df_l7.to_csv(L7_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_04_integrate_capacity_2():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        # Load and unify ECR data
        gdf_sub1 = gpd.read_file(ECR_SUB_1MW_PATH).to_crs(TARGET_CRS)
        gdf_over1 = gpd.read_file(ECR_OVER_1MW_PATH).to_crs(TARGET_CRS)
        gdf_sub1 = clean_col_names(gdf_sub1)
        gdf_over1 = clean_col_names(gdf_over1)
        gdf_sub1['capacity_scale'] = 'sub_1mw'
        gdf_over1['capacity_scale'] = 'over_1mw'
        ecr_master = pd.concat([gdf_sub1, gdf_over1], ignore_index=True)
        ecr_master = gpd.GeoDataFrame(ecr_master, geometry='geometry', crs=TARGET_CRS)

        # Distill and filter ECR
        for col in ecr_master.columns:
            if 'capacity_mw' in col or 'storage_mwh' in col:
                ecr_master[col] = pd.to_numeric(ecr_master[col], errors='coerce')
        ecr_master['[REDACTED_BY_SCRIPT]'] = ecr_master.filter(like='capacity_mw').sum(axis=1)
        ecr_master['total_storage_mwh'] = ecr_master.filter(like='storage_mwh').sum(axis=1)
        
        essential_cols = ['geometry', 'capacity_scale', 'date_accepted', 'connection_status', 'energy_source_1', '[REDACTED_BY_SCRIPT]', 'total_storage_mwh']
        ecr_final = ecr_master[essential_cols]
        ecr_final = ecr_final[ecr_final['connection_status'].isin(['Connected', 'Accepted to Connect'])].copy()
        ecr_final['date_accepted'] = pd.to_datetime(ecr_final['date_accepted'], errors='coerce')
        ecr_final.dropna(subset=['date_accepted'], inplace=True)

        # Load solar data and calculate features
        df_solar = pd.read_csv(L7_ARTIFACT)
        df_solar['submission_date'] = pd.to_datetime(df_solar['submission_year'].astype(str) + '-' + df_solar['submission_month'].astype(str) + '-' + df_solar['submission_day'].astype(str), errors='coerce')
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS)
        gdf_solar.dropna(subset=['submission_date'], inplace=True)

        ecr_sindex = ecr_final.sindex
        RADII_METERS = [2000, 5000, 10000]
        
        all_features = []
        for _, site in tqdm(gdf_solar.iterrows(), total=len(gdf_solar), desc="[REDACTED_BY_SCRIPT]"):
            features = {}
            for r in RADII_METERS:
                possible_matches_idx = list(ecr_sindex.intersection(site.geometry.buffer(r).bounds))
                possible_matches = ecr_final.iloc[possible_matches_idx]
                actual_matches = possible_matches[possible_matches.intersects(site.geometry.buffer(r))]
                temporal_matches = actual_matches[actual_matches['date_accepted'] < site.submission_date]
                
                features[f'[REDACTED_BY_SCRIPT]'] = len(temporal_matches)
                features[f'[REDACTED_BY_SCRIPT]'] = temporal_matches['[REDACTED_BY_SCRIPT]'].sum()
            all_features.append(features)

        der_features = pd.DataFrame(all_features, index=gdf_solar.index)
        gdf_solar_l9 = gdf_solar.join(der_features)
        df_solar_l9 = pd.DataFrame(gdf_solar_l9.drop(columns=['geometry', 'submission_date']))
        
        df_solar_l9.to_csv(L9_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_04_integrate_transformers():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        K_NEIGHBORS = 5
        RADIUS_METERS = 10000

        gdf_tx = gpd.read_file(TRANSFORMERS_GRID_SITE_PATH).to_crs(TARGET_CRS)
        gdf_tx = clean_col_names(gdf_tx)
        gdf_tx['onanrating_kva'] = pd.to_numeric(gdf_tx['onanrating_kva'], errors='coerce')
        gdf_tx.dropna(subset=['onanrating_kva'], inplace=True)
        
        substation_key = find_column_by_pattern(gdf_tx, 'sitefunctional')
        agg_data = gdf_tx.groupby(substation_key).agg(total_onan_rating_kva=('onanrating_kva', 'sum'), geometry=('geometry', 'first')).reset_index()
        gdf_sub_capacity = gpd.GeoDataFrame(agg_data, geometry='geometry', crs=TARGET_CRS)
        
        df_solar = pd.read_csv(L9_ARTIFACT)
        solar_coords = df_solar[['easting', 'northing']].values
        
        sub_coords = np.array([g.coords[0] for g in gdf_sub_capacity.geometry])
        tree = cKDTree(sub_coords)
        
        results = []
        for coords in tqdm(solar_coords, desc="[REDACTED_BY_SCRIPT]"):
            features = {}
            dist, idx = tree.query(coords, k=K_NEIGHBORS)
            valid_idx = idx[np.isfinite(dist)]
            if len(valid_idx) > 0:
                neighbors = gdf_sub_capacity.iloc[valid_idx]
                features['[REDACTED_BY_SCRIPT]'] = neighbors.iloc[0]['total_onan_rating_kva']
                features['avg_total_kva_5nn'] = neighbors['total_onan_rating_kva'].mean()
            else:
                features['[REDACTED_BY_SCRIPT]'] = 0
                features['avg_total_kva_5nn'] = 0
            results.append(features)
        
        tx_features = pd.DataFrame(results)
        df_solar_l10 = pd.concat([df_solar, tx_features], axis=1)
        df_solar_l10.fillna(NULL_SENTINEL, inplace=True)
        
        df_solar_l10.to_csv(L10_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_05_integrate_idnos():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        # Ingest and unify IDNO data
        gdf_idno_raw = gpd.read_file(IDNO_RAW_PATH)
        gdf_idno = gdf_idno_raw.to_crs(TARGET_CRS)
        gdf_idno.to_file(IDNO_L1_UNIFIED_PATH, driver='GeoJSON')
        
        # Load prerequisite data
        df_solar_l10 = pd.read_csv(L10_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar_l10, geometry=gpd.points_from_xy(df_solar_l10.easting, df_solar_l10.northing), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index

        gdf_subs = gpd.read_file(TRANSFORMERS_GRID_SITE_PATH).to_crs(TARGET_CRS)
        gdf_lpa = gpd.read_file(LPA_BOUNDARIES_PATH).to_crs(TARGET_CRS)
        gdf_lpa = clean_col_names(gdf_lpa)

        # Generate core IDNO features
        within_join = gpd.sjoin(gdf_solar, gdf_idno, how='left', predicate='within')
        gdf_solar['idno_is_within'] = ~within_join['index_right'].isnull()
        
        sites_outside = gdf_solar[~gdf_solar['idno_is_within']]
        distances = sites_outside.geometry.apply(lambda g: gdf_idno.distance(g).min())
        gdf_solar['[REDACTED_BY_SCRIPT]'] = distances
        gdf_solar['[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)
        
        # Generate grid interaction features
        sub_sindex = gdf_subs.sindex
        nearest_indices = [list(sub_sindex.nearest(g, return_all=False))[1][0] for g in gdf_solar.geometry]
        nearest_subs = gdf_subs.iloc[nearest_indices].reset_index(drop=True)
        paths = [LineString([sg, subg]) for sg, subg in zip(gdf_solar.geometry, nearest_subs.geometry)]
        gdf_paths = gpd.GeoDataFrame(geometry=paths, crs=TARGET_CRS, index=gdf_solar.index)
        path_crosses = gpd.sjoin(gdf_paths, gdf_idno, how='inner', predicate='intersects')
        gdf_solar['[REDACTED_BY_SCRIPT]'] = gdf_solar.index.isin(path_crosses.index.unique())

        # Final Integration
        gdf_solar_with_lpa_key = gpd.sjoin(gdf_solar, gdf_lpa[['lpa23nm', 'geometry']], how='left', predicate='within').drop_duplicates(subset=['solar_farm_id'])
        
        df_solar_l11 = pd.DataFrame(gdf_solar_with_lpa_key.drop(columns=['geometry', 'index_right']))
        bool_cols = df_solar_l11.select_dtypes(include=['bool']).columns
        df_solar_l11[bool_cols] = df_solar_l11[bool_cols].astype(int)
        
        df_solar_l11.to_csv(L11_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_06_integrate_ltds():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    def calculate_ltds_features(app_row, master_gdf):
        features = {'[REDACTED_BY_SCRIPT]': NULL_SENTINEL, 'ltds_count_in_10km': 0}
        valid_ltds = master_gdf[master_gdf['[REDACTED_BY_SCRIPT]'] <= app_row['submission_date']]
        if valid_ltds.empty: return pd.Series(features)
        
        latest_pub_date = valid_ltds['[REDACTED_BY_SCRIPT]'].max()
        pit_ltds = valid_ltds[valid_ltds['[REDACTED_BY_SCRIPT]'] == latest_pub_date].copy()
        
        pit_ltds['distance'] = pit_ltds.geometry.distance(app_row.geometry)
        nearest = pit_ltds.loc[pit_ltds['distance'].idxmin()]
        features['[REDACTED_BY_SCRIPT]'] = nearest['distance']
        
        in_10km = pit_ltds[pit_ltds.intersects(app_row.geometry.buffer(10000))]
        features['ltds_count_in_10km'] = len(in_10km)
        return pd.Series(features)
        
    try:
        gdfs = []
        for path, date in LTDS_SOURCE_FILES.items():
            gdf = gpd.read_file(path)
            gdf['[REDACTED_BY_SCRIPT]'] = pd.to_datetime(date)
            gdf = gdf.set_crs('EPSG:4326', allow_override=True).to_crs(TARGET_CRS)
            gdfs.append(gdf)
        gdf_ltds_master = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=TARGET_CRS)
        gdf_ltds_master.rename(columns=lambda x: x.lower(), inplace=True)
        
        df_solar = pd.read_csv(L11_ARTIFACT)
        df_solar['submission_date'] = pd.to_datetime(df_solar['submission_date'], errors='coerce')
        df_solar.dropna(subset=['submission_date', 'easting', 'northing'], inplace=True)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS)
        
        ltds_features = gdf_solar.progress_apply(lambda row: calculate_ltds_features(row, gdf_ltds_master), axis=1)
        
        df_solar_l12 = pd.concat([df_solar, ltds_features], axis=1).drop(columns=['geometry', 'submission_date'])
        
        df_solar_l12.to_csv(L12_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_08_integrate_lct():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_lct = gpd.read_file(LCT_SECONDARY_SITES_PATH).to_crs(TARGET_CRS)
        gdf_lct = clean_col_names(gdf_lct)
        for col in ['lct_connections', 'import', 'export']:
            gdf_lct[col] = pd.to_numeric(gdf_lct[col], errors='coerce')
        gdf_lct.dropna(subset=['lct_connections', '[REDACTED_BY_SCRIPT]'], inplace=True)
        
        df_solar = pd.read_csv(L12_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index
        
        gdf_primary_subs = gdf_lct.drop_duplicates(subset='[REDACTED_BY_SCRIPT]').set_index('[REDACTED_BY_SCRIPT]')
        gdf_solar = gpd.sjoin_nearest(gdf_solar, gdf_primary_subs[['geometry']], how='left').rename(columns={'index_right': '[REDACTED_BY_SCRIPT]'})
        
        gdf_solar_buffers = gdf_solar.copy()
        gdf_solar_buffers['geometry'] = gdf_solar_buffers.geometry.buffer(5000)
        sjoined = gpd.sjoin(gdf_lct, gdf_solar_buffers, how="inner", predicate="within")
        
        grouped = sjoined.groupby('solar_farm_id')
        buffer_features = grouped.agg(lct_secondary_sub_count_in_5km=('solar_farm_id', 'size'), lct_total_connections_in_5km=('lct_connections', 'sum'))
        
        df_solar_l13 = pd.DataFrame(gdf_solar).merge(buffer_features, on='solar_farm_id', how='left').drop(columns=['geometry', 'index_right'])
        df_solar_l13.fillna(NULL_SENTINEL, inplace=True)
        
        df_solar_l13.to_csv(L13_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_08_integrate_lct_2():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        df_solar_l13 = pd.read_csv(L13_ARTIFACT)
        
        df_primary_lct = pd.read_csv(LCT_PRIMARY_AGGREGATE_PATH)
        df_primary_lct.columns = [c.lower().strip().replace(' ', '_') for c in df_primary_lct.columns]
        df_primary_lct[['latitude', 'longitude']] = df_primary_lct['spatialcoordinates'].str.split(',', expand=True).astype(float)
        gdf_primary_lct = gpd.GeoDataFrame(df_primary_lct, geometry=gpd.points_from_xy(df_primary_lct.longitude, df_primary_lct.latitude), crs="EPSG:4326").to_crs(TARGET_CRS)

        primary_lct_agg = gdf_primary_lct.groupby('sitefunctionallocation').agg(primary_direct_total_connections=('lct_connections', 'sum')).reset_index()

        df_enriched = df_solar_l13.merge(primary_lct_agg, left_on='[REDACTED_BY_SCRIPT]', right_on='sitefunctionallocation', how='left')
        
        gdf_granular = gpd.read_file(LCT_SECONDARY_SITES_PATH)
        gdf_granular.columns = [c.lower().strip() for c in gdf_granular.columns]
        gdf_granular['lct_connections'] = pd.to_numeric(gdf_granular['lct_connections'], errors='coerce').fillna(0)
        concentration_lookup = gdf_granular.groupby('[REDACTED_BY_SCRIPT]')['lct_connections'].max().reset_index().rename(columns={'lct_connections': '[REDACTED_BY_SCRIPT]'})
        
        df_enriched = df_enriched.merge(concentration_lookup, on='[REDACTED_BY_SCRIPT]', how='left')
        
        df_enriched.fillna(NULL_SENTINEL, inplace=True)
        df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] / (df_enriched['[REDACTED_BY_SCRIPT]'] + 1)
        
        df_enriched.to_csv(L14_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_09_integrate_dnoa():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_dnoa = gpd.read_file(DNOA_LV_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_dnoa.columns = [c.lower().strip().replace(' ', '_') for c in gdf_dnoa.columns]
        for col in ['constraint_year', '[REDACTED_BY_SCRIPT]']:
            gdf_dnoa[col] = pd.to_numeric(gdf_dnoa[col], errors='coerce')
        gdf_dnoa.dropna(subset=['constraint_year', '[REDACTED_BY_SCRIPT]'], inplace=True)
        
        df_solar = pd.read_csv(L14_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index

        dnoa_tree = BallTree(np.array(list(zip(gdf_dnoa.geometry.x, gdf_dnoa.geometry.y))))

        results = []
        for _, app in gdf_solar.iterrows():
            features = {'solar_farm_id': app['solar_farm_id']}
            relevant_dnoa = gdf_dnoa[(gdf_dnoa['constraint_year'] >= app['submission_year']) & (gdf_dnoa['constraint_year'] <= app['submission_year'] + 3)]
            if not relevant_dnoa.empty:
                dist, idx = dnoa_tree.query([[app.geometry.x, app.geometry.y]], k=1)
                nearest_idx = [i for i in idx[0] if gdf_dnoa.index[i] in relevant_dnoa.index]
                if nearest_idx:
                    nearest = gdf_dnoa.iloc[nearest_idx[0]]
                    features['[REDACTED_BY_SCRIPT]'] = dist[0][0]
                    features['[REDACTED_BY_SCRIPT]'] = nearest['constraint_year'] - app['submission_year']
            results.append(features)

        knn_features = pd.DataFrame(results)
        df_enriched = df_solar.merge(knn_features, on='solar_farm_id', how='left').drop(columns=['geometry'])
        df_enriched.fillna(0, inplace=True)
        
        df_enriched.to_csv(L15_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_10_integrate_pq():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    def aggregate_site(group):
        pivot = group.pivot_table(index='sitefunctionallocation', columns='harmonic_order', values='highest')
        res = {
            'pq_thd_highest': pivot.get(-1, pd.Series(0)).iloc[0],
            'pq_h5_highest': pivot.get(5, pd.Series(0)).iloc[0],
            '[REDACTED_BY_SCRIPT]': group[group['harmonic_order'] > 0]['highest'].mean()
        }
        return pd.Series(res)
    
    try:
        gdf_pq = gpd.read_file(PQ_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_pq.columns = [c.lower().strip() for c in gdf_pq.columns]
        gdf_pq['highest'] = pd.to_numeric(gdf_pq['highest'], errors='coerce')
        gdf_pq.dropna(subset=['highest', 'sitefunctionallocation'], inplace=True)
        gdf_pq['harmonic_order'] = gdf_pq['harmonic'].replace('THD', '-1').str.replace('H', '').astype(int)
        
        agg_data = gdf_pq.groupby('sitefunctionallocation').apply(aggregate_site)
        sites_geom = gdf_pq[['sitefunctionallocation', 'geometry']].drop_duplicates(subset='sitefunctionallocation').set_index('sitefunctionallocation')
        gdf_pq_l1 = sites_geom.join(agg_data).reset_index()

        df_solar = pd.read_csv(L15_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index

        pq_coords = np.array(list(zip(gdf_pq_l1.geometry.x, gdf_pq_l1.geometry.y)))
        pq_tree = BallTree(pq_coords)
        app_coords = np.array(list(zip(gdf_solar.geometry.x, gdf_solar.geometry.y)))
        dist, indices = pq_tree.query(app_coords, k=5)
        
        results = []
        for i in range(len(gdf_solar)):
            cohort = gdf_pq_l1.iloc[indices[i]]
            weights = 1 / (dist[i] ** 2 + 1e-6)
            res = {
                'solar_farm_id': gdf_solar.iloc[i]['solar_farm_id'],
                'pq_idw_thd_knn5': np.average(cohort['pq_thd_highest'], weights=weights),
                '[REDACTED_BY_SCRIPT]': dist[i][0] / 1000
            }
            results.append(res)

        pq_features = pd.DataFrame(results)
        df_enriched = df_solar.merge(pq_features, on='solar_farm_id', how='left')
        df_enriched.fillna(0, inplace=True)

        df_enriched.to_csv(L16_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_11_integrate_transformers():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_tx = gpd.read_file(TRANSFORMERS_PRIMARY_SITE_PATH).to_crs(TARGET_CRS)
        gdf_tx.columns = [c.lower().strip().replace(' ', '_') for c in gdf_tx.columns]
        gdf_tx['onanrating_kva'] = pd.to_numeric(gdf_tx['onanrating_kva'], errors='coerce') * 1000
        gdf_tx.dropna(subset=['onanrating_kva', 'sitefunctionallocation'], inplace=True)

        agg_rules = {'onanrating_kva': ['sum', 'count']}
        agg_data = gdf_tx.groupby('sitefunctionallocation').agg(agg_rules)
        agg_data.columns = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
        
        sites_geom = gdf_tx[['sitefunctionallocation', 'geometry']].drop_duplicates(subset='sitefunctionallocation').set_index('sitefunctionallocation')
        gdf_tx_l1 = sites_geom.join(agg_data).reset_index()

        df_solar = pd.read_csv(L16_ARTIFACT)
        df_enriched = df_solar.merge(gdf_tx_l1.drop(columns='geometry'), left_on='[REDACTED_BY_SCRIPT]', right_on='sitefunctionallocation', how='left')
        
        df_enriched.fillna(0, inplace=True)
        df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] / (df_enriched['[REDACTED_BY_SCRIPT]'] * 1000)
        df_enriched['[REDACTED_BY_SCRIPT]'].replace([np.inf, -np.inf], 0, inplace=True)
        
        df_enriched.to_csv(L17_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_12_integrate_gandp():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_bio = gpd.read_file(SUBSTATION_BIO_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_bio.columns = [c.lower().strip().replace('.', '_') for c in gdf_bio.columns]
        
        numeric_cols = ['powertransformercount', 'maxdemandwinter', 'maxdemandsummer']
        for col in numeric_cols:
            gdf_bio[col] = pd.to_numeric(gdf_bio[col], errors='coerce')
        
        df_solar = pd.read_csv(L17_ARTIFACT)
        df_solar['submission_date'] = pd.to_datetime(df_solar['submission_date'], errors='coerce')

        df_enriched = df_solar.merge(gdf_bio.drop(columns='geometry'), left_on='[REDACTED_BY_SCRIPT]', right_on='sitefunctionallocation', how='left')
        
        df_enriched['[REDACTED_BY_SCRIPT]'] = (df_enriched['maxdemandwinter'] * 1000) / df_enriched['[REDACTED_BY_SCRIPT]']
        df_enriched['[REDACTED_BY_SCRIPT]'] = (df_enriched['maxdemandsummer'] * 1000) / df_enriched['[REDACTED_BY_SCRIPT]']
        df_enriched.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        new_cols = [c for c in df_enriched.columns if c not in df_solar.columns]
        df_enriched[new_cols] = df_enriched[new_cols].fillna(0)
        
        df_enriched.to_csv(L18_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_13_integrate_ohl_1():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_ohl = gpd.read_file(OHL_33KV_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_ohl_corridors = gdf_ohl.copy()
        gdf_ohl_corridors['geometry'] = gdf_ohl.geometry.buffer(20)
        gdf_ohl_corridors.reset_index(drop=True, inplace=True)
        
        df_solar = pd.read_csv(L18_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting_x, df_solar.northing_x), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index

        _, tree_indices = gdf_ohl_corridors.sindex.nearest(gdf_solar.geometry, return_distance=False)
        nearest_geoms = gdf_ohl_corridors.geometry.iloc[tree_indices]
        distances = gdf_solar.geometry.distance(nearest_geoms.reset_index(drop=True), align=True)
        
        df_features = pd.DataFrame({'solar_farm_id': gdf_solar.index, '[REDACTED_BY_SCRIPT]': distances})
        
        df_enriched = df_solar.merge(df_features, on='solar_farm_id', how='left')
        
        df_enriched.fillna(0, inplace=True)
        
        df_enriched.to_csv(L19_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_14_integrate_tandp_1():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_structures = gpd.read_file(OHL_33KV_POINTS_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_structures.columns = [c.lower().strip() for c in gdf_structures.columns]
        gdf_towers = gdf_structures[gdf_structures['ob_class'].str.contains("tower", case=False, na=False)].copy()

        df_solar = pd.read_csv(L19_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting_x, df_solar.northing_x), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index
        
        gdf_solar_with_towers = gpd.sjoin_nearest(gdf_solar, gdf_towers, distance_col="[REDACTED_BY_SCRIPT]", how="left").drop_duplicates(subset='solar_farm_id')
        
        gdf_solar_buffers = gdf_solar.copy()
        gdf_solar_buffers['geometry'] = gdf_solar_buffers.geometry.buffer(2000)
        towers_in_buffer = gpd.sjoin(gdf_towers, gdf_solar_buffers, how="inner", predicate="within")
        tower_counts = towers_in_buffer.groupby('solar_farm_id').size().reset_index(name='ohl_local_tower_count')

        df_enriched = gdf_solar.merge(gdf_solar_with_towers[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
        df_enriched = df_enriched.merge(tower_counts, on='solar_farm_id', how='left')
        
        df_enriched.fillna(0, inplace=True)

        df_enriched.to_csv(L20_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_15_integrate_ohl_tandp_2():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_lines_132 = gpd.read_file(OHL_132KV_LINES_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_towers_132 = gpd.read_file(OHL_132KV_TOWERS_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_corridors_132 = gdf_lines_132.copy()
        gdf_corridors_132['geometry'] = gdf_lines_132.geometry.buffer(30)
        
        df_solar = pd.read_csv(L20_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting_x, df_solar.northing_x), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index
        
        gdf_solar_with_dist = gpd.sjoin_nearest(gdf_solar, gdf_corridors_132, distance_col="[REDACTED_BY_SCRIPT]", how="left").drop_duplicates(subset='solar_farm_id')
        
        df_enriched = gdf_solar.merge(gdf_solar_with_dist[['solar_farm_id', '[REDACTED_BY_SCRIPT]']], on='solar_farm_id', how='left')
        
        df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] / (df_enriched['[REDACTED_BY_SCRIPT]'] + 1)
        df_enriched['ohl_nearest_voltage'] = np.where(df_enriched['[REDACTED_BY_SCRIPT]'] < df_enriched['[REDACTED_BY_SCRIPT]'], 132, 33)
        
        df_enriched.fillna(0, inplace=True)
        
        df_enriched.to_csv(L21_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_16_integrate_ohl_tandp_hv():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        dhv_lines = gpd.read_file(HV_LINES_INPUT_PATH).to_crs(TARGET_CRS)
        dhv_corridors = dhv_lines.copy()
        dhv_corridors['geometry'] = dhv_lines.geometry.buffer(20)

        df_solar = pd.read_csv(L21_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting_x, df_solar.northing_x), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index
        
        gdf_subs = gpd.read_file(SUBSTATION_L1_ARTIFACT).to_crs(TARGET_CRS)
        
        gdf_nearest_sub = gpd.sjoin_nearest(gdf_solar, gdf_subs, how='left').drop_duplicates(subset=['solar_farm_id'])
        nearest_sub_geoms = gdf_nearest_sub['index_right'].map(gdf_subs.geometry)
        paths = [LineString([sg, subg]) if pd.notna(subg) else None for sg, subg in zip(gdf_nearest_sub.geometry, nearest_sub_geoms)]
        gdf_paths = gpd.GeoDataFrame(gdf_nearest_sub[['solar_farm_id']], geometry=paths, crs=TARGET_CRS).dropna()

        path_intersections = gpd.sjoin(gdf_paths, dhv_corridors, how='inner', predicate='intersects')
        intersection_counts = path_intersections.groupby('solar_farm_id').size().reset_index(name='[REDACTED_BY_SCRIPT]')

        df_enriched = gdf_solar.merge(intersection_counts, on='solar_farm_id', how='left')
        
        df_enriched.fillna(0, inplace=True)

        df_enriched.to_csv(L22_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_17_integrate_ohl_tandp_lv():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        dlv_lines = gpd.read_file(LV_LINES_INPUT_PATH).to_crs(TARGET_CRS)
        dlv_corridors = dlv_lines.copy()
        dlv_corridors['geometry'] = dlv_lines.geometry.buffer(5)

        df_solar = pd.read_csv(L22_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting_x, df_solar.northing_x), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index
        
        gdf_subs = gpd.read_file(SUBSTATION_L1_ARTIFACT).to_crs(TARGET_CRS)
        
        gdf_nearest_sub = gpd.sjoin_nearest(gdf_solar, gdf_subs, how='left').drop_duplicates(subset=['solar_farm_id'])
        nearest_sub_geoms = gdf_nearest_sub['index_right'].map(gdf_subs.geometry)
        paths = [LineString([sg, subg]) if pd.notna(subg) else None for sg, subg in zip(gdf_nearest_sub.geometry, nearest_sub_geoms)]
        gdf_paths = gpd.GeoDataFrame(gdf_nearest_sub[['solar_farm_id']], geometry=paths, crs=TARGET_CRS).dropna()
        
        path_intersections = gpd.sjoin(gdf_paths, dlv_corridors, how='inner', predicate='intersects')
        intersection_counts = path_intersections.groupby('solar_farm_id').size().reset_index(name='[REDACTED_BY_SCRIPT]')

        df_enriched = gdf_solar.merge(intersection_counts, on='solar_farm_id', how='left')
        
        df_enriched.fillna(0, inplace=True)
        
        df_enriched['[REDACTED_BY_SCRIPT]'] = df_enriched['[REDACTED_BY_SCRIPT]'] + df_enriched['[REDACTED_BY_SCRIPT]']
        
        df_enriched.to_csv(L23_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_18_integrate_substation_headroom():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_service_areas = gpd.read_file(SERVICE_AREA_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_service_areas = clean_col_names(gdf_service_areas)
        gdf_service_areas.rename(columns={'firmcapacitywinter': '[REDACTED_BY_SCRIPT]'}, inplace=True)
        gdf_service_areas['[REDACTED_BY_SCRIPT]'] = gdf_service_areas['demand'] / 100.0
        
        df_solar = pd.read_csv(L23_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting_x, df_solar.northing_x), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index
        
        gdf_joined = gpd.sjoin(gdf_solar, gdf_service_areas, how='left', predicate='within').drop_duplicates(subset=['solar_farm_id'])
        
        df_final = pd.DataFrame(gdf_joined.drop(columns=['geometry', 'index_right']))
        
        df_final.to_csv(L24_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_19_integrate_secondary_sites():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_sec_subs = gpd.read_file(SECONDARY_SUB_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_sec_subs = clean_col_names(gdf_sec_subs)
        
        gdf_sec_subs['utilisation_midpoint_pct'] = ((pd.to_numeric(gdf_sec_subs['utilisation_band'].str.extract(r'(\d+)')[0], errors='coerce') + pd.to_numeric(gdf_sec_subs['utilisation_band'].str.extract(r'-(\d+)%')[0], errors='coerce')) / 2) / 100.0
        gdf_sec_subs['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(gdf_sec_subs['[REDACTED_BY_SCRIPT]'], errors='coerce').fillna(9999)

        df_solar = pd.read_csv(L24_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting_x, df_solar.northing_x), crs=TARGET_CRS)

        sec_sub_coords = np.array(list(zip(gdf_sec_subs.geometry.x, gdf_sec_subs.geometry.y)))
        sec_sub_tree = cKDTree(sec_sub_coords)

        results = []
        for _, site in gdf_solar.iterrows():
            _, indices = sec_sub_tree.query([site.geometry.x, site.geometry.y], k=50)
            neighbors = gdf_sec_subs.iloc[indices]
            mask = (neighbors['[REDACTED_BY_SCRIPT]'] >= site['submission_year']) & (neighbors['[REDACTED_BY_SCRIPT]'] < site['submission_year'] + 5)
            features = {'[REDACTED_BY_SCRIPT]': neighbors['utilisation_midpoint_pct'].mean(), '[REDACTED_BY_SCRIPT]': mask.sum()}
            results.append(features)

        knn_features = pd.DataFrame(results, index=gdf_solar.index)
        df_enriched = gdf_solar.merge(knn_features, left_index=True, right_index=True, how='left')
        
        df_final = pd.DataFrame(df_enriched.drop(columns=['geometry']))
        df_final.fillna(0, inplace=True)
        
        df_final.to_csv(L25_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


def run_step_6_20_integrate_secondary_sites_2():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_sec_areas = gpd.read_file(SEC_SUB_AREA_INPUT_PATH).to_crs(TARGET_CRS)
        gdf_sec_areas = clean_col_names(gdf_sec_areas)
        gdf_sec_areas['utilisation_midpoint_pct'] = ((pd.to_numeric(gdf_sec_areas['utilisation_band'].str.extract(r'(\d+)')[0], errors='coerce') + pd.to_numeric(gdf_sec_areas['utilisation_band'].str.extract(r'-(\d+)%')[0], errors='coerce')) / 2) / 100.0
        
        df_solar = pd.read_csv(L25_ARTIFACT)
        gdf_solar = gpd.GeoDataFrame(df_solar, geometry=gpd.points_from_xy(df_solar.easting_x, df_solar.northing_x), crs=TARGET_CRS)
        gdf_solar['solar_farm_id'] = gdf_solar.index
        
        gdf_joined = gpd.sjoin(gdf_solar, gdf_sec_areas, how='left', predicate='within').drop_duplicates(subset=['solar_farm_id'])
        
        knn_cols_to_drop = [c for c in gdf_joined.columns if '_knn50' in c]
        gdf_joined.drop(columns=knn_cols_to_drop, inplace=True)
        
        gdf_joined.rename(columns={'utilisation_midpoint_pct': '[REDACTED_BY_SCRIPT]'}, inplace=True)
        
        df_final = pd.DataFrame(gdf_joined.drop(columns=['geometry', 'index_right']))
        
        df_final.to_csv(L26_ARTIFACT, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)


# =================================================================================================
# V. MASTER ORCHESTRATION
# =================================================================================================

def main():
    """
    Main orchestrator function for the entire Amaryllis data processing pipeline.
    Executes each step in the correct, sequential order.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Execute Foundation Steps (L1-L5)
    run_step_1_ingest_repd()
    run_step_2_standardise_and_encode()
    run_step_3_process_lpa_csvs()
    run_step_4_aggregate_years()
    run_step_4_5_uid()
    run_step_5_aggregate_non_solar()

    # Execute Geospatial & Grid Integration Steps (L6-L26)
    run_step_6_01_integrate_dfes()
    run_step_6_02_integrate_earthing()
    run_step_6_04_integrate_capacity_2()
    run_step_6_04_integrate_transformers()
    run_step_6_05_integrate_idnos()
    run_step_6_06_integrate_ltds()
    run_step_6_08_integrate_lct()
    run_step_6_08_integrate_lct_2()
    run_step_6_09_integrate_dnoa()
    run_step_6_10_integrate_pq()
    run_step_6_11_integrate_transformers()
    run_step_6_12_integrate_gandp()
    run_step_6_13_integrate_ohl_1()
    run_step_6_14_integrate_tandp_1()
    run_step_6_15_integrate_ohl_tandp_2()
    run_step_6_16_integrate_ohl_tandp_hv()
    run_step_6_17_integrate_ohl_tandp_lv()
    run_step_6_18_integrate_substation_headroom()
    run_step_6_19_integrate_secondary_sites()
    run_step_6_20_integrate_secondary_sites_2()
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")


if __name__ == "__main__":
    main()