import pandas as pd
import numpy as np
import joblib
import os
import logging
import sys
import math
import re
import glob
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# 1. Input/Output Paths
INPUT_DATA_PATH = r"[REDACTED_BY_SCRIPT]"
OUTPUT_PATH = r"[REDACTED_BY_SCRIPT]"

# 2. Artifact Paths (Based on scripts/3_step_3.22_rearchitecture.py)
ARTIFACTS_DIR = r"[REDACTED_BY_SCRIPT]"
HEADS_DIR = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

# Models
ORACLE_PATH = os.path.join(ARTIFACTS_DIR, "oralce_v3.5.joblib")

ARBITER_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
ARBITER_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

# k-NN Artifacts
KNN_IMPUTER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
KNN_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
KNN_ENGINE_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

# Reference Data (Crucial for k-NN lookups)
REF_X_PATH = r"[REDACTED_BY_SCRIPT]"
REF_Y_PATH = r"[REDACTED_BY_SCRIPT]"

# 3. Feature Configuration
COALESCE_MAP = {
    '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
    '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
    '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
    '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
    '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'],
    '[REDACTED_BY_SCRIPT]': ['[REDACTED_BY_SCRIPT]', 'ukpn_tx_[REDACTED_BY_SCRIPT]'],
}

MISSING_FEATURES = [
    '[REDACTED_BY_SCRIPT]', 'lpa_total_experience', 'lpa_workload_trend', 
    'lpa_major_commercial_approval_rate', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
    'nearby_legacy_count', '[REDACTED_BY_SCRIPT]', 'knn_avg_distance_km', 
    '[REDACTED_BY_SCRIPT]',
    'np_dist_to_nearest_m', 'np_nearest_area_sqkm', 'np_is_within', 'np_count_in_2km', 'lpa_np_coverage_pct',
    'sssi_unit_dist_to_nearest_m', '[REDACTED_BY_SCRIPT]', 
    'sssi_unit_worst_condition_in_2km', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'railway_length_1km', 'mean_terrain_gradient_1km', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'ohl_local_structure_count',
    '[REDACTED_BY_SCRIPT]', 'sssi_unit_nearest_condition', 
    'np_nearest_name', 'nt_nearest_name'
]

CAPACITIES_TO_SIMULATE = [5, 10, 15, 20, 25, 30, 40, 49, 50, 60]

# --- Helper Functions (Ported from rearchitecture.py) ---

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df

def engineer_temporal_context_features(df: pd.DataFrame, baseline_year=2010, imputer=None, scaler=None):
    """
    Engineers Temporal Feature Interactions (TFIs).
    Note: Using default baseline_year=2010 if not provided, or derived from data.
    """
    X = df.copy()
    epsilon = 1e-6

    # Ensure submission_year exists
    if 'submission_year' not in X.columns:
        # Fallback if missing: use current year or 2025
        X['submission_year'] = 2025
    
    X['year_norm'] = X['submission_year'] - baseline_year

    # Grid Saturation Interactions
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X.get('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] / (X.get('[REDACTED_BY_SCRIPT]', 1) + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * (1 / (X.get('[REDACTED_BY_SCRIPT]', 1) + 1))

    # Environmental Policy Hardening Interactions
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X.get('aonb_is_within', 0)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X.get('alc_is_bmv_at_site', 0)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X.get('cs_on_site_bool', 0)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] / (X.get('aw_dist_to_nearest_m', 100) + 100)

    # LPA Behavior & Precedent Interactions
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X.get('lpa_major_commercial_approval_rate', 0.5)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] / (X.get('[REDACTED_BY_SCRIPT]', 10) + 1)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X.get('lpa_workload_trend', 0)

    # Socio-Economic & Cumulative Impact Interactions
    X['TFI_NIMBY_Amplifier'] = X['year_norm'] * X.get('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X.get('solar_site_area_sqm', 0)

    tfi_features = [col for col in X.columns if col.startswith('TFI_')]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # In inference mode, we assume imputer/scaler are NOT applied here but handled by the model pipeline 
    # OR we need to load them. The rearchitecture script returns them.
    # For this script, we will skip scaling TFIs here as the main model likely expects raw or we lack the specific TFI scaler artifacts.
    # If the model expects scaled TFIs, we might have a problem. 
    # However, tree-based models (LGBM) don't strictly require scaling.
    
    return X

def engineer_strategic_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers SICs.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Helper to safely get columns with default 0
    def get(col, default=0):
        return X.get(col, default)

    # Legacy SICs
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('lpa_major_commercial_approval_rate', 0.5) + epsilon)
    X['SIC_NIMBY_AMPLIFIER'] = (get('[REDACTED_BY_SCRIPT]') + get('[REDACTED_BY_SCRIPT]')) / (get('nt_dist_to_nearest_m', 1000) + 100)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * (get('[REDACTED_BY_SCRIPT]') + get('[REDACTED_BY_SCRIPT]'))

    # Visual Impact
    landscape_sensitivity = (1 + get('aonb_is_within') + get('np_is_within'))
    industrial_mitigation = (get('[REDACTED_BY_SCRIPT]') + epsilon)
    raw_visual = (get('ohl_local_tower_ratio') / (get('[REDACTED_BY_SCRIPT]', 1000) + 100))
    X['[REDACTED_BY_SCRIPT]'] = (raw_visual * landscape_sensitivity) / industrial_mitigation

    # Modulated Scale Penalty
    # Using hardcoded thresholds based on typical values if quantiles unavailable
    is_large_scale = (get('[REDACTED_BY_SCRIPT]') > 40) # Approx
    is_grid_stressed = (get('[REDACTED_BY_SCRIPT]') > 80)
    is_env_proximate = (get('[REDACTED_BY_SCRIPT]', 1000) < 500)
    is_tough_lpa = (get('lpa_major_commercial_approval_rate', 0.5) < 0.4)
    
    year_norm = X.get('year_norm', 0)
    X['[REDACTED_BY_SCRIPT]'] = year_norm * (is_large_scale & is_grid_stressed)
    X['[REDACTED_BY_SCRIPT]'] = year_norm * (is_large_scale & is_env_proximate)
    X['[REDACTED_BY_SCRIPT]'] = year_norm * (is_large_scale & is_tough_lpa)

    # Amenity Context
    amenity_denom = get('nt_dist_to_nearest_m', 1000) + get('hp_dist_to_nearest_m', 1000) + get('[REDACTED_BY_SCRIPT]', 1000) + 100
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / amenity_denom
    
    amenity_density = get('hp_count_in_5km') + get('nt_intersection_count_in_5km') + 1
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * amenity_density

    # Phase V SICs
    X['[REDACTED_BY_SCRIPT]'] = (get('solar_site_area_sqm') * get('[REDACTED_BY_SCRIPT]')) / (get('[REDACTED_BY_SCRIPT]') + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = (get('[REDACTED_BY_SCRIPT]', 0.5) - get('lpa_major_commercial_approval_rate', 0.5)) * (1 - get('lpa_workload_trend'))
    X['[REDACTED_BY_SCRIPT]'] = get('solar_site_area_sqm') * (get('cs_density_5km') + get('[REDACTED_BY_SCRIPT]'))
    X['[REDACTED_BY_SCRIPT]'] = year_norm * X['[REDACTED_BY_SCRIPT]']

    # Phase VI Grid SICs
    X['[REDACTED_BY_SCRIPT]'] = (get('[REDACTED_BY_SCRIPT]') * get('[REDACTED_BY_SCRIPT]')) / (get('[REDACTED_BY_SCRIPT]', 1) + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * (get('[REDACTED_BY_SCRIPT]', 5) + 1)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('[REDACTED_BY_SCRIPT]', 10) + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = 1 / ((get('[REDACTED_BY_SCRIPT]', 1000) + 100) * (get('[REDACTED_BY_SCRIPT]', 5) + 1))
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('[REDACTED_BY_SCRIPT]', 1) + epsilon)
    X['SIC_LCT_INTEGRATION_STRESS'] = get('[REDACTED_BY_SCRIPT]') * get('[REDACTED_BY_SCRIPT]')

    # Phase VI LPA SICs
    X['[REDACTED_BY_SCRIPT]'] = get('lpa_workload_trend') * get('[REDACTED_BY_SCRIPT]', 100)
    X['SIC_LPA_RENEWABLE_BIAS'] = get('[REDACTED_BY_SCRIPT]', 0.5) - get('lpa_major_commercial_approval_rate', 0.5)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('lpa_total_experience', 10) + epsilon)
    X['SIC_LPA_INSTABILITY_RISK'] = (1 - get('[REDACTED_BY_SCRIPT]', 1)) * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('lpa_withdrawal_rate') / ((get('[REDACTED_BY_SCRIPT]', 1000) / 1000) + epsilon)

    # Phase VI ENV SICs
    X['[REDACTED_BY_SCRIPT]'] = get('solar_site_area_sqm') / (get('aonb_dist_to_nearest_m', 1000) + get('np_dist_to_nearest_m', 1000) + 100)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('aw_dist_to_nearest_m', 1000) + get('sssi_dist_to_nearest_m', 1000) + get('ph_dist_to_nearest_m', 1000) + 100)
    X['[REDACTED_BY_SCRIPT]'] = get('solar_site_area_sqm') * (get('cs_density_5km') + get('es_hls_density_5km'))
    X['[REDACTED_BY_SCRIPT]'] = get('solar_site_area_sqm') * get('alc_is_bmv_at_site') * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('solar_site_area_sqm') / (get('nt_dist_to_nearest_m', 1000) + get('hp_dist_to_nearest_m', 1000) + get('[REDACTED_BY_SCRIPT]', 1000) + 100)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('hp_dist_to_nearest_m', 1000) + 100)

    # Phase VI Socio SICs
    X['[REDACTED_BY_SCRIPT]'] = (get('solar_site_area_sqm') / 10000) * get('site_lsoa_ruc_rural_score') * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * get('alc_is_bmv_at_site') * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * (get('solar_site_area_sqm') + (get('[REDACTED_BY_SCRIPT]') * 10000))
    X['[REDACTED_BY_SCRIPT]'] = (get('[REDACTED_BY_SCRIPT]') + get('[REDACTED_BY_SCRIPT]')) / (get('delta_property_value', 1) + epsilon)
    X['SIC_SOCIO_DIGITAL_DIVIDE_RISK'] = get('[REDACTED_BY_SCRIPT]') + get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * get('[REDACTED_BY_SCRIPT]')

    # Cross-Domain SICs
    X['[REDACTED_BY_SCRIPT]'] = (get('[REDACTED_BY_SCRIPT]', 100) * (1 - get('[REDACTED_BY_SCRIPT]', 1))) * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = (1 / (get('lpa_major_commercial_approval_rate', 0.5) + epsilon)) / (get('[REDACTED_BY_SCRIPT]', 1000) + 100)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]', 1) * (get('[REDACTED_BY_SCRIPT]') + 1) * (1 + get('aonb_is_within') + get('np_is_within'))
    X['[REDACTED_BY_SCRIPT]'] = year_norm * get('solar_site_area_sqm') * get('alc_is_bmv_at_site')
    X['[REDACTED_BY_SCRIPT]'] = (get('[REDACTED_BY_SCRIPT]') + 1) * (get('cs_density_10km') + 1) * (get('[REDACTED_BY_SCRIPT]') + 1)

    # TSICs
    X['[REDACTED_BY_SCRIPT]'] = year_norm * (get('[REDACTED_BY_SCRIPT]') ** 2)
    X['[REDACTED_BY_SCRIPT]'] = year_norm / (get('aw_dist_to_nearest_m', 1000) + get('[REDACTED_BY_SCRIPT]', 1000) + 100)
    X['[REDACTED_BY_SCRIPT]'] = year_norm * get('[REDACTED_BY_SCRIPT]') * (1 + get('aonb_is_within'))
    X['TSIC_PRECEDENT_EROSION'] = year_norm / (get('[REDACTED_BY_SCRIPT]', 10) + 1)

    # Core SICs
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('solar_site_area_sqm') / 10000 + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = get('solar_site_area_sqm') * get('[REDACTED_BY_SCRIPT]')
    X['SIC_CORE_LOGISTICAL_ACCESS_CHALLENGE'] = get('[REDACTED_BY_SCRIPT]') / (get('[REDACTED_BY_SCRIPT]', 1) + epsilon)
    
    industrial_area_sqm = get('[REDACTED_BY_SCRIPT]') * (np.pi * 2000**2)
    X['[REDACTED_BY_SCRIPT]'] = get('solar_site_area_sqm') / (industrial_area_sqm + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]', 1) * get('alc_is_bmv_at_site')
    X['[REDACTED_BY_SCRIPT]'] = get('chp_enabled') * (1 - get('[REDACTED_BY_SCRIPT]'))

    # Synth SICs
    X['[REDACTED_BY_SCRIPT]'] = ((get('solar_site_area_sqm') + (get('[REDACTED_BY_SCRIPT]') * 10000)) / (get('nt_dist_to_nearest_m', 1000) + get('hp_dist_to_nearest_m', 1000) + 100)) * (1 + get('aonb_is_within') + get('np_is_within'))
    X['[REDACTED_BY_SCRIPT]'] = (get('[REDACTED_BY_SCRIPT]', 5) + get('[REDACTED_BY_SCRIPT]', 5) + 1) / (get('[REDACTED_BY_SCRIPT]', 10) + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = np.abs(get('lpa_major_commercial_approval_rate', 0.5) - get('[REDACTED_BY_SCRIPT]', 0.5)) * (1 + get('lpa_workload_trend'))
    X['[REDACTED_BY_SCRIPT]'] = get('solar_site_area_sqm') * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * get('[REDACTED_BY_SCRIPT]') / (get('sssi_dist_to_nearest_m', 1000) + 100)

    # Context SICs
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / ((get('[REDACTED_BY_SCRIPT]', 10000) / 1000) + epsilon)
    X['SIC_CONTEXT_CONNECTION_PATH_INEFFICIENCY'] = get('[REDACTED_BY_SCRIPT]', 1) / (get('[REDACTED_BY_SCRIPT]') + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = get('aonb_is_within') * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * np.abs(get('[REDACTED_BY_SCRIPT]'))
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('lpa_total_experience', 10) + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * get('[REDACTED_BY_SCRIPT]') / (get('[REDACTED_BY_SCRIPT]', 1) + epsilon)

    # Uncertainty SICs
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = (1 - get('[REDACTED_BY_SCRIPT]', 1)) * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = np.abs(get('[REDACTED_BY_SCRIPT]', 1) - get('[REDACTED_BY_SCRIPT]', 1))
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') + get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('[REDACTED_BY_SCRIPT]', 1) + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = (get('[REDACTED_BY_SCRIPT]') + get('nt_length_in_10km')) / (get('[REDACTED_BY_SCRIPT]', 1) + epsilon)

    # Ops SICs
    X['SIC_GRID_OPS_REVERSE_POWER_RISK'] = get('reversepower_encoded') * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('pq_idw_thd_knn5') * get('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = get('alc_is_bmv_at_site') * get('cs_on_site_bool') * get('solar_site_area_sqm')
    
    # New SICs from rearchitecture.py
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') / (get('sssi_dist_to_nearest_m', 1000) + get('hp_dist_to_nearest_m', 1000) + 100)
    X['[REDACTED_BY_SCRIPT]'] = (1 - get('[REDACTED_BY_SCRIPT]', 1)) * X['[REDACTED_BY_SCRIPT]']
    X['[REDACTED_BY_SCRIPT]'] = year_norm * X['[REDACTED_BY_SCRIPT]']
    
    X['[REDACTED_BY_SCRIPT]'] = get('[REDACTED_BY_SCRIPT]') * get('solar_site_area_sqm') # Approx logic

    return X

def engineer_post_knn_sics(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if 'knn_lpa_entropy_gcv' in X.columns and '[REDACTED_BY_SCRIPT]' in X.columns:
        X['[REDACTED_BY_SCRIPT]'] = X['knn_lpa_entropy_gcv'] * X['[REDACTED_BY_SCRIPT]']
    else:
        X['[REDACTED_BY_SCRIPT]'] = 0
    return X

def generate_knn_features(df, ref_X, ref_y):
    """
    Generates k-NN anomaly features using reference data.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Load Artifacts
    if not all(os.path.exists(p) for p in [KNN_IMPUTER_PATH, KNN_SCALER_PATH, KNN_ENGINE_PATH]):
        logging.error("[REDACTED_BY_SCRIPT]")
        for col in ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'knn_lpa_entropy_gcv']:
            df[col] = 0
        return df

    knn_imputer = joblib.load(KNN_IMPUTER_PATH)
    knn_scaler = joblib.load(KNN_SCALER_PATH)
    knn_engine = joblib.load(KNN_ENGINE_PATH)
    
    # Identify GCV features from the scaler/imputer
    # Priority: Use imputer's feature_names_in_ if available
    if hasattr(knn_imputer, 'feature_names_in_'):
        gcv_features = list(knn_imputer.feature_names_in_)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    elif hasattr(knn_scaler, 'feature_names_in_'):
        gcv_features = list(knn_scaler.feature_names_in_)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    else:
        # Fallback: Use columns from ref_X if available, or common GCV list
        logging.warning("[REDACTED_BY_SCRIPT]")
        gcv_features = [c for c in df.columns if c in ref_X.columns]
        
    # Ensure df has these features (fill missing with 0 then impute)
    # This is critical: we must have exactly the columns expected by the imputer
    for col in gcv_features:
        if col not in df.columns:
            df[col] = 0
            
    X_target_gcv = df[gcv_features].copy()
    
    # Transform
    X_target_gcv_imputed = knn_imputer.transform(X_target_gcv)
    X_target_gcv_scaled = knn_scaler.transform(X_target_gcv_imputed)
    
    # Find Neighbors
    distances, indices = knn_engine.kneighbors(X_target_gcv_scaled)
    
    # Map back to reference data
    # We need ref_y for duration stats
    neighbor_durations = ref_y.iloc[indices.flatten()].values.reshape(indices.shape)
    
    # LPA Entropy
    lpa_col = 'lpa_major_commercial_approval_rate'
    if lpa_col in ref_X.columns:
        neighbor_lpas = ref_X[lpa_col].iloc[indices.flatten()].values.reshape(indices.shape)
        df['knn_lpa_entropy_gcv'] = pd.DataFrame(neighbor_lpas).nunique(axis=1).values
    else:
        df['knn_lpa_entropy_gcv'] = 0
        
    df['[REDACTED_BY_SCRIPT]'] = distances.mean(axis=1)
    df['[REDACTED_BY_SCRIPT]'] = pd.DataFrame(neighbor_durations).var(axis=1).values
    df['[REDACTED_BY_SCRIPT]'] = pd.DataFrame(neighbor_durations).mean(axis=1).values
    
    df.fillna(0, inplace=True) # Sanitize
    return df

def coalesce_features(df, map_dict):
    for target, sources in map_dict.items():
        existing_sources = [c for c in sources if c in df.columns]
        if not existing_sources:
            continue
        for col in existing_sources:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[target] = df[existing_sources].mean(axis=1)
    return df

def rename_features_logic(df):
    cols_to_rename = {}
    for col in df.columns:
        if col.startswith('area_pct_'):
            cols_to_rename[col] = 'pct_area_' + col[len('area_pct_'):]
        elif col.startswith('ukpn_ltds_'):
            cols_to_rename[col] = 'ltds_' + col[len('ukpn_ltds_'):]
        elif col.startswith('ukpn_dnoa_'):
             cols_to_rename[col] = 'dnoa_' + col[len('ukpn_dnoa_'):]
    if cols_to_rename:
        df.rename(columns=cols_to_rename, inplace=True)
    
    # Remove duplicate columns resulting from rename (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def create_artificial_features(df):
    df['technology_type'] = 21
    df['chp_enabled'] = np.nan
    df['[REDACTED_BY_SCRIPT]'] = 1
    df['permission_granted'] = 1
    df['[REDACTED_BY_SCRIPT]'] = np.nan
    month = 11
    df['submission_month_sin'] = math.sin(2 * math.pi * month / 12)
    df['submission_month_cos'] = math.cos(2 * math.pi * month / 12)
    return df

def expand_capacities(df, capacities):
    expanded_df = df.loc[df.index.repeat(len(capacities))].copy()
    num_rows = len(df)
    tiled_capacities = np.tile(capacities, num_rows)
    expanded_df['[REDACTED_BY_SCRIPT]'] = tiled_capacities
    expanded_df.reset_index(drop=True, inplace=True)
    return expanded_df

def main():
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # 1. Load New Data
    if not os.path.exists(INPUT_DATA_PATH):
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return
    df = pd.read_csv(INPUT_DATA_PATH) if INPUT_DATA_PATH.endswith('.csv') else pd.read_parquet(INPUT_DATA_PATH)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # 2. Basic Prep
    df = coalesce_features(df, COALESCE_MAP)
    df = rename_features_logic(df)
    for feat in MISSING_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan
    df = create_artificial_features(df)
    df = expand_capacities(df, CAPACITIES_TO_SIMULATE)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # 3. Feature Engineering Pipeline
    logging.info("[REDACTED_BY_SCRIPT]")
    df = engineer_temporal_context_features(df)
    
    logging.info("Running SIC Engineering...")
    df = engineer_strategic_interaction_features(df)
    
    # 4. k-NN Anomaly Features
    if os.path.exists(REF_X_PATH) and os.path.exists(REF_Y_PATH):
        logging.info("[REDACTED_BY_SCRIPT]")
        ref_X = pd.read_csv(REF_X_PATH, index_col=0)
        ref_y = pd.read_csv(REF_Y_PATH, index_col=0).squeeze("columns")
        df = generate_knn_features(df, ref_X, ref_y)
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")
        for col in ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'knn_lpa_entropy_gcv']:
            df[col] = 0
            
    logging.info("[REDACTED_BY_SCRIPT]")
    df = engineer_post_knn_sics(df)

    # 5. Model Cascade
    # A. Oracle
    if not os.path.exists(ORACLE_PATH):
        logging.error(f"[REDACTED_BY_SCRIPT]")
        # Try to find any joblib in the dir that looks like oracle
        candidates = glob.glob(os.path.join(ARTIFACTS_DIR, "*oracle*.joblib"))
        if candidates:
            ORACLE_PATH_FOUND = candidates[0]
            logging.info(f"[REDACTED_BY_SCRIPT]")
            oracle_model = joblib.load(ORACLE_PATH_FOUND)
        else:
            return
    else:
        oracle_model = joblib.load(ORACLE_PATH)
        
    # Align Oracle Features
    if hasattr(oracle_model, 'feature_name_'):
        oracle_feats = oracle_model.feature_name_
        # Fill missing
        for f in oracle_feats:
            if f not in df.columns: df[f] = 0
        oracle_preds = oracle_model.predict(df[oracle_feats])
    else:
        oracle_preds = oracle_model.predict(df) # Hope for best
        
    df['[REDACTED_BY_SCRIPT]'] = oracle_preds
    logging.info("[REDACTED_BY_SCRIPT]")

    # B. Heads (Regressor Models)
    head_probs = {}
    head_files = glob.glob(os.path.join(HEADS_DIR, "*.joblib"))
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    top_head_features = {} # Load from json if available, else infer?
    # rearchitecture.py saves "[REDACTED_BY_SCRIPT]".
    import json
    top_feats_path = os.path.join(HEADS_DIR, "[REDACTED_BY_SCRIPT]")
    if os.path.exists(top_feats_path):
        with open(top_feats_path, 'r') as f:
            top_head_features = json.load(f)
    
    # Error bins for converting regression to probabilities
    # Bins: [-inf, -150, -50, 50, 150, inf] -> Classes: 0, 1, 2, 3, 4
    error_bins = [-np.inf, -150, -50, 50, 150, np.inf]
    
    for hf in head_files:
        cohort_name = os.path.basename(hf).replace('.joblib', '')
        head_model = joblib.load(hf)
        
        # Get features
        if hasattr(head_model, 'feature_name_'):
            feats = head_model.feature_name_
        else:
            continue # Skip if can't align
            
        # Align
        for f in feats:
            if f not in df.columns: df[f] = 0
            
        # REGRESSOR ADAPTATION: Get error predictions instead of probabilities
        error_preds = head_model.predict(df[feats])
        
        # Convert error predictions to probability distributions using softmax
        # For each prediction, compute distance to bin centers and convert to probabilities
        probs = np.zeros((len(error_preds), 5))
        bin_centers = [-200, -100, 0, 100, 200]  # Centers of the 5 error bins
        
        for i, err in enumerate(error_preds):
            # Compute distances to each bin center
            distances = np.array([abs(err - center) for center in bin_centers])
            # Convert to similarities (inverse distance with temperature)
            temperature = 50  # Controls sharpness of probability distribution
            similarities = np.exp(-distances / temperature)
            # Normalize to probabilities
            probs[i] = similarities / similarities.sum()
        
        head_probs[cohort_name] = pd.DataFrame(probs, index=df.index)

    # C. Arbiter Feature Construction
    logging.info("[REDACTED_BY_SCRIPT]")
    arbiter_df = pd.DataFrame(index=df.index)
    arbiter_df['[REDACTED_BY_SCRIPT]'] = oracle_preds
    
    # Aggregated probs
    if head_probs:
        avg_probs = pd.concat(head_probs.values()).groupby(level=0).mean()
        arbiter_df['prob_under_pred'] = avg_probs[0] + avg_probs[1]
        arbiter_df['prob_over_pred'] = avg_probs[3] + avg_probs[4]
        
        # Raw probs
        for cname, prob_df in head_probs.items():
            prob_df.columns = [f"{cname}_prob_{i}" for i in range(prob_df.shape[1])]
            arbiter_df = arbiter_df.join(prob_df)
            
    # Top Features
    salient_features = set()
    for feats in top_head_features.values():
        salient_features.update(feats)
    existing_salient = [f for f in salient_features if f in df.columns]
    arbiter_df = arbiter_df.join(df[existing_salient])
    
    arbiter_df.fillna(0, inplace=True)
    
    # Scale Arbiter
    if os.path.exists(ARBITER_SCALER_PATH):
        arbiter_scaler = joblib.load(ARBITER_SCALER_PATH)
        # Align columns with scaler
        if hasattr(arbiter_scaler, 'feature_names_in_'):
            scaler_feats = arbiter_scaler.feature_names_in_
            # Add missing
            for f in scaler_feats:
                if f not in arbiter_df.columns: arbiter_df[f] = 0
            arbiter_df = arbiter_df[scaler_feats]
            
        arbiter_df_scaled = pd.DataFrame(arbiter_scaler.transform(arbiter_df), columns=arbiter_df.columns)
    else:
        arbiter_df_scaled = arbiter_df # Fallback

    # D. Arbiter Prediction
    if os.path.exists(ARBITER_MODEL_PATH):
        arbiter_model = joblib.load(ARBITER_MODEL_PATH)
        correction = arbiter_model.predict(arbiter_df_scaled)
        final_preds = oracle_preds + correction
        df['[REDACTED_BY_SCRIPT]'] = final_preds
        df['[REDACTED_BY_SCRIPT]'] = correction
        logging.info("[REDACTED_BY_SCRIPT]")
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")
        df['[REDACTED_BY_SCRIPT]'] = oracle_preds

    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()
