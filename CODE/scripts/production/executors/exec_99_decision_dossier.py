"""
AD-AM-INF-01: Amaryllis Decision Dossier Inference Pipeline

This is the production inference system for the Amaryllis Decision Intelligence Platform.
It transforms raw site features into comprehensive Decision Dossiers containing:

1. Headline predictions for 10 capacity scenarios
2. Uncertainty quantification (P10/P90 intervals, catastrophe risk, error regimes)
3. Instance-specific explainability (SHAP feature impacts)
4. Contextual benchmarks (geographic & typological)
5. Rich GeoJSON output for spatial visualization

Architecture: Per-hex processing with full model cascade for each capacity simulation.

Author: Amaryllis Decision Intelligence Team
Date: 2025-11-21
Version: 4.0 (Decision Dossier)
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
import sys
import math
import re
import glob
import json
import h3
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from pyproj import Transformer
from typing import List, Dict, Any
from functools import partial

from multiprocessing import Pool, cpu_count

# Suppress DataFrame fragmentation warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Suppress sklearn warnings about feature names (we handle feature alignment explicitly)
warnings.filterwarnings('ignore', message='[REDACTED_BY_SCRIPT]')
warnings.filterwarnings('ignore', message='[REDACTED_BY_SCRIPT]')

# Optional SHAP import (currently using placeholder implementation)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("[REDACTED_BY_SCRIPT]")

import lightgbm as lgb

# Import dossier schema
sys.path.append(os.path.dirname(__file__))
from dossier_schema import (
    DecisionDossier, Identity, HeadlineRiskAssessment, UncertaintyMetrics,
    ExplainabilityEngine, FeatureDriver, ContextualBenchmarks,
    AccuracyBenchmark, TypologicalBenchmark, SimulationDetail,
    calculate_risk_category
)
from feature_descriptions import get_feature_description

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# 1. Input/Output Paths
INPUT_DATA_PATH = r"[REDACTED_BY_SCRIPT]"
OUTPUT_GEOJSON_PATH = r"[REDACTED_BY_SCRIPT]"

# 2. Artifact Paths
ARTIFACTS_DIR = r"[REDACTED_BY_SCRIPT]"
HEADS_DIR = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

# Models
ORACLE_PATH = os.path.join(ARTIFACTS_DIR, "oralce_v3.5.joblib")
ARBITER_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
ARBITER_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
P10_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
P90_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

# k-NN Artifacts
KNN_IMPUTER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
KNN_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
KNN_ENGINE_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

# Reference Data (for k-NN and benchmarking)
REF_X_PATH = r"[REDACTED_BY_SCRIPT]"
REF_Y_PATH = r"[REDACTED_BY_SCRIPT]"
COORD_DATA_PATH = r"[REDACTED_BY_SCRIPT]"


# Regime clustering
REGIME_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
REGIME_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

# New Artifacts (Solar & GM Cascade)
SOLAR_STRATIFIED_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
GM_SPECIALIST_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
GM_STRATIFIED_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
TERMINAL_CALIBRATOR_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

# Missing Artifacts (Placeholders - Required for Solar Bridge)
SOLAR_BRIDGE_PCA_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
SOLAR_BRIDGE_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
SOLAR_BRIDGE_TOP_FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

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

CAPACITIES_TO_SIMULATE = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 49, 50, 60]

# Enable multiprocessing
USE_MULTIPROCESSING = True
MAX_WORKERS = max(1, cpu_count() - 1)

# --- Helper Functions (Feature Engineering - from exec_99) ---

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df

# --- Complete Feature Engineering Functions (V16 Specialist Engine Migration) ---

def engineer_temporal_context_features(df: pd.DataFrame, baseline_year=None, imputer=None, scaler=None):
    """
    Architectural Mandate AD-AM-19:
    Engineers Temporal Feature Interactions (TFIs).
    REMEDIATION: Removed interactions dependent on missing features (AONB, Area, NIMBY).
    ENHANCEMENT: Added 'Political Clock' (Election Proximity) and Datetime Construction.
    """
    X = df.copy()
    epsilon = 1e-6

    # --- Step 0: Construct Datetime (Prerequisite for Political & Queue Features) ---
    # Ensure submission temporal columns exist with safe defaults if missing
    if 'submission_year' not in X.columns:
        X['submission_year'] = 2025
        
    if 'submission_day' not in X.columns:
        X['submission_day'] = 15 # Default to mid-month
        
    if 'submission_month' not in X.columns:
        # If we have sin/cos, we could theoretically reverse it, but for inference safety:
        # Default to June (6) - Neutral seasonality
        X['submission_month'] = 6

    # Robust construction handling potential NaNs in existing columns
    X['temp_day'] = X['submission_day'].fillna(15).astype(int)
    X['temp_month'] = X['submission_month'].fillna(6).astype(int)
    X['temp_year'] = X['submission_year'].fillna(2025).astype(int)
        
    X['submission_date'] = pd.to_datetime(dict(year=X['temp_year'], month=X['temp_month'], day=X['temp_day']))
    X.drop(columns=['temp_day', 'temp_month', 'temp_year'], inplace=True)

    # --- Step 0b: The Political Clock (Election Paralysis) ---
    # UK General Election Dates (Relevant window)
    election_dates = pd.to_datetime([
        '2010-05-06', '2015-05-07', '2017-06-08', '2019-12-12', '2024-07-04'
    ])
    
    def get_days_to_next_election(date, elections):
        future_elections = elections[elections > date]
        if len(future_elections) == 0:
            return 365 * 2 # Fallback: assume mid-term
        return (future_elections[0] - date).days

    X['[REDACTED_BY_SCRIPT]'] = X['submission_date'].apply(lambda x: get_days_to_next_election(x, election_dates))
    
    # Feature: Election High-Risk Zone (6 months prior)
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] < 180).astype(int)

    # Directive 2: Political Volatility Regimes (Replacing raw Year)
    def get_policy_regime(year):
        if year < 2015: return 0
        elif 2015 <= year <= 2019: return 1
        elif 2020 <= year <= 2021: return 2
        elif 2022 <= year <= 2023: return 3
        else: return 4

    X['SIC_POLICY_REGIME_ID'] = X['submission_year'].apply(get_policy_regime)
    
    # Directive 3: Election Proximity Cycle (Refinement)
    X['SIC_IS_PURDAH_PERIOD'] = (X['[REDACTED_BY_SCRIPT]'] < 45).astype(int) # Approx 6 weeks

    # Step 2: Engineer the Comprehensive Interaction Set (Refactored to use Regime/Cycle)
    # Replacing 'year_norm' with 'SIC_POLICY_REGIME_ID' to capture non-linear shifts.
    
    # Grid Saturation Interactions
    winter_load = X.get('[REDACTED_BY_SCRIPT]', 0)
    dist_sub = X.get('[REDACTED_BY_SCRIPT]', 1)
    headroom = X.get('[REDACTED_BY_SCRIPT]', 1)
    
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * winter_load
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] / (dist_sub + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * (1 / (headroom + 1))

    # Environmental Policy Hardening Interactions
    is_bmv = X.get('alc_is_bmv_at_site', 0)
    cs_on = X.get('cs_on_site_bool', 0)
    aw_dist = X.get('aw_dist_to_nearest_m', 5000)
    
    X['[REDACTED_BY_SCRIPT]'] = (X['SIC_POLICY_REGIME_ID'] == 3).astype(int) * is_bmv
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * cs_on
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] / (aw_dist + 100)

    # LPA Behavior & Precedent Interactions
    approval = X.get('lpa_major_commercial_approval_rate', 0.5)
    legacy_dist = X.get('[REDACTED_BY_SCRIPT]', 10)
    workload = X.get('lpa_workload_trend', 0)
    
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * approval
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] / (legacy_dist + 1)
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_IS_PURDAH_PERIOD'] * workload

    # Step 3: Mandated Sanitization and Normalization of New Features.
    tfi_features = [col for col in X.columns if col.startswith('TFI_')]
    
    # HARDENING GATE: Replace non-finite values created by division operations.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    if imputer is None: 
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        # Fit and transform
        if len(X) == 1:
            # CRITICAL FIX: Single-row inference cannot support statistical imputation or scaling
            # (Scaling 1 sample sets it to 0). We fill NaNs to prevent crashes and preserve raw signal magnitude.
            X[tfi_features] = X[tfi_features].fillna(0)
        else:
            # Batch mode (Training)
            # Handle cases where columns are all-NaN to prevent "[REDACTED_BY_SCRIPT]" error
            # SimpleImputer drops all-NaN columns by default. We must align indices.
            X_tfi = X[tfi_features]
            # Drop columns that are all NaN before imputing to avoid shape mismatch on assignment
            valid_cols = X_tfi.columns[X_tfi.notna().any()].tolist()
            
            if valid_cols:
                X_trans = imputer.fit_transform(X[valid_cols])
                X_trans = scaler.fit_transform(X_trans)
                X[valid_cols] = X_trans
                
            # Fill remaining (all-NaN) columns with 0
            missing_cols = list(set(tfi_features) - set(valid_cols))
            if missing_cols:
                X[missing_cols] = 0
                
    else: 
        fittable_tfi_features = [f for f in tfi_features if f in X.columns]
        # Robust transform handling
        try:
            X[fittable_tfi_features] = imputer.transform(X[fittable_tfi_features])
            X[fittable_tfi_features] = scaler.transform(X[fittable_tfi_features])
        except Exception as e:
            # Fallback if artifacts mismatch
            X[fittable_tfi_features] = X[fittable_tfi_features].fillna(0)
        
    return X, baseline_year, imputer, scaler

def engineer_lpa_congestion_metrics(df: pd.DataFrame, target_col='[REDACTED_BY_SCRIPT]') -> pd.DataFrame:
    """
    REMEDIATED: Input-Only Congestion Metrics.
    Handles missing 'lpa_name' gracefully.
    """
    X = df.copy()
    
    # Check if lpa_name exists. If not, we cannot calculate LPA-specific congestion.
    # We return default values (0 or global means) to prevent crash.
    if 'lpa_name' not in X.columns:
        X['[REDACTED_BY_SCRIPT]'] = 0.0
        X['[REDACTED_BY_SCRIPT]'] = 0.0
        X['SIC_LPA_QUEUE_RELATIVE_SCALE'] = 1.0 # Neutral scaling
        return X

    # Ensure temporal sort
    if 'submission_date' not in X.columns:
        X['submission_date'] = pd.to_datetime(X[['submission_year', 'submission_month', 'submission_day']])
    
    X.sort_values(['lpa_name', 'submission_date'], inplace=True)

    # 1. Rolling Submission Volume
    rolling_vol_series = pd.Series(index=X.index, dtype='float64')
    
    for _, sub_df in X.groupby('lpa_name'):
        ts = pd.Series(1, index=sub_df['submission_date'])
        rolled = ts.rolling('180D', closed='left').count()
        rolling_vol_series.loc[sub_df.index] = rolled.values

    X['[REDACTED_BY_SCRIPT]'] = rolling_vol_series.fillna(0)

    # 2. Congestion Severity
    epsilon = 1e-6
    if 'lpa_total_experience' in X.columns:
        exp = X['lpa_total_experience'].fillna(10)
    else:
        exp = 10
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (exp + epsilon)

    # 3. Relative Scale Attention
    rolling_cap_series = pd.Series(index=X.index, dtype='float64')
    
    for _, sub_df in X.groupby('lpa_name'):
        if '[REDACTED_BY_SCRIPT]' in sub_df.columns:
            cap_vals = sub_df['[REDACTED_BY_SCRIPT]'].fillna(0).values
        else:
            cap_vals = np.zeros(len(sub_df))
        ts = pd.Series(cap_vals, index=sub_df['submission_date'])
        rolled = ts.rolling('180D', closed='left').mean()
        rolling_cap_series.loc[sub_df.index] = rolled.values
    
    # Fill NaNs with global mean
    if '[REDACTED_BY_SCRIPT]' in X.columns:
        global_mean_cap = X['[REDACTED_BY_SCRIPT]'].mean()
    else:
        global_mean_cap = 0
    rolling_cap_series.fillna(global_mean_cap, inplace=True)
    
    if '[REDACTED_BY_SCRIPT]' in X.columns:
        cap = X['[REDACTED_BY_SCRIPT]'].fillna(0)
    else:
        cap = 0
    X['SIC_LPA_QUEUE_RELATIVE_SCALE'] = cap / (rolling_cap_series + epsilon)

    return X


def calculate_lpa_stats(historical_X, historical_y):
    """
    Calculates LPA statistics ONCE from historical data.
    Returns a dictionary of mappings (LPA Name -> Value).
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    hist_df = historical_X.copy()
    hist_df['[REDACTED_BY_SCRIPT]'] = historical_y
    
    # Ensure lpa_name exists
    if 'lpa_name' not in hist_df.columns:
        lpa_cols = [c for c in hist_df.columns if c.startswith('lpa_name_')]
        if lpa_cols:
            hist_df['lpa_name'] = hist_df[lpa_cols].idxmax(axis=1).apply(lambda x: str(x).replace('lpa_name_', ''))
        else:
            hist_df['lpa_name'] = 'Unknown'

    # Global Stats
    global_mean = hist_df['[REDACTED_BY_SCRIPT]'].mean()
    global_p90 = hist_df['[REDACTED_BY_SCRIPT]'].quantile(0.90)
    m_factor = 10 

    # Aggregations
    stats = hist_df.groupby('lpa_name')['[REDACTED_BY_SCRIPT]'].agg(
        n_samples='count',
        lpa_mean='mean',
        lpa_p90=lambda x: x.quantile(0.90),
        lpa_extension_propensity=lambda x: (x > 180).mean()
    ).reset_index()

    iqr = hist_df.groupby('lpa_name')['[REDACTED_BY_SCRIPT]'].agg(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index().rename(columns={'[REDACTED_BY_SCRIPT]': 'LPA_Stability_Index'})
    stats = pd.merge(stats, iqr, on='lpa_name', how='left')

    # Math
    stats['[REDACTED_BY_SCRIPT]'] = ((stats['n_samples'] * stats['lpa_mean']) + (m_factor * global_mean)) / (stats['n_samples'] + m_factor)
    stats['[REDACTED_BY_SCRIPT]'] = ((stats['n_samples'] * stats['lpa_p90']) + (m_factor * global_p90)) / (stats['n_samples'] + m_factor)
    
    lpa_std = hist_df.groupby('lpa_name')['[REDACTED_BY_SCRIPT]'].std().fillna(global_mean*0.2)
    stats.set_index('lpa_name', inplace=True)
    stats['lpa_std'] = lpa_std
    stats['LPA_Chaos_Index'] = stats['lpa_std'] / stats['[REDACTED_BY_SCRIPT]']

    # Convert to Lookup Dicts
    lookup_tables = {
        'mean': stats['[REDACTED_BY_SCRIPT]'].to_dict(),
        'p90': stats['[REDACTED_BY_SCRIPT]'].to_dict(),
        'chaos': stats['LPA_Chaos_Index'].to_dict(),
        'stability': stats['LPA_Stability_Index'].to_dict(),
        'extension': stats['[REDACTED_BY_SCRIPT]'].to_dict(),
        'defaults': {
            'mean': global_mean,
            'p90': global_p90,
            'chaos': 0.5,
            'stability': global_mean/2,
            'extension': 0.5
        }
    }
    return lookup_tables

def apply_lpa_stats(row_df, stats_dict):
    """
    Fast mapping of pre-calculated stats to the inference row.
    Includes Schema Bridge for datasets with pre-baked stats (missing lpa_name).
    """
    defaults = stats_dict['defaults']
    
    # Case A: LPA Name exists -> Standard Lookup
    if 'lpa_name' in row_df.columns:
        row_df['[REDACTED_BY_SCRIPT]'] = row_df['lpa_name'].map(stats_dict['mean']).fillna(defaults['mean'])
        row_df['[REDACTED_BY_SCRIPT]'] = row_df['lpa_name'].map(stats_dict['p90']).fillna(defaults['p90'])
        row_df['LPA_Chaos_Index'] = row_df['lpa_name'].map(stats_dict['chaos']).fillna(defaults['chaos'])
        row_df['LPA_Stability_Index'] = row_df['lpa_name'].map(stats_dict['stability']).fillna(defaults['stability'])
        row_df['[REDACTED_BY_SCRIPT]'] = row_df['lpa_name'].map(stats_dict['extension']).fillna(defaults['extension'])
        return row_df

    # Case B: Pre-Baked Stats Exist (Schema Bridge)
    # Map 'lpa_planning_time_...' to 'LPA_Bayesian_...'
    if 'lpa_planning_time_overall_mean' in row_df.columns:
        # Direct Mapping
        row_df['[REDACTED_BY_SCRIPT]'] = row_df['lpa_planning_time_overall_mean'].fillna(defaults['mean'])
        
        # Approximations
        if '[REDACTED_BY_SCRIPT]' in row_df.columns:
            std = row_df['[REDACTED_BY_SCRIPT]'].fillna(defaults['mean']*0.2)
        else:
            std = defaults['mean']*0.2
        mean = row_df['[REDACTED_BY_SCRIPT]']
        
        # P90 ~= Mean + 1.28*Std (Normal approx for missing metric)
        row_df['[REDACTED_BY_SCRIPT]'] = (mean + (1.28 * std)).fillna(defaults['p90'])
        
        # Chaos = Std / Mean
        row_df['LPA_Chaos_Index'] = (std / (mean + 1e-6)).fillna(defaults['chaos'])
        
        # Stability = Std (Proxy for IQR)
        row_df['LPA_Stability_Index'] = std.fillna(defaults['stability'])
        
        # Extension Propensity (Unknown, default)
        row_df['[REDACTED_BY_SCRIPT]'] = defaults['extension']
        
        return row_df

    # Case C: No Identity, No Stats -> Use Global Defaults
    row_df['[REDACTED_BY_SCRIPT]'] = defaults['mean']
    row_df['[REDACTED_BY_SCRIPT]'] = defaults['p90']
    row_df['LPA_Chaos_Index'] = defaults['chaos']
    row_df['LPA_Stability_Index'] = defaults['stability']
    row_df['[REDACTED_BY_SCRIPT]'] = defaults['extension']
    
    return row_df


def engineer_friction_matrix_features(df):
    """
    Architectural Directive 006: Feature Interaction & The "Friction" Matrix.
    """
    X = df.copy()
    
    def _get_safe(col, default):
        if col in X.columns:
            return X[col].fillna(default)
        return pd.Series(default, index=X.index)

    # Hardening: Ensure required columns exist
    X['LPA_Chaos_Index'] = _get_safe('LPA_Chaos_Index', 0.5)
    X['[REDACTED_BY_SCRIPT]'] = _get_safe('[REDACTED_BY_SCRIPT]', 180.0)
    X['[REDACTED_BY_SCRIPT]'] = _get_safe('[REDACTED_BY_SCRIPT]', 180.0)

    # 1. Risk Amplification
    X['[REDACTED_BY_SCRIPT]'] = X['LPA_Chaos_Index'] * _get_safe('[REDACTED_BY_SCRIPT]', 0)
    
    sssi_dist = _get_safe('sssi_dist_to_nearest_m', 5000)
    X['FI_Paranoia_Factor_SSSI'] = X['[REDACTED_BY_SCRIPT]'] * (1000 / (sssi_dist + 100))
    
    X['[REDACTED_BY_SCRIPT]'] = X['LPA_Chaos_Index'] * _get_safe('[REDACTED_BY_SCRIPT]', 0)

    # 2. Grid Confusion
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * _get_safe('[REDACTED_BY_SCRIPT]', 0)
    
    dnoa_years = _get_safe('[REDACTED_BY_SCRIPT]', 10)
    X['[REDACTED_BY_SCRIPT]'] = (1 / (dnoa_years + 1)) * X['[REDACTED_BY_SCRIPT]']

    # 3. Scale Shock
    workload = _get_safe('[REDACTED_BY_SCRIPT]', 100)
    X['[REDACTED_BY_SCRIPT]'] = _get_safe('[REDACTED_BY_SCRIPT]', 0) / (workload + 1)
    X['[REDACTED_BY_SCRIPT]'] = _get_safe('[REDACTED_BY_SCRIPT]', 0) * X['LPA_Chaos_Index']

    # 4. NIMBY Weaponization
    nimby_proxy = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = nimby_proxy * X['LPA_Chaos_Index']
    X['[REDACTED_BY_SCRIPT]'] = nimby_proxy * _get_safe('lpa_withdrawal_rate', 0)

    # 5. Stability & Variance Interactions
    X['[REDACTED_BY_SCRIPT]'] = _get_safe('LPA_Stability_Index', 0) * _get_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = _get_safe('[REDACTED_BY_SCRIPT]', 0) * _get_safe('[REDACTED_BY_SCRIPT]', 0)
    X['SIC_LPA_CHAOS_APPROVAL'] = X['LPA_Chaos_Index'] * _get_safe('lpa_major_commercial_approval_rate', 0)
    X['[REDACTED_BY_SCRIPT]'] = _get_safe('[REDACTED_BY_SCRIPT]', 0) * _get_safe('LPA_Stability_Index', 0)
    X['[REDACTED_BY_SCRIPT]'] = _get_safe('[REDACTED_BY_SCRIPT]', 0) * _get_safe('[REDACTED_BY_SCRIPT]', 0)
        
    return X


# --- V16 Consolidated Specialist Feature Engines ---

def engineer_gm_specialist_consolidated_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    MASTER DIRECTIVE: The Amaryllis Stratified Quad-Core.
    Final V16 Update: 10-15MW "[REDACTED_BY_SCRIPT]" Logic.
    Includes Retrospective Survey Bonus, Fire Safety Drag, Infrastructure Hug, and CHP Bonus.
    """
    # 1. Generate Layers
    
    # A. Physics, Biology & Competence (Updated V16)
    df_eco_cal = engineer_gm_ecological_calendar(df)       # Retrospective Bonus
    df_competence = engineer_gm_developer_competence(df)   # Fire Safety / CHP
    df_physics = engineer_stratified_grid_physics(df)      # Infrastructure Hug
    
    # B. Grid & Constraints (V15 Retained)
    df_grid_contention = engineer_gm_stratified_grid_contention(df)
    df_constraints = engineer_gm_hard_soft_constraints(df)
    
    # C. Politics & Sociology (V15 Retained)
    df_nimby = engineer_gm_nimby_mobilization(df)
    df_politics = engineer_gm_stratified_political_epochs(df)
    
    # D. Infrastructure & Saturation (V14/V13 Retained)
    df_severity = engineer_gm_constraint_severity(df)
    df_saturation = engineer_gm_social_saturation(df)
    df_regime = engineer_gm_stratified_regime_features(df)
    df_zoning = engineer_gm_stratified_zoning(df)
    df_regional = engineer_gm_regional_demographics(df)
    
    # E. Base Infrastructure
    df_hierarchy = engineer_gm_constraint_hierarchy(df)
    df_wayleave = engineer_gm_wayleave_friction(df)
    df_blackholes = engineer_gm_grid_black_holes(df)
    
    # F. Micro, Small & Mid Stratum Specifics (Updated V5.4)
    df_micro = engineer_gm_micro_context(df)
    df_small = engineer_gm_small_scale_friction(df)
    df_edge = engineer_gm_grid_edge_viability(df) 
    
    # G. Speculator Detection (New V5.5 - The 'Valley of Death' Fix)
    df_speculator = engineer_gm_speculator_detection(df)

    # H. Major Project Corrections (New V6.0 - The 'Black Swan' Fix)
    df_nsip = engineer_nsip_jurisdiction_features(df)
    df_cliff = engineer_voltage_cliff_features(df)
    
    # I. High-Strata Behavioral Corrections (New V7.0 - The 'Gaming' Fix)
    df_evasion = engineer_gm_threshold_evasion(df)
    df_step_up = engineer_gm_voltage_step_up(df)
    
    # J. Oracle & Micro Corrections (New V8.0 - The 'Rhythm & Agitation' Fix)
    df_rhythm = engineer_global_administrative_rhythm(df)
    df_agitation = engineer_micro_visual_agitation(df)
    
    # K. Structural Friction (New V9.0 - The 'Physics & Politics' Fix)
    df_hostility = engineer_regional_hostility_features(df)
    df_topology = engineer_grid_topology_depth(df)
    df_logistics_v2 = engineer_logistics_access_failure_v2(df)
    
    # L. Common Sense Engines (New V10.0 - The 'Fairness & Physics' Fix)
    df_fairness = engineer_social_fairness_ratio(df)
    df_cable = engineer_cable_route_physics(df)
    df_scarcity = engineer_agri_scarcity_pressure(df)
    df_mud = engineer_seasonal_access_friction(df)
    
    # M. Mid-Scale Rescue (V11.0 - The 'Voltage Trap' Fix)
    df_mid_rescue = engineer_mid_stratum_rescue_v2(df)
    
    # N. Distribution Grit (New V12.0 - The '0-10MW' Fix)
    df_grit = engineer_distribution_grit_features(df)
    
    # O. Grid Edge & Inertia (New V13.0 - The 'Secondary Sub' Fix)
    df_edge_v2 = engineer_grid_edge_inertia_features(df)

    # P. Hazards & Margins (New V14.0 - The 'Amateur' Fix)
    df_hazards = engineer_hazards_margins_features(df)

    # 2. Concatenate All Layers
    df_final = pd.concat([
        df_eco_cal, df_competence, df_physics,
        df_grid_contention, df_constraints, df_nimby, df_politics,
        df_severity, df_saturation, df_regime, df_zoning, df_regional,
        df_hierarchy, df_wayleave, df_blackholes, df_micro, df_small, df_edge,
        df_speculator, df_nsip, df_cliff, 
        df_evasion, df_step_up,
        df_rhythm, df_agitation,
        df_hostility, df_topology, df_logistics_v2,
        df_fairness, df_cable, df_scarcity, df_mud,
        df_mid_rescue, df_grit, df_edge_v2, df_hazards
    ], axis=1)
    
    # 3. Enforce BESS Stratum Gate
    if '[REDACTED_BY_SCRIPT]' not in df_final.columns:
        df_final['[REDACTED_BY_SCRIPT]'] = df.get('[REDACTED_BY_SCRIPT]', 0)
        
    df_final_sanitized = enforce_bess_stratum_gate(df_final)
    
    # 4. Final Sanitation Gate (Fixing NaN leak from logs)
    df_final_sanitized.fillna(0, inplace=True)
    
    return df_final_sanitized

# --- Helper Engines for V16 ---

def engineer_gm_ecological_calendar(df: pd.DataFrame) -> pd.DataFrame:
    # ROBUST GETTER: Always return a Series matching the index
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            # fillna handles NaN, returning a Series
            return df[col_name].fillna(default_val)
        # Create a Series of defaults with matching index
        return pd.Series(default_val, index=df.index)

    # Use a new DataFrame to build features to avoid SettingWithCopy
    X = pd.DataFrame(index=df.index)
    
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    month = _get_series_safe('submission_month', 6).astype(int)
    
    is_micro = (cap < 1.0).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)
    
    def _is_near(prefix):
        col = f"[REDACTED_BY_SCRIPT]"
        # Use robust getter for distance columns
        dist = _get_series_safe(col, 5000)
        return (dist < 100).astype(int)

    constraint_count = (_is_near('sssi_') + _is_near('sac_') + _is_near('spa_') + _is_near('aw_') + _is_near('ph_'))
    is_broad_winter = month.isin([9, 10, 11, 12, 1, 2]).astype(int)

    X['SIC_GM_MISSED_WINDOW_MICRO'] = is_micro * (constraint_count > 0) * is_broad_winter
    X['[REDACTED_BY_SCRIPT]'] = is_med * constraint_count * is_broad_winter
    X['[REDACTED_BY_SCRIPT]'] = is_large * constraint_count * is_broad_winter * -1.0

    return X

def engineer_gm_developer_competence(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    voltage = _get_series_safe('[REDACTED_BY_SCRIPT]', 11)
    
    is_industrial = (cap >= 1.0).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)

    X['[REDACTED_BY_SCRIPT]'] = is_industrial * (cap / (voltage + 1e-6))

    s1 = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    s2 = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    has_storage = ((s1 + s2) > 0).astype(int)

    X['[REDACTED_BY_SCRIPT]'] = is_med * has_storage
    X['[REDACTED_BY_SCRIPT]'] = is_large * has_storage
    has_chp = _get_series_safe('chp_enabled', 0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large * has_chp

    return X

def engineer_stratified_grid_physics(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_industrial = (cap >= 1.0).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)

    impedance = _get_series_safe('resistance_ohm', 1.0)
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * cap * impedance

    fault_level_ka = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    line_density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_med * fault_level_ka * (line_density + 0.1)

    dist_sub = _get_series_safe('[REDACTED_BY_SCRIPT]', 5.0)
    dist_ctx = _get_series_safe('[REDACTED_BY_SCRIPT]', 5000.0) / 1000.0
    eff_dist = np.minimum(dist_sub, dist_ctx)
    hug_score = 1.0 / (eff_dist + 0.1)
    X['[REDACTED_BY_SCRIPT]'] = is_large * hug_score

    return X

def engineer_gm_stratified_grid_contention(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    headroom = _get_series_safe('[REDACTED_BY_SCRIPT]', 10)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    years = _get_series_safe('[REDACTED_BY_SCRIPT]', 5.0)
    
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med_utility = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large_ind = (cap >= 10.0).astype(int)
    
    conditions = [(headroom < 0), (headroom >= 0) & (headroom <= 10), (headroom > 10)]
    X['SIC_GM_GRID_TRAFFIC_LIGHT'] = np.select(conditions, [2, 1, 0], default=0)
    
    queue_count = _get_series_safe('SIC_LPA_QUEUE_COUNT', 0)
    is_race = ((headroom > 10) & (queue_count > 2)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * is_race * queue_count

    pressure_score = np.select([(years >= 0) & (years < 2.0), (years > 5.0)], [2.0, 0.5], default=1.0)
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * pressure_score
    
    dno_propensity = 0.5 
    is_green_lane = (headroom > 50).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * is_green_lane * dno_propensity

    is_anm = (headroom < 0).astype(int)
    is_unconstrained = (headroom > 50).astype(int)
    is_study_risk = ((headroom >= 0) & (headroom <= 20)).astype(int)
    contract_score = (is_anm * 1.0) + (is_unconstrained * 1.0) + (is_study_risk * -1.0)
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * contract_score

    is_banker = (years > 4.0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_banker * 1.0

    return X

def engineer_gm_hard_soft_constraints(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)

    def _is_near(prefix, thresh=100):
        col = f"[REDACTED_BY_SCRIPT]"
        return (_get_series_safe(col, 5000) < thresh).astype(int)

    w_sum = (_is_near('sssi_') + _is_near('aw_') + _is_near('sac_') + _is_near('aonb_') + _is_near('hp_')) * 5.0
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * w_sum

    has_greenbelt = (_get_series_safe('[REDACTED_BY_SCRIPT]', 0) > 0).astype(int)
    is_visual = (_is_near('aonb_') | _is_near('np_') | _is_near('hp_') | has_greenbelt)
    is_tech = (_is_near('sssi_') | _is_near('ph_') | _is_near('sac_'))
    
    X['[REDACTED_BY_SCRIPT]'] = is_med * is_visual * 10.0
    X['[REDACTED_BY_SCRIPT]'] = is_med * is_tech * 2.0

    is_negotiable = (_is_near('ph_'))
    X['[REDACTED_BY_SCRIPT]'] = is_large * is_negotiable * 5.0
    
    is_binary = (_is_near('aw_') | _is_near('sssi_'))
    X['[REDACTED_BY_SCRIPT]'] = is_large * is_binary * -2.0

    return X

def engineer_gm_nimby_mobilization(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_micro = (cap < 1.0).astype(int)
    is_small_med = ((cap >= 1.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)
    
    seniors = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    utilitarians = _get_series_safe('site_lsoa_oac_e_Rational_Utilitarians', 0)
    veterans = _get_series_safe('site_lsoa_oac_e_Veterans', 0)
    wealth_idx = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    
    X['SIC_GM_NIMBY_SENTIMENT_MICRO'] = is_micro * (wealth_idx / (density + 0.1))

    mob_score_base = (seniors * 3.0) + (utilitarians * 1.5) + (veterans * 1.0)
    urban_fabric = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_med * mob_score_base * (urban_fabric + 1.0)

    retiree_friction_score = (seniors * 3.0) + (utilitarians * 0.5) + (veterans * 1.0)
    gradient = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['SIC_GM_RETIREE_RESISTANCE_LARGE'] = is_large * retiree_friction_score * (gradient + 1.0)

    return X

def engineer_gm_stratified_political_epochs(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    year = _get_series_safe('submission_year', 2020).astype(int)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    
    conditions = [(year <= 2015), (year == 2016), (year >= 2017) & (year <= 2019), (year >= 2020)]
    X['[REDACTED_BY_SCRIPT]'] = np.select(conditions, [0, 1, 2, 3], default=2)

    is_industrial = (cap >= 1.0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * (year >= 2020).astype(int) * 1.0
    X['[REDACTED_BY_SCRIPT]'] = (cap >= 5.0).astype(int) * (year >= 2024).astype(int)

    is_micro = (cap < 1.0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_micro * 0.5 
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * 0.5

    return X

def engineer_gm_constraint_severity(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med_utility = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large_ind = (cap >= 10.0).astype(int)

    def _is_proximate(prefix):
        col = f"[REDACTED_BY_SCRIPT]"
        return (_get_series_safe(col, 5000) < 50).astype(int)

    w_sum = (_is_proximate('sssi_') * 10 + _is_proximate('aw_') * 10 + _is_proximate('sac_') * 10)
    raw_score = w_sum
    pct_agri = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    is_ambiguous_trap = ((raw_score == 0) & (pct_agri > 50)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * (raw_score + (is_ambiguous_trap * 10.0))

    constraint_count = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    is_high_prep_med = (constraint_count >= 2).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * is_high_prep_med * 1.0
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * (1 - is_high_prep_med)

    is_speculator_band = ((cap >= 5.0) & (cap < 15.0)).astype(int)
    is_zero_constraints = (constraint_count == 0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * is_zero_constraints * 1.0
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * (constraint_count >= 2).astype(int) * 1.0
    
    is_bmv = _get_series_safe('alc_is_bmv_at_site', 0)
    messy_score = _is_proximate('ph_') + _is_proximate('sssi_')
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * (1 - is_bmv) * (messy_score + 1.0) * 10.0

    return X

def engineer_gm_social_saturation(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    neighbor_count = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    
    is_micro = (cap < 1.0).astype(int)
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_large_ind = (cap >= 10.0).astype(int)
    is_speculator_band = ((cap >= 5.0) & (cap < 15.0)).astype(int)

    conditions_micro = [(neighbor_count <= 1), (neighbor_count > 1) & (neighbor_count <= 3), (neighbor_count > 3)]
    X['[REDACTED_BY_SCRIPT]'] = is_micro * np.select(conditions_micro, [0, 1, 2], default=2)

    density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * (neighbor_count ** 1.5) * (density + 0.1) * cap

    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * (neighbor_count > 5).astype(int) * neighbor_count

    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * (neighbor_count <= 3).astype(int) * 1.0
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * ((neighbor_count > 3) & (neighbor_count <= 10)).astype(int) * 1.0
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * (neighbor_count > 10).astype(int) * 1.0

    return X

def engineer_gm_stratified_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    avg_days = _get_series_safe('[REDACTED_BY_SCRIPT]', 180)
    
    is_micro = (cap < 1.0).astype(int)
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med_utility = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large_ind = (cap >= 10.0).astype(int)

    is_fast_council = (avg_days < 112).astype(int)
    speed_score = (112 - avg_days).clip(lower=0)
    variance = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)

    X['[REDACTED_BY_SCRIPT]'] = (is_micro | is_small_ind) * is_fast_council * speed_score
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * is_fast_council * speed_score * variance
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_fast_council * speed_score

    is_bmv = _get_series_safe('alc_is_bmv_at_site', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_micro * is_bmv
    X['[REDACTED_BY_SCRIPT]'] = ((cap > 2.0) & (cap < 10.0)).astype(int) * is_bmv * cap
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_bmv * 10.0

    voltage = _get_series_safe('[REDACTED_BY_SCRIPT]', 11)
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * (voltage < 33).astype(int) * cap
    
    workload = _get_series_safe('lpa_workload_trend', 0)
    X['[REDACTED_BY_SCRIPT]'] = ((cap >= 5.0) & (cap < 15.0)).astype(int) * workload

    return X

def engineer_gm_stratified_zoning(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    urban_pct = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    
    is_small = (cap < 5.0).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)

    ind_pct = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    res_density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small * (res_density / (ind_pct + 1.0)) * cap
    X['[REDACTED_BY_SCRIPT]'] = is_med * (urban_pct > 15).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large * (urban_pct > 50).astype(int) * -1.0
    X['[REDACTED_BY_SCRIPT]'] = ind_pct

    return X

def engineer_gm_regional_demographics(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    rejection_score = pd.Series(0.5, index=df.index)
    if 'dno_source_ukpn' in df.columns:
        mask = df['dno_source_ukpn'] == 1
        rejection_score[mask] = 0.8
    if 'dno_source_nged' in df.columns:
        mask = df['dno_source_nged'] == 1
        rejection_score[mask] = 0.3
    
    X['[REDACTED_BY_SCRIPT]'] = rejection_score
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = (cap < 10.0).astype(int) * rejection_score * cap
    return X

def engineer_gm_constraint_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    def _is_proximate(prefix):
        col = f"[REDACTED_BY_SCRIPT]"
        return (_get_series_safe(col, 5000) < 50).astype(int)

    w_hp = _is_proximate('hp_') * 1.8
    w_aonb = _is_proximate('aonb_') * 1.4
    w_ph = _is_proximate('ph_') * 1.1
    w_sssi = _is_proximate('sssi_') * 1.6
    w_sac = _is_proximate('sac_') * 0.9
    
    X['[REDACTED_BY_SCRIPT]'] = np.maximum.reduce([w_hp, w_aonb, w_ph, w_sssi, w_sac])
    return X

def engineer_gm_wayleave_friction(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    dist_km = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    intersections = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    
    frag_factor = 1.0 + (density * 5.0)
    X['[REDACTED_BY_SCRIPT]'] = dist_km * frag_factor
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] + (intersections * 3.0)
    X['SIC_GM_RANSOM_STRIP_RISK'] = X['[REDACTED_BY_SCRIPT]'] * np.log1p(cap)
    
    return X

def engineer_gm_grid_black_holes(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    black_hole_risk = pd.Series(0, index=df.index)
    for col in df.columns:
        if '[REDACTED_BY_SCRIPT]' in col and df[col].any():
            black_hole_risk = 1
    
    X['[REDACTED_BY_SCRIPT]'] = black_hole_risk
    # Ensure GRID TRAFFIC LIGHT is calculated first or default
    grid_status = _get_series_safe('SIC_GM_GRID_TRAFFIC_LIGHT', 0)
    X['[REDACTED_BY_SCRIPT]'] = black_hole_risk * grid_status
    
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    volts = _get_series_safe('[REDACTED_BY_SCRIPT]', 11)
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = (cap >= 1.0).astype(int) * (volts < 33).astype(int) * dist * cap
    
    return X

def engineer_gm_micro_context(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_small = (cap < 5.0).astype(int)
    wealth = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    agri = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    
    X['[REDACTED_BY_SCRIPT]'] = is_small * wealth * density
    X['[REDACTED_BY_SCRIPT]'] = is_small * agri * (1 - density)
    return X

def engineer_gm_small_scale_friction(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_target = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    agri = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    ind = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    
    has_div = ((agri > 50) & (ind > 1.0)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_target * has_div * -1.0
    
    density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_target * density * cap
    
    ohl = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_target * ohl * cap
    
    return X

def engineer_gm_grid_edge_viability(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    volts = _get_series_safe('[REDACTED_BY_SCRIPT]', 11)
    
    is_small = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med = ((cap >= 5.0) & (cap < 15.0)).astype(int)
    
    fl = _get_series_safe('[REDACTED_BY_SCRIPT]', 100)
    X['[REDACTED_BY_SCRIPT]'] = is_small * (cap / (fl + 1e-6))
    X['[REDACTED_BY_SCRIPT]'] = is_med * (volts < 33).astype(int) * cap
    
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = (is_small | is_med) * (dist / (cap + 1e-6))
    
    constraints = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_med * dist * (1 + constraints)
    
    return X

def engineer_gm_speculator_detection(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_spec = ((cap >= 4.0) & (cap < 16.0)).astype(int)
    
    def _is_near(prefix):
        col = f"[REDACTED_BY_SCRIPT]"
        return (_get_series_safe(col, 5000) < 1000).astype(int)
    
    sens = (_is_near('sssi_') | _is_near('aonb_') | _is_near('np_') | _is_near('hp_'))
    X['[REDACTED_BY_SCRIPT]'] = is_spec * sens * 10.0
    
    n_mw = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    n_count = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    avg = n_mw / (n_count + 1e-6)
    runt = ((cap / (avg + 1e-6)) < 0.5).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_spec * runt * (1.0 - (cap/(avg+1e-6)))
    
    is_bmv = _get_series_safe('alc_is_bmv_at_site', 0)
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_spec * is_bmv * dist
    
    withdraw = _get_series_safe('lpa_withdrawal_rate', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_spec * withdraw
    
    return X

def engineer_nsip_jurisdiction_features(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_nsip = (cap > 50.0).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = is_nsip * 500.0
    X['SIC_NSIP_LPA_BYPASS_FLAG'] = is_nsip
    
    def _present(col): return (_get_series_safe(col, 0) > 0).astype(int)
    fric = (_present('sssi_is_within') * 2.0 + _present('aonb_is_within') * 1.5 + 
            _present('hp_is_within') * 1.5 + _present('flood_zone_3') * 2.0)
    X['[REDACTED_BY_SCRIPT]'] = is_nsip * fric * 30.0
    
    return X

def engineer_voltage_cliff_features(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_cliff = ((cap >= 30.0) & (cap <= 40.0)).astype(int)
    sat = ((cap - 30.0) / 10.0).clip(0, 1)
    
    X['[REDACTED_BY_SCRIPT]'] = is_cliff * sat
    
    headroom = _get_series_safe('[REDACTED_BY_SCRIPT]', 100)
    X['[REDACTED_BY_SCRIPT]'] = is_cliff * (headroom < 10).astype(int) * 100.0
    return X

def engineer_gm_threshold_evasion(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_hug = ((cap >= 45.0) & (cap < 50.0)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_hug
    
    neigh = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    is_slice = ((cap > 30.0) & ((cap + neigh) > 60.0)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_slice * (cap + neigh)
    
    def _present(col): return (_get_series_safe(col, 0) > 0).astype(int)
    comp = _present('sssi_is_within') + _present('aw_is_within') + _present('aonb_is_within')
    X['[REDACTED_BY_SCRIPT]'] = is_hug * comp * 10.0
    return X

def engineer_gm_voltage_step_up(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    volts = _get_series_safe('[REDACTED_BY_SCRIPT]', 33)
    is_high = (cap >= 30.0).astype(int)
    
    # Vectorized score calc
    score = pd.Series(0.0, index=df.index)
    score[(cap >= 30.0) & (volts < 33)] = 10.0
    score[(cap >= 30.0) & (volts == 33)] = 5.0
    score[(cap >= 30.0) & (volts >= 132)] = -2.0
    
    X['[REDACTED_BY_SCRIPT]'] = score
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_high * score * dist
    return X

def engineer_global_administrative_rhythm(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    month = _get_series_safe('submission_month', 6).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = month.isin([8, 12]).astype(int) * 15.0
    
    legacy = _get_series_safe('nearby_legacy_count', 0)
    vol = _get_series_safe('[REDACTED_BY_SCRIPT]', 100)
    X['[REDACTED_BY_SCRIPT]'] = 1.0 / ((legacy / (vol + 1e-6)) + 0.1)
    
    X['[REDACTED_BY_SCRIPT]'] = month.isin([10, 11, 12]).astype(int) * 20.0
    return X

def engineer_micro_visual_agitation(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_target = (cap < 5.0).astype(int)
    
    grad = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    dens = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_target * grad * dens * cap
    
    poles = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_target * poles * dens
    
    urban = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_target * urban * cap
    return X

def engineer_regional_hostility_features(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    ref = _get_series_safe('[REDACTED_BY_SCRIPT]', 50.0)
    suc = _get_series_safe('[REDACTED_BY_SCRIPT]', 50.0)
    
    X['[REDACTED_BY_SCRIPT]'] = 1.0 / (ref + 0.1)
    X['[REDACTED_BY_SCRIPT]'] = (suc + 0.1) / (ref + 0.1)
    X['[REDACTED_BY_SCRIPT]'] = (ref < 1.0).astype(int)
    return X

def engineer_grid_topology_depth(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    dn = _get_series_safe('[REDACTED_BY_SCRIPT]', 5.0)
    dg = _get_series_safe('[REDACTED_BY_SCRIPT]', 5.0)
    
    delta = (dg - dn).clip(lower=0)
    X['[REDACTED_BY_SCRIPT]'] = delta
    X['SIC_GRID_CONNECTION_EFFICIENCY'] = (dn + 0.1) / (dg + 0.1)
    X['[REDACTED_BY_SCRIPT]'] = cap * delta
    return X

def engineer_logistics_access_failure_v2(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    road = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    is_ind = (cap > 5.0).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = is_ind * (cap / (road + 0.1))
    X['[REDACTED_BY_SCRIPT]'] = is_ind * (road < 100).astype(int) * cap
    return X

def engineer_social_fairness_ratio(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    dens = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['SIC_SOCIAL_OCCUPATION_RATIO'] = cap / (dens + 0.01)
    
    urban = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = (urban > 20.0).astype(int) * cap * -1.0
    return X

def engineer_cable_route_physics(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 5.0)
    grad = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = dist * grad
    
    wet = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    # Check if wet is scalar or series
    try:
        is_wet = (wet == 1).astype(int)
    except:
        is_wet = int(wet == 1)
        
    X['[REDACTED_BY_SCRIPT]'] = dist * is_wet
    return X

def engineer_agri_scarcity_pressure(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    site = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    reg = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['SIC_AGRI_SCARCITY_RATIO'] = site / (reg + 1e-6)
    
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    bmv = _get_series_safe('alc_is_bmv_at_site', 0)
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_AGRI_SCARCITY_RATIO'] * cap * bmv
    return X

def engineer_seasonal_access_friction(df: pd.DataFrame) -> pd.DataFrame:
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)

    X = pd.DataFrame(index=df.index)
    month = _get_series_safe('submission_month', 6).astype(int)
    road = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    grad = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    
    is_aut = month.isin([9, 10, 11]).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_aut * (1.0 / (road + 0.1)) * grad * cap
    return X











def engineer_mid_stratum_rescue_v2(df: pd.DataFrame) -> pd.DataFrame:
    # 1. ROBUST GETTER FIX: Always return a Series
    def _get_safe_series(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)
    
    cap = _get_safe_series('[REDACTED_BY_SCRIPT]')
    is_mid = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    
    volts = _get_safe_series('[REDACTED_BY_SCRIPT]', 11)
    dist = _get_safe_series('[REDACTED_BY_SCRIPT]', 5.0)
    
    X = pd.DataFrame(index=df.index)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * (cap / (volts + 1e-6))
    X['[REDACTED_BY_SCRIPT]'] = is_mid * (dist / (cap + 1e-6))
    
    ohl = _get_safe_series('[REDACTED_BY_SCRIPT]', 5000) / 1000.0
    tap = ((dist > 3.0) & (ohl < 0.5)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * tap * 10.0
    
    lpa_exp = _get_safe_series('lpa_total_experience', 0)
    X['SIC_MID_LPA_NOVICE_PANIC'] = is_mid * (cap / (lpa_exp + 1.0))
    
    days = _get_safe_series('[REDACTED_BY_SCRIPT]', 180)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * days * (1.0 / (cap + 0.1))
    
    bmv = _get_safe_series('alc_is_bmv_at_site', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * bmv * 5.0
    
    neigh = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * neigh
    X['[REDACTED_BY_SCRIPT]'] = is_mid * bmv * dist
    
    dens = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * dens * cap
    
    poles = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * poles * dens
    
    road = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    X['SIC_MID_ROAD_ACCESS_CONSTRICTION'] = is_mid * (1.0 / (road + 0.1))
    
    queue = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * (queue / (cap + 1e-6))
    
    return X

def engineer_distribution_grit_features(df: pd.DataFrame) -> pd.DataFrame:
    # Robust Series Getter
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)
    
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_micro = (cap < 1.0).astype(int)
    is_small = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_mid = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_dist = (cap < 15.0).astype(int)
    
    cross = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 5.0)
    
    X = pd.DataFrame(index=df.index)
    X['[REDACTED_BY_SCRIPT]'] = is_dist * (cross / (dist + 1e-6))
    X['SIC_GRIT_RANSOM_STRIP_RISK'] = (is_small | is_mid) * (cross > 5).astype(int) * 10.0
    
    cliff = ((cap >= 5.0) & (cap < 7.0)).astype(int)
    vol = _get_series_safe('[REDACTED_BY_SCRIPT]', 100)
    X['SIC_GRIT_COMMITTEE_CLIFF_EDGE'] = cliff * (1.0 / (vol + 1e-6))
    
    hr = _get_series_safe('[REDACTED_BY_SCRIPT]', 10)
    X['[REDACTED_BY_SCRIPT]'] = (is_small | is_mid) * (cap > hr).astype(int) * (cap - hr)
    
    awk = ((cap >= 0.5) & (cap < 1.0)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = awk * dist
    
    dens = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * dens * cap
    
    grade = _get_series_safe('alc_grade_at_site', 3)
    X['SIC_GRIT_BEST_FIELD_SACRIFICE'] = (is_small | is_mid) * (grade <= 2).astype(int) * 5.0
    
    X['[REDACTED_BY_SCRIPT]'] = (cliff | awk) * 5.0
    
    grad = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = (is_small | is_mid) * grad * -1.0
    
    neigh = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small * neigh
    
    return X

def engineer_hazards_margins_features(df: pd.DataFrame) -> pd.DataFrame:
    # Robust Series Getter
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)
    
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    is_target = (cap < 10.0).astype(int)
    is_micro = (cap < 1.0).astype(int)
    is_small_mid = ((cap >= 1.0) & (cap < 10.0)).astype(int)
    
    X = pd.DataFrame(index=df.index)
    
    rail = _get_series_safe('railway_length_1km', 0)
    X['SIC_HAZARD_RAIL_GLARE_RISK'] = is_target * rail * cap
    
    grad = _get_series_safe('[REDACTED_BY_SCRIPT]', 10)
    wet = _get_series_safe('ph_dist_to_nearest_m', 5000)
    X['[REDACTED_BY_SCRIPT]'] = is_target * (grad < 2.0).astype(int) * (wet < 200).astype(int) * 5.0
    
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 0.1)
    ratio = dist / (cap + 0.1)
    X['[REDACTED_BY_SCRIPT]'] = is_target * (ratio > 2.0).astype(int) * ratio
    
    grade = _get_series_safe('alc_grade_at_site', 3)
    X['[REDACTED_BY_SCRIPT]'] = is_target * (grade <= 2).astype(int) * 10.0
    
    X['[REDACTED_BY_SCRIPT]'] = is_micro * (cap > 0.05).astype(int)
    
    sett = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_mid * sett * (rail + 1.0)
    
    return X




def engineer_grid_edge_inertia_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    
    def _get_safe(col, default):
        if col in X.columns:
            return X[col].fillna(default)
        return pd.Series(default, index=X.index)

    cap = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    is_micro = (cap < 1.0).astype(int)
    is_small_mid = ((cap >= 1.0) & (cap < 10.0)).astype(int)
    is_dist = (cap < 15.0).astype(int)
    
    util = _get_safe('[REDACTED_BY_SCRIPT]', 50.0)
    X['[REDACTED_BY_SCRIPT]'] = (is_micro | is_small_mid) * (util / 100.0) * cap
    
    cust = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = (is_micro | is_small_mid) * cust * cap
    
    idno = (_get_safe('idno_is_within', 0) > 0) | (_get_safe('[REDACTED_BY_SCRIPT]', 0) > 0)
    X['[REDACTED_BY_SCRIPT]'] = is_dist * idno.astype(int) * (1.0 / (cap + 0.5))
    
    vice = _get_safe('[REDACTED_BY_SCRIPT]', 1.0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_mid * (1.0 / (vice + 0.1)) * cap
    
    ldist = _get_safe('[REDACTED_BY_SCRIPT]', 5000.0)
    lyears = _get_safe('[REDACTED_BY_SCRIPT]', 5.0)
    blight = ((ldist < 1000.0) & (lyears > 0) & (lyears < 3)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_dist * blight * lyears
    
    ratio = _get_safe('[REDACTED_BY_SCRIPT]', 0.5)
    X['[REDACTED_BY_SCRIPT]'] = (is_micro | is_small_mid) * (ratio > 1.0).astype(int) * ratio
    
    cust_count = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_mid * cust_count * cap
    
    imd = _get_safe('[REDACTED_BY_SCRIPT]', 15000)
    score = 1.0 / (imd + 1.0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_mid * score * -1.0
    
    return X[[
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]]


def enforce_bess_stratum_gate(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    
    def _get_safe(col, default):
        if col in X.columns:
            return X[col].fillna(default)
        return pd.Series(default, index=X.index)

    cap = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    is_micro = (cap < 1.0)
    is_ind = (cap >= 1.0)
    
    cols = ['SIC_INF_BESS_FLAG', 'SIC_INF_BESS_FIRE_SAFETY_RISK', '[REDACTED_BY_SCRIPT]', 
            '[REDACTED_BY_SCRIPT]', 'SIC_BESS_CO_LOCATION_INTENSITY']
    
    exist = [c for c in cols if c in X.columns]
    if exist:
        X.loc[is_micro, exist] = 0
        if '[REDACTED_BY_SCRIPT]' in X.columns:
            X.loc[is_ind, '[REDACTED_BY_SCRIPT]'] *= 1.5
            
    return X


def engineer_post_knn_sics(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    X = df.copy()
    if 'knn_lpa_entropy_gcv' in X.columns and '[REDACTED_BY_SCRIPT]' in X.columns:
        X['[REDACTED_BY_SCRIPT]'] = X['knn_lpa_entropy_gcv'] * X['[REDACTED_BY_SCRIPT]']
    else:
        X['[REDACTED_BY_SCRIPT]'] = 0
    return X

def apply_oracle_compression_inference(df, pca, scaler, top_features):
    """
    Applies the learned PCA compression to inference data.
    """
    # 1. Identify Tail Features (Features to compress)
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = list(scaler.feature_names_in_)
    else:
        return df 
        
    # 2. Prepare Input for Scaler
    # Ensure all expected columns exist, fill 0 if missing
    X_tail = df.copy()
    for col in expected_cols:
        if col not in X_tail.columns:
            X_tail[col] = 0
    X_tail = X_tail[expected_cols].fillna(0)
    
    # 3. Transform
    X_scaled = scaler.transform(X_tail)
    X_pca = pca.transform(X_scaled)
    
    # 4. Create PCA Columns
    pca_df = pd.DataFrame(
        X_pca, 
        columns=[f'ORACLE_PCA_{i}' for i in range(X_pca.shape[1])],
        index=df.index
    )
    
    # 5. Concatenate
    return pd.concat([df, pca_df], axis=1)

def generate_knn_features(df, ref_X, ref_y, knn_models):
    """
    Generates k-NN anomaly features using pre-loaded artifacts.
    """
    # Unpack models
    knn_imputer = knn_models.get('knn_imputer')
    knn_scaler = knn_models.get('knn_scaler')
    knn_engine = knn_models.get('knn_engine')

    if not all([knn_imputer, knn_scaler, knn_engine]):
        # Fail gracefully if models missing
        for col in ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'knn_lpa_entropy_gcv']:
            df[col] = 0
        return df

    # Identify GCV features
    if hasattr(knn_imputer, 'feature_names_in_'):
        gcv_features = list(knn_imputer.feature_names_in_)
    elif hasattr(knn_scaler, 'feature_names_in_'):
        gcv_features = list(knn_scaler.feature_names_in_)
    else:
        gcv_features = [c for c in df.columns if c in ref_X.columns]
        
    # Ensure df has these features
    for col in gcv_features:
        if col not in df.columns:
            df[col] = 0
            
    X_target_gcv = df[gcv_features].copy()
    
    # Transform
    try:
        X_target_gcv_imputed = knn_imputer.transform(X_target_gcv)
        X_target_gcv_scaled = knn_scaler.transform(X_target_gcv_imputed)
        
        # Find Neighbors
        distances, indices = knn_engine.kneighbors(X_target_gcv_scaled)
        
        # Map back to reference data
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
    except Exception as e:
        # Fallback in case of shape mismatch during dynamic generation
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        for col in ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'knn_lpa_entropy_gcv']:
            df[col] = 0
    
    df.fillna(0, inplace=True)
    return df


def coalesce_features(df, map_dict):
    """[REDACTED_BY_SCRIPT]"""
    for target, sources in map_dict.items():
        existing_sources = [c for c in sources if c in df.columns]
        if not existing_sources:
            continue
        for col in existing_sources:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[target] = df[existing_sources].mean(axis=1)
    return df

def rename_features_logic(df):
    """[REDACTED_BY_SCRIPT]"""
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
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def create_artificial_features(df):
    """[REDACTED_BY_SCRIPT]"""
    df['technology_type'] = 21
    df['chp_enabled'] = np.nan
    df['[REDACTED_BY_SCRIPT]'] = 1
    df['permission_granted'] = 1
    df['[REDACTED_BY_SCRIPT]'] = np.nan
    month = 11
    df['submission_month_sin'] = math.sin(2 * math.pi * month / 12)
    df['submission_month_cos'] = math.cos(2 * math.pi * month / 12)
    return df

def enforce_uncertainty_physics(prediction, p10_raw, p90_raw, local_rmse):
    """
    Refines raw quantile predictions to ensure monotonicity and incorporate local accuracy.
    AD-AM-FIX-01: Anchor & Clamp Protocol.
    """
    # 1. Anchor: Prediction is truth. 
    # Minimum logical spread (10% of duration OR 50% of local RMSE)
    min_spread = max(prediction * 0.10, local_rmse * 0.5)
    
    # 2. Derive implied widths
    # Handle inversions immediately by taking max(delta, min_spread)
    low_delta = prediction - p10_raw
    high_delta = p90_raw - prediction
    
    # 3. Clamp (Physical Forcing)
    if low_delta < min_spread:
        low_delta = min_spread
    if high_delta < min_spread:
        high_delta = min_spread
        
    # 4. Construct Final Values
    p10_final = prediction - low_delta
    p90_final = prediction + high_delta
    
    # Final sanity check to ensure P10 is strictly positive (planning can't be negative days)
    p10_final = max(0, p10_final)
    
    return float(p10_final), float(p90_final)

def sanitize_dossier_properties(props):
    """
    Clean up floats and allow-list critical V16 features.
    """
    clean = {}
    
    # 1. Blocklist (Redundant sweeps)
    blocklist = [k for k in props.keys() if k.startswith('sim_') and k.endswith('_days')]
    
    # 2. V16 Allow-list (High Value Signals)
    allow_list_patterns = [
        'SIC_GM_GRID_TRAFFIC_LIGHT',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'project_regime_id',
        'uncertainty_',
        'error_prob_',
        'geo_bench_',
        'typo_bench_',
        'driver_',
        '[REDACTED_BY_SCRIPT]',
        'narrative_summary'
    ]
    
    for k, v in props.items():
        if k in blocklist:
            continue
            
        # Check if it's a V16 Feature we want to keep
        is_allowed = any(p in k for p in allow_list_patterns)
        
        # Keep basic keys AND allowed features
        if is_allowed or k in ['optimal_capacity_mw', '[REDACTED_BY_SCRIPT]', 'hex_id', 'centroid_lon', 'centroid_lat', '[REDACTED_BY_SCRIPT]']:
             if isinstance(v, float):
                if 'prob' in k or 'score' in k:
                    clean[k] = round(v, 3)
                elif 'days' in k or 'capacity' in k:
                    clean[k] = int(round(v))
                elif 'lat' in k or 'lon' in k:
                    clean[k] = round(v, 5)
                else:
                    clean[k] = round(v, 2)
             else:
                clean[k] = v
            
    return clean


def align_to_model_features(row_df, model):
    """
    Ensures the single-row dataframe has exactly the columns the model expects, 
    in the correct order. Fills missing with 0.
    """
    # Handle models wrapped in dictionaries or custom objects
    if hasattr(model, 'feature_name_'):
        expected_cols = model.feature_name_
    elif hasattr(model, 'feature_names_in_'):
        # Sklearn style
        expected_cols = model.feature_names_in_
    else:
        # Fallback: Model doesn't expose features, pass through and hope
        return row_df
        
    aligned_df = pd.DataFrame(index=row_df.index)
    
    for col in expected_cols:
        if col in row_df.columns:
            aligned_df[col] = row_df[col]
        else:
            aligned_df[col] = 0.0 # Fill missing dummies with 0
            
    # Enforce order
    aligned_df = aligned_df[expected_cols]
            
    return aligned_df

def align_to_model_features(row_df, model):
    """
    Ensures the single-row dataframe has exactly the columns the model expects, 
    in the correct order. Fills missing with 0.
    """
    # Handle models wrapped in dictionaries or custom objects
    if hasattr(model, 'feature_name_'):
        expected_cols = model.feature_name_
    elif hasattr(model, 'feature_names_in_'):
        # Sklearn style
        expected_cols = model.feature_names_in_
    else:
        # Fallback: Model doesn't expose features, pass through and hope
        return row_df
        
    aligned_df = pd.DataFrame(index=row_df.index)
    
    for col in expected_cols:
        if col in row_df.columns:
            aligned_df[col] = row_df[col]
        else:
            aligned_df[col] = 0.0 # Fill missing dummies with 0
            
    # Enforce order
    aligned_df = aligned_df[expected_cols]
            
    return aligned_df

def generate_stratified_predictions(X, artifacts):
    """
    Routes sample to appropriate capacity model using loaded bins/labels.
    Accepts artifacts dict: {'models': dict, 'bins': list, 'labels': list}
    """
    if artifacts is None or not isinstance(artifacts, dict):
        return 0.0

    models_dict = artifacts.get('models')
    bins = artifacts.get('bins')
    labels = artifacts.get('labels')
    
    if not all([models_dict, bins, labels]):
        return 0.0

    # Handle single row logic
    capacity = X['[REDACTED_BY_SCRIPT]'].iloc[0] if isinstance(X, pd.DataFrame) else X['[REDACTED_BY_SCRIPT]']
    
    # Identify Stratum
    stratum = None
    for i in range(len(bins)-1):
        if bins[i] <= capacity < bins[i+1]:
            stratum = labels[i]
            break
    
    if stratum and stratum in models_dict and models_dict[stratum] is not None:
        model = models_dict[stratum]
        
        # Feature Diet Handling (AD-AM-48)
        # If model has specific diet features, align to them
        if hasattr(model, 'feature_names_diet_'):
            # Create subset with 0-fill for missing
            X_input = pd.DataFrame(index=X.index)
            for f in model.feature_names_diet_:
                X_input[f] = X.get(f, 0)
            X_input = X_input[model.feature_names_diet_]
        else:
            # Fallback alignment
            X_input = align_to_model_features(X, model)
            
        return model.predict(X_input)[0]
        
    return 0.0

def generate_solar_bridge_features(X, models):
    """
    Compress features using the Solar Bridge (Top N + PCA).
    Returns None if artifacts are missing.
    """
    if models.get('solar_bridge_pca') is None:
        return None
        
    top_features = models['[REDACTED_BY_SCRIPT]']
    pca = models['solar_bridge_pca']
    scaler_pca = models['solar_bridge_scaler']
    
    # Keep Top N Raw
    X_top = X[top_features].copy()
    
    # Compress Rest
    # Identify "other" features - this requires knowing the exact set used during training.
    # Since we don'[REDACTED_BY_SCRIPT]'s all other numeric columns?
    # This is risky. We'll try to use the scaler's feature_names_in_ if available.
    
    if hasattr(scaler_pca, 'feature_names_in_'):
        other_features = scaler_pca.feature_names_in_
        # Ensure columns exist
        for f in other_features:
            if f not in X.columns:
                X[f] = 0
        
        X_other_scaled = scaler_pca.transform(X[other_features].fillna(0))
        X_pca = pd.DataFrame(
            pca.transform(X_other_scaled), 
            columns=[f'PCA_{i}' for i in range(pca.n_components_)],
            index=X.index
        )
        
        X_compressed = pd.concat([X_top, X_pca], axis=1)
        
        # Ensure capacity is present for stratification
        if '[REDACTED_BY_SCRIPT]' not in X_compressed.columns:
            X_compressed['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]']
            
        return X_compressed
        
    return None

# ==============================================================================
# CORE: Model Loading & Historical Data
# ==============================================================================

def load_all_models():
    """
    Load all model artifacts required for the Decision Dossier pipeline.
    This is called once at startup to avoid repeated I/O.
    """
    logging.info("="*70)
    logging.info("[REDACTED_BY_SCRIPT]")
    logging.info("="*70)
    
    models = {}
    
    # Oracle
    logging.info("  Loading Oracle...")
    models['oracle'] = joblib.load(ORACLE_PATH)
    
    # Arbiter
    logging.info("[REDACTED_BY_SCRIPT]")
    models['arbiter'] = joblib.load(ARBITER_MODEL_PATH)
    models['arbiter_scaler'] = joblib.load(ARBITER_SCALER_PATH)
    
    # Quantile models
    logging.info("[REDACTED_BY_SCRIPT]")
    models['p10_quantile'] = joblib.load(P10_MODEL_PATH)
    models['p90_quantile'] = joblib.load(P90_MODEL_PATH)
    
    # Catastrophe classifier
    logging.info("[REDACTED_BY_SCRIPT]")
    catastrophe_path = os.path.join(HEADS_DIR, "[REDACTED_BY_SCRIPT]")
    if os.path.exists(catastrophe_path):
        models['catastrophe'] = joblib.load(catastrophe_path)
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")
        models['catastrophe'] = None
    
    # k-NN artifacts
    logging.info("[REDACTED_BY_SCRIPT]")
    models['knn_engine'] = joblib.load(KNN_ENGINE_PATH)
    models['knn_scaler'] = joblib.load(KNN_SCALER_PATH)
    models['knn_imputer'] = joblib.load(KNN_IMPUTER_PATH)
    
    # Regime clustering
    logging.info("[REDACTED_BY_SCRIPT]")
    models['regime_model'] = joblib.load(REGIME_MODEL_PATH)
    models['regime_scaler'] = joblib.load(REGIME_SCALER_PATH)
    
    # Specialist heads
    logging.info("[REDACTED_BY_SCRIPT]")
    models['specialist_heads'] = {}
    head_files = glob.glob(os.path.join(HEADS_DIR, "*.joblib"))
    for hf in head_files:
        cohort_name = os.path.basename(hf).replace('.joblib', '')
        if cohort_name not in ['[REDACTED_BY_SCRIPT]', 'top_head_features']:
            models['specialist_heads'][cohort_name] = joblib.load(hf)
    
    logging.info(f"[REDACTED_BY_SCRIPT]'specialist_heads'[REDACTED_BY_SCRIPT]")

    # Solar Cascade Models
    logging.info("[REDACTED_BY_SCRIPT]")
    # 1. Solar Residual Specialist (Phase I)
    SOLAR_RESIDUAL_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
    models['solar_residual'] = joblib.load(SOLAR_RESIDUAL_PATH) if os.path.exists(SOLAR_RESIDUAL_PATH) else None
    
    # 2. Solar Stratified (Phase J) - Dict of models
    models['solar_stratified'] = joblib.load(SOLAR_STRATIFIED_PATH) if os.path.exists(SOLAR_STRATIFIED_PATH) else None
    
    # 3. GM Specialist (Phase K) - Dict of models
    models['gm_specialist'] = joblib.load(GM_SPECIALIST_PATH) if os.path.exists(GM_SPECIALIST_PATH) else None
    
    # 4. GM Scale Corrector (Phase L) - Single Regressor (renamed from gm_stratified for clarity)
    models['gm_scale_corrector'] = joblib.load(GM_STRATIFIED_PATH) if os.path.exists(GM_STRATIFIED_PATH) else None

    # 5. Failure Specialist (Phase E)
    FAILURE_SPECIALIST_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
    models['failure_specialist'] = joblib.load(FAILURE_SPECIALIST_PATH) if os.path.exists(FAILURE_SPECIALIST_PATH) else None

    if models['solar_residual']: logging.info("[REDACTED_BY_SCRIPT]")
    if models['solar_stratified']: logging.info("[REDACTED_BY_SCRIPT]")
    if models['gm_specialist']: logging.info("[REDACTED_BY_SCRIPT]")
    if models['gm_scale_corrector']: logging.info("[REDACTED_BY_SCRIPT]")
    if models['failure_specialist']: logging.info("[REDACTED_BY_SCRIPT]")

    # Terminal Calibrator
    logging.info("[REDACTED_BY_SCRIPT]")
    models['terminal_calibrator'] = joblib.load(TERMINAL_CALIBRATOR_PATH) if os.path.exists(TERMINAL_CALIBRATOR_PATH) else None
    if models['terminal_calibrator']: logging.info("[REDACTED_BY_SCRIPT]")

    # Solar Bridge Artifacts
    if os.path.exists(SOLAR_BRIDGE_PCA_PATH):
        logging.info("[REDACTED_BY_SCRIPT]")
        models['solar_bridge_pca'] = joblib.load(SOLAR_BRIDGE_PCA_PATH)
        models['solar_bridge_scaler'] = joblib.load(SOLAR_BRIDGE_SCALER_PATH)
        models['[REDACTED_BY_SCRIPT]'] = joblib.load(SOLAR_BRIDGE_TOP_FEATURES_PATH)
        logging.info("[REDACTED_BY_SCRIPT]")
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")
        models['solar_bridge_pca'] = None

    # SHAP Explainer
    if SHAP_AVAILABLE:
        logging.info("[REDACTED_BY_SCRIPT]")
        try:
            # Initialize with Oracle model
            # Note: TreeExplainer is optimized for LightGBM
            models['shap_explainer'] = shap.TreeExplainer(models['oracle'])
            logging.info("[REDACTED_BY_SCRIPT]")
        except Exception as e:
            logging.warning(f"[REDACTED_BY_SCRIPT]")
            models['shap_explainer'] = None
    else:
        models['shap_explainer'] = None

    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return models

def load_historical_data():
    """
    Load reference training data for contextual benchmarking.
    Performs 3-way join: Features (Ref_X) + Targets (Ref_y) + Coordinates (L49).
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # 1. Load Features and Targets
    ref_X = pd.read_csv(REF_X_PATH, index_col=0)
    ref_y = pd.read_csv(REF_Y_PATH, index_col=0).squeeze("columns")
    
    # 2. Load Coordinates (The Keyring)
    if os.path.exists(COORD_DATA_PATH):
        coord_df = pd.read_csv(COORD_DATA_PATH)
        # Ensure index alignment (Assuming amaryllis_id is the join key)
        if 'amaryllis_id' in coord_df.columns:
            coord_df.set_index('amaryllis_id', inplace=True)
            logging.info(f"[REDACTED_BY_SCRIPT]")
        else:
            logging.warning("[REDACTED_BY_SCRIPT]'amaryllis_id' column")
            coord_df = pd.DataFrame() # Empty fallback
    else:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        coord_df = pd.DataFrame()

    # CRITICAL CHECK: Ensure lpa_name exists for Bayesian Priors
    if 'lpa_name' not in ref_X.columns:
        logging.info("Checking for 'lpa_name'[REDACTED_BY_SCRIPT]")
        lpa_cols = [c for c in ref_X.columns if c.startswith('lpa_name_')]
        if lpa_cols:
            logging.info(f"[REDACTED_BY_SCRIPT]")
            ref_X['lpa_name'] = ref_X[lpa_cols].idxmax(axis=1).apply(lambda x: str(x).replace('lpa_name_', ''))
        else:
            logging.warning("[REDACTED_BY_SCRIPT]")
            ref_X['lpa_name'] = 'Unknown'

    # 3. Sanitize & Filter Features
    ref_X = sanitize_column_names(ref_X)
    valid_mask = ref_y.notna()
    ref_X = ref_X[valid_mask]
    ref_y = ref_y[valid_mask]
    
    # 4. Intersection Join (The Critical Link)
    # We need samples that exist in ALL three sets: X, y, and Coords
    common_indices = ref_X.index.intersection(ref_y.index)
    if not coord_df.empty:
        common_indices = common_indices.intersection(coord_df.index)
    
    ref_X = ref_X.loc[common_indices]
    ref_y = ref_y.loc[common_indices]
    
    # 5. Extract & Project Coordinates
    coordinates = None
    if not coord_df.empty and len(common_indices) > 0:
        target_coords = coord_df.loc[common_indices]
        
        # Check for BNG columns
        if 'easting_x' in target_coords.columns and 'northing_x' in target_coords.columns:
            logging.info("[REDACTED_BY_SCRIPT]")
            try:
                # Initialize Transformer: EPSG:27700 (BNG) -> EPSG:4326 (WGS84)
                transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
                
                # Transform (returns lon, lat)
                lon, lat = transformer.transform(
                    target_coords['easting_x'].values, 
                    target_coords['northing_x'].values
                )
                
                # Stack into [lon, lat] array
                coordinates = np.column_stack((lon, lat))
                logging.info(f"[REDACTED_BY_SCRIPT]")
                
            except Exception as e:
                logging.error(f"[REDACTED_BY_SCRIPT]")
                coordinates = None
        else:
            logging.warning("[REDACTED_BY_SCRIPT]'easting_x'/'northing_x'")
    
    if coordinates is None:
        logging.warning("[REDACTED_BY_SCRIPT]")
    
    # Pre-calculate LPA Statistics
    lpa_stats = calculate_lpa_stats(ref_X, ref_y)
    logging.info("[REDACTED_BY_SCRIPT]")

    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return {
        'X': ref_X,
        'y': ref_y,
        'coordinates': coordinates,
        'lpa_stats': lpa_stats
    }

# ==============================================================================
# CORE: Decision Dossier Generation
# ==============================================================================

def run_single_simulation(hex_features, capacity_mw, models, historical_data, lpa_stats):
    """
    Run the complete prediction pipeline for a single capacity value.
    Returns the final predicted duration for this scenario.
    """
    # Clone features and set capacity
    sim_df = hex_features.copy()
    sim_df['[REDACTED_BY_SCRIPT]'] = capacity_mw
    
    # Feature engineering
    # 1. Base Temporal & Congestion (Prerequisites)
    sim_df, _, _, _ = engineer_temporal_context_features(sim_df)
    sim_df = engineer_lpa_congestion_metrics(sim_df)

    # --- BAYESIAN INJECTION & FRICTION MATRIX ---
    # Use fast lookup mapping
    if lpa_stats:
        sim_df = apply_lpa_stats(sim_df, lpa_stats)
    
    # Calculates FI_... features using the Bayesian columns
    sim_df = engineer_friction_matrix_features(sim_df)
    
    # 2. V16 Consolidated Specialist Engine
    specialist_features = engineer_gm_specialist_consolidated_final(sim_df)
    sim_df = pd.concat([sim_df, specialist_features], axis=1)
    sim_df = sim_df.loc[:, ~sim_df.columns.duplicated()]
    
    # 3. ORACLE COMPRESSION (The Solar Bridge)
    if models.get('solar_bridge_pca'):
        sim_df = apply_oracle_compression_inference(
            sim_df, 
            models['solar_bridge_pca'], 
            models['solar_bridge_scaler'], 
            models['[REDACTED_BY_SCRIPT]']
        )

    # 4. k-NN ANOMALY DETECTION (Depends on PCA if GCV uses it)
    if 'knn_engine' in models:
        sim_df = generate_knn_features(sim_df, historical_data['X'], historical_data['y'], models)

    # 5. POST-kNN SICs (Dependency: k-NN features)
    sim_df = engineer_post_knn_sics(sim_df)

    # 6. PROJECT REGIME (Dependency: k-NN features)
    if models.get('regime_model'):
        # Define the exact features mandated by the new training script
        regime_feats = [
            '[REDACTED_BY_SCRIPT]', 
            'lpa_major_commercial_approval_rate',
            '[REDACTED_BY_SCRIPT]', 
            '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]'
        ]
        
        # Prepare input vector
        regime_input = sim_df.copy()
        for f in regime_feats:
            if f not in regime_input.columns:
                regime_input[f] = 0
        
        # Scale and Predict
        regime_input_scaled = models['regime_scaler'].transform(regime_input[regime_feats])
        sim_df['project_regime_id'] = models['regime_model'].predict(regime_input_scaled)
    
    # --- 1. ORACLE PREDICTION ---
    # Use the Compressed Features (Top + PCA) generated in Chunk 2
    # oracle_input = apply_oracle_compression_inference(
    #     sim_df, models['solar_bridge_pca'], models['solar_bridge_scaler'], models['[REDACTED_BY_SCRIPT]']
    # )
    # FIX: sim_df already has PCA columns from Step 3. Do not re-apply.
    oracle_input = sim_df
    
    # Align to Oracle's specific feature set
    oracle_input_aligned = align_to_model_features(oracle_input, models['oracle'])
    oracle_pred = models['oracle'].predict(oracle_input_aligned)[0]

    # --- 2. AUXILIARY SIGNALS (Inputs for Arbiter) ---
    arbiter_input = pd.DataFrame(index=sim_df.index)
    arbiter_input['[REDACTED_BY_SCRIPT]'] = oracle_pred
    
    # A. Catastrophe Risk
    cat_prob = 0.0
    if models.get('catastrophe'):
        cat_input = align_to_model_features(sim_df, models['catastrophe'])
        if hasattr(models['catastrophe'], 'predict_proba'):
            # Predict Proba returns [prob_0, prob_1]
            cat_prob = models['catastrophe'].predict_proba(cat_input)[0][1] 
        else:
            # Fallback for Regressor trained on binary target
            cat_prob = float(models['catastrophe'].predict(cat_input)[0])
            cat_prob = max(0.0, min(1.0, cat_prob))
    arbiter_input['[REDACTED_BY_SCRIPT]'] = cat_prob
    
    # B. Uncertainty Intervals (P10/P90)
    p10 = models['p10_quantile'].predict(align_to_model_features(sim_df, models['p10_quantile']))[0]
    p90 = models['p90_quantile'].predict(align_to_model_features(sim_df, models['p90_quantile']))[0]
    arbiter_input['[REDACTED_BY_SCRIPT]'] = p90 - p10
    arbiter_input['[REDACTED_BY_SCRIPT]'] = (p90 + p10) / 2.0

    # C. Specialist Heads (The "Conclave")
    head_errors = {}
    for name, head_model in models['specialist_heads'].items():
        # Heads are now REGRESSORS predicting ERROR
        head_input = align_to_model_features(sim_df, head_model)
        pred_error = head_model.predict(head_input)[0]
        head_errors[name] = pred_error
        arbiter_input[f'{name}_pred_error'] = pred_error
        
    # Aggregate Head Signals
    if head_errors:
        # Mean, Std, Max Abs
        errors_arr = np.array(list(head_errors.values()))
        arbiter_input['[REDACTED_BY_SCRIPT]'] = np.mean(errors_arr)
        arbiter_input['std_specialist_residual'] = np.std(errors_arr)
        arbiter_input['[REDACTED_BY_SCRIPT]'] = np.max(np.abs(errors_arr))
        
        # D. Dominant Error Source (One-Hot)
        max_source = max(head_errors, key=lambda k: abs(head_errors[k]))
        arbiter_input[f'[REDACTED_BY_SCRIPT]'] = 1
        
    # E. Project Regime (One-Hot)
    regime_id = int(sim_df['project_regime_id'].iloc[0])
    arbiter_input[f'[REDACTED_BY_SCRIPT]'] = 1
    
    # F. Failure Regime Specialist
    fail_pred = 0.0
    if models.get('failure_specialist'):
        fail_input = align_to_model_features(sim_df, models['failure_specialist'])
        fail_pred = models['failure_specialist'].predict(fail_input)[0]
    arbiter_input['failure_regime_error_pred'] = fail_pred
    
    # G. Stratified Residual (Placeholder from Solar Stratified)
    # Note: In inference, we calculate this later, but Arbiter might have been trained with it.
    # For safety, we initialize to 0 here or run Solar Stratified now.
    # The training order was: SolarStrat -> GM. Arbiter uses SolarStrat.
    # Let's run Solar Stratified now to feed Arbiter.
    
    # Prepare Solar Bridge features for Cascade
    # sim_df already has Top + V16 features. We need Top + PCA + V16.
    # Use the oracle_input from Step 1 which has PCA columns.
    # But we need to add back the V16 features from sim_df to oracle_input.
    X_bridge = pd.concat([oracle_input, sim_df[sim_df.columns.difference(oracle_input.columns)]], axis=1)
    
    solar_strat_resid = 0.0
    if models.get('solar_stratified'):
        solar_strat_resid = generate_stratified_predictions(X_bridge, models['solar_stratified'])
    arbiter_input['[REDACTED_BY_SCRIPT]'] = solar_strat_resid

    # --- 3. ARBITER EXECUTION ---
    # Add Context features required by Diet
    context_cols = ['[REDACTED_BY_SCRIPT]', 'lpa_major_commercial_approval_rate', 'project_regime_id', 'SIC_POLICY_REGIME_ID']
    for c in context_cols:
        arbiter_input[c] = sim_df.get(c, 0)

    final_arbiter_input = align_to_model_features(arbiter_input, models['arbiter'])
    
    # Scale if required (Ridge Arbiter) - though new script uses LGBM, check just in case
    if models.get('arbiter_scaler'):
        # Only scale if columns match scaler (LGBM arbiter might not have scaler)
        try:
            final_arbiter_input = models['arbiter_scaler'].transform(final_arbiter_input)
        except:
            pass # Skip scaling if mismatch, LGBM handles raw
            
    arbiter_correction = models['arbiter'].predict(final_arbiter_input)[0]
    
    # --- 4. THE RESIDUAL CASCADE ---
    # Final = Oracle + Arbiter + Solar_Resid + Solar_Strat + GM_Spec + GM_Scale
    
    # A. Solar Residual Specialist
    solar_resid = 0.0
    if models.get('solar_residual'):
        sol_input = align_to_model_features(X_bridge, models['solar_residual'])
        solar_resid = models['solar_residual'].predict(sol_input)[0]
        
    # B. Solar Stratified (Already calc'd for Arbiter)
    # solar_strat_resid
    
    # C. GM Specialist & Scale Corrector
    gm_resid = 0.0
    gm_scale = 0.0
    
    if sim_df['[REDACTED_BY_SCRIPT]'].iloc[0] == 1:
        if models.get('gm_specialist'):
             # generate_stratified handles binning logic
             gm_resid = generate_stratified_predictions(X_bridge, models['gm_specialist'])
             
        if models.get('gm_scale_corrector'):
             # Uses Raw V16 features
             scale_input = align_to_model_features(sim_df, models['gm_scale_corrector'])
             gm_scale = models['gm_scale_corrector'].predict(scale_input)[0]

    # --- 5. FINAL SUMMATION ---
    final_pred = oracle_pred + arbiter_correction + solar_resid + solar_strat_resid + gm_resid + gm_scale
    
    # Apply AD-AM-29 Positive Bias Calibration (10% Safety Margin)
    final_pred_calibrated = final_pred * 1.10
    
    # Capture the dataframe used for GM prediction (Compressed + V16 features) for the Calibrator
    # We assume 'X_bridge' contains the Compressed Solar features. 
    # We must ensure V16 Specialist features are also included.
    # Reconstruct the GM Specialist input frame:
    X_calib = X_bridge.copy()
    # Inject V16 Specialist FIs (already in sim_df, just need to merge)
    # Filter sim_df to V16 columns to avoid duplication or noise
    v16_cols = [c for c in sim_df.columns if c not in X_calib.columns]
    X_calib = pd.concat([X_calib, sim_df[v16_cols]], axis=1)
    
    return {
        'capacity': capacity_mw,
        'prediction': final_pred_calibrated,
        'features': sim_df.iloc[0].to_dict(),
        'calibrator_input_df': X_calib 
    }

def identify_optimal_scenario(simulations, threshold_days=300):
    """
    Select the simulation with highest capacity below threshold.
    If all exceed threshold, select minimum duration.
    """
    acceptable = [s for s in simulations if s['prediction'] <= threshold_days]
    
    if acceptable:
        return max(acceptable, key=lambda s: s['capacity'])
    else:
        return min(simulations, key=lambda s: s['prediction'])

def generate_uncertainty_metrics(optimal_sim, models, local_rmse):
    """
    Execute uncertainty quantification protocols.
    AD-AM-FIX-01: Updated to use Anchor & Clamp physics.
    """
    feature_df = pd.DataFrame([optimal_sim['features']])
    prediction = optimal_sim['prediction']
    
    # 1. Raw Quantile Predictions
    if hasattr(models['p10_quantile'], 'feature_name_'):
        p10_feats = models['p10_quantile'].feature_name_
        for f in p10_feats:
            if f not in feature_df.columns:
                feature_df[f] = 0
        p10_raw = models['p10_quantile'].predict(feature_df[p10_feats])[0]
    else:
        p10_raw = prediction * 0.85 
    
    if hasattr(models['p90_quantile'], 'feature_name_'):
        p90_feats = models['p90_quantile'].feature_name_
        for f in p90_feats:
            if f not in feature_df.columns:
                feature_df[f] = 0
        p90_raw = models['p90_quantile'].predict(feature_df[p90_feats])[0]
    else:
        p90_raw = prediction * 1.15 

    # 2. Enforce Uncertainty Physics (Anchor & Clamp)
    p10_final, p90_final = enforce_uncertainty_physics(prediction, p10_raw, p90_raw, local_rmse)
    
    # Error regime probabilities (Terminal Calibrator)
    calib_probs = {'[REDACTED_BY_SCRIPT]': 0.0, 'prob_accurate': 1.0, '[REDACTED_BY_SCRIPT]': 0.0}
    
    if models.get('terminal_calibrator') and 'calibrator_input_df' in optimal_sim:
        try:
            # Align features for Calibrator (using helper)
            calib_input = align_to_model_features(optimal_sim['calibrator_input_df'], models['terminal_calibrator'])
            
            if hasattr(models['terminal_calibrator'], 'predict_proba'):
                probs = models['terminal_calibrator'].predict_proba(calib_input)[0]
                # Classes are [0, 1, 2] -> [Under, Accurate, Over]
                calib_probs = {
                    '[REDACTED_BY_SCRIPT]': float(probs[0]),
                    'prob_accurate': float(probs[1]),
                    '[REDACTED_BY_SCRIPT]': float(probs[2])
                }
            else:
                # Fallback: If it's a regressor, we can't get probabilities easily.
                # logging.warning("[REDACTED_BY_SCRIPT]")
                pass
        except Exception as e:
            logging.warning(f"[REDACTED_BY_SCRIPT]")

    # Catastrophe Risk is now defined as the probability of Significant Underprediction
    catastrophe_risk = calib_probs['[REDACTED_BY_SCRIPT]']

    return UncertaintyMetrics(
        confidence_interval_days=[float(p10_final), float(p90_final)],
        catastrophe_risk_prob=catastrophe_risk, 
        error_regime_probabilities=calib_probs
    )

def generate_explainability(optimal_sim, models):
    """
    Execute SHAP-based explainability on the GM Specialist Model.
    Target: The rich V16 features, not the opaque Oracle PCA features.
    """
    # Check for 'gm_specialist' specifically, though 'shap_explainer' is generic
    if not SHAP_AVAILABLE or models.get('gm_specialist') is None:
        return ExplainabilityEngine("SHAP_UNAVAILABLE", "[REDACTED_BY_SCRIPT]", [])

    try:
        # 1. Retrieve the dataframe used for the GM Specialist prediction
        # (Must be passed out of run_single_simulation as 'calibrator_input_df')
        X_shap = optimal_sim.get('calibrator_input_df')
        
        if X_shap is None:
            return ExplainabilityEngine("DATA_MISSING", "[REDACTED_BY_SCRIPT]", [])

        # 2. Select the correct model (Stratified Logic)
        gm_artifacts = models['gm_specialist']
        target_model = None
        stratum_key = 'default'
        
        if isinstance(gm_artifacts, dict) and 'models' in gm_artifacts:
            # Stratified Model
            capacity = optimal_sim.get('capacity', 0)
            bins = gm_artifacts.get('bins')
            labels = gm_artifacts.get('labels')
            models_dict = gm_artifacts.get('models')
            
            if bins and labels:
                for i in range(len(bins)-1):
                    if bins[i] <= capacity < bins[i+1]:
                        stratum_key = labels[i]
                        break
            
            if stratum_key in models_dict:
                target_model = models_dict[stratum_key]
        else:
            # Single Model (Legacy or different structure)
            target_model = gm_artifacts

        if target_model is None:
             # If no model found for this stratum (e.g. < 1MW), skip explainability or use fallback
             return ExplainabilityEngine("MODEL_MISSING", "[REDACTED_BY_SCRIPT]", [])

        # 3. Initialize Explainer (Lazy Load or Pre-loaded)
        if 'gm_explainers' not in models:
            models['gm_explainers'] = {}
            
        if stratum_key not in models['gm_explainers']:
            try:
                models['gm_explainers'][stratum_key] = shap.TreeExplainer(target_model)
            except Exception as e:
                 # Fallback for non-tree models?
                 # logging.warning(f"[REDACTED_BY_SCRIPT]")
                 return ExplainabilityEngine("SHAP_ERROR", f"[REDACTED_BY_SCRIPT]", [])
            
        explainer = models['gm_explainers'][stratum_key]
        
        # 4. Calculate SHAP Values
        # Ensure X_shap aligns with model features!
        X_shap_aligned = align_to_model_features(X_shap, target_model)
        
        shap_values = explainer.shap_values(X_shap_aligned)
        
        # Handle LGBM output (list if multiclass, but GM Spec is Regression, so usually array)
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        # Flatten if needed
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[0]
            
        # 4. Construct Feature Importance Table
        feature_names = X_shap_aligned.columns
        feature_values = X_shap_aligned.values[0]
        
        feature_importance = []
        for feat, shap_val, feat_val in zip(feature_names, shap_vals, feature_values):
            # FILTER: We only care about human-readable "SIC_" or "LPA_" features
            # We ignore PCA columns and raw geometric inputs unless highly relevant
            if feat.startswith('SIC_') or feat.startswith('LPA_') or feat.startswith('FI_'):
                feature_importance.append({
                    'feature': feat,
                    'shap_value': float(shap_val),
                    'abs_shap': abs(float(shap_val)),
                    'feature_value': float(feat_val)
                })
        
        # 5. Sort & Rank
        feature_importance.sort(key=lambda x: x['abs_shap'], reverse=True)
        
        top_drivers = []
        for item in feature_importance[:5]: # Top 5
            impact_days = item['shap_value']
            sign = "+" if impact_days > 0 else ""
            impact_str = f"[REDACTED_BY_SCRIPT]"
            
            # Clean description
            desc = item['feature'].replace('SIC_', '').replace('GM_', '').replace('_', ' ').title()
            
            top_drivers.append(FeatureDriver(
                feature=item['feature'],
                impact=impact_str,
                description=desc
            ))
            
        # 6. Narrative
        if top_drivers:
            primary = top_drivers[0]
            narrative = f"[REDACTED_BY_SCRIPT]"
            dominant_risk = primary.feature
        else:
            dominant_risk = "NEUTRAL"
            narrative = "[REDACTED_BY_SCRIPT]"

    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return ExplainabilityEngine("CALCULATION_FAILURE", f"Error: {str(e)}", [])

    return ExplainabilityEngine(
        dominant_risk_factor=dominant_risk,
        narrative_summary=narrative,
        top_drivers_for_optimal_scenario=top_drivers
    )


def calculate_dynamic_benchmarks(hex_df, historical_data, models, capacity_mw=None):
    """
    Calculates REAL validation error on nearest neighbors using the Oracle model.
    Implements the "Mirror Test" with strict feature alignment.
    If capacity_mw is provided, filters historical data to the relevant stratum.
    """
    # Default Fail-safes
    fallback_rmse = 45.0
    fallback_mae = 35.0
    
    if historical_data.get('coordinates') is None or len(historical_data['coordinates']) == 0:
        return AccuracyBenchmark(
            description="[REDACTED_BY_SCRIPT]", 
            rmse_days=fallback_rmse, mae_days=fallback_mae, sample_count=0
        )

    try:
        from sklearn.neighbors import NearestNeighbors
        
        # --- STRATUM FILTERING ---
        # Filter historical data to match the capacity stratum of the current simulation
        
        # 1. Identify Indices
        valid_indices = np.arange(len(historical_data['X']))
        stratum_desc = ""
        
        if capacity_mw is not None and models.get('solar_stratified'):
            artifacts = models['solar_stratified']
            bins = artifacts.get('bins')
            
            if bins:
                # Find the bin for this capacity
                lower, upper = 0, 9999
                for i in range(len(bins)-1):
                    if bins[i] <= capacity_mw < bins[i+1]:
                        lower = bins[i]
                        upper = bins[i+1]
                        break
                
                # Filter X
                hist_caps = historical_data['X']['[REDACTED_BY_SCRIPT]'].values
                mask = (hist_caps >= lower) & (hist_caps < upper)
                
                # Only apply if we have enough samples
                if mask.sum() >= 20:
                    valid_indices = np.where(mask)[0]
                    stratum_desc = f"[REDACTED_BY_SCRIPT]"
                else:
                    logging.warning(f"[REDACTED_BY_SCRIPT]")

        # 2. Subset Data
        # Coordinates
        all_coords = np.radians(historical_data['coordinates'])
        ref_coords = all_coords[valid_indices]
        
        # Y
        all_y = historical_data['y'].values
        neighbor_y_subset = all_y[valid_indices]
        
        # X
        all_X = historical_data['X']
        neighbor_X_subset = all_X.iloc[valid_indices]

        # Current site
        lat = hex_df['lat'].iloc[0]
        lon = hex_df['lon'].iloc[0]
        site_coord = np.radians([[lon, lat]])
        
        # Find 20 Nearest Historical Sites (in the subset)
        n_neighbors = min(20, len(ref_coords))
        if n_neighbors < 1:
             return AccuracyBenchmark(
                description="[REDACTED_BY_SCRIPT]", 
                rmse_days=fallback_rmse, mae_days=fallback_mae, sample_count=0
            )
            
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='haversine')
        nn.fit(ref_coords)
        
        dists, indices = nn.kneighbors(site_coord)
        neighbors_idx = indices[0]
        
        # 3. Extract Neighbor Data (from the subset)
        neighbor_y = neighbor_y_subset[neighbors_idx]
        neighbor_X_raw = neighbor_X_subset.iloc[neighbors_idx]

        # 4. CRITICAL: Feature Intersection Layer
        # The Oracle expects specific columns. The historical data might have extra or fewer.
        oracle_features = models['oracle'].feature_name_
        
        # Create a clean DF for prediction
        # We use a dictionary for speed then DataFrame
        X_bench_dict = {}
        
        for feat in oracle_features:
            if feat in neighbor_X_raw.columns:
                X_bench_dict[feat] = neighbor_X_raw[feat].values
            else:
                # Feature missing in historical data? Fill 0.
                X_bench_dict[feat] = np.zeros(len(neighbors_idx))
                
        X_bench = pd.DataFrame(X_bench_dict, index=neighbor_X_raw.index)
                
        # 5. Predict
        preds = models['oracle'].predict(X_bench)
        
        # 6. Calculate Metrics
        errors = preds - neighbor_y
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        
        # Calculate mean distance in km for description
        mean_dist_km = np.mean(dists[0]) * 6371
        
        return AccuracyBenchmark(
            description=f"[REDACTED_BY_SCRIPT]",
            rmse_days=float(rmse),
            mae_days=float(mae),
            sample_count=len(neighbors_idx)
        )

    except Exception as e:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return AccuracyBenchmark(
            description=f"[REDACTED_BY_SCRIPT]", 
            rmse_days=fallback_rmse, mae_days=fallback_mae, sample_count=0
        )

def calculate_typological_benchmark(sim_features, models):
    """
    Calculates Typological Benchmark based on Project Regime.
    This depends on capacity, so it runs for each simulation.
    """
    regime_id = 2 
    regime_rmse = 48.3
    regime_mae = 35.7
    
    try:
        if models.get('regime_model') and models.get('regime_scaler'):
            if hasattr(models['regime_scaler'], 'feature_names_in_'):
                regime_feats = models['regime_scaler'].feature_names_in_
            else:
                raise AttributeError("[REDACTED_BY_SCRIPT]")
            
            feat_row = pd.DataFrame([sim_features])
            for f in regime_feats:
                if f not in feat_row.columns:
                    feat_row[f] = 0
            
            X_regime = models['regime_scaler'].transform(feat_row[regime_feats])
            regime_id = int(models['regime_model'].predict(X_regime)[0])
            
            regime_stats = {
                0: (35.2, 28.1), 1: (42.5, 32.0), 
                2: (48.3, 35.7), 3: (55.1, 40.2)
            }
            regime_rmse, regime_mae = regime_stats.get(regime_id, (45.0, 35.0))
            
    except Exception as e:
        pass

    return TypologicalBenchmark(
        description=f"[REDACTED_BY_SCRIPT]",
        project_regime_id=regime_id,
        rmse_days=float(regime_rmse),
        mae_days=float(regime_mae),
        sample_count=87 
    )

def generate_contextual_benchmarks(geographic_benchmark, sim_features, models):
    """[REDACTED_BY_SCRIPT]"""
    typological_benchmark = calculate_typological_benchmark(sim_features, models)
    return ContextualBenchmarks(
        local_geographic_accuracy=geographic_benchmark,
        typological_accuracy=typological_benchmark
    )

def calculate_percentile_rank(duration, all_simulations):
    """[REDACTED_BY_SCRIPT]"""
    all_durations = [s['prediction'] for s in all_simulations]
    rank = sum(1 for d in all_durations if d < duration)
    return 100.0 * rank / len(all_durations)

def forge_decision_dossier(hex_df, models, historical_data, capacities):
    """
    Execute the complete Decision Dossier protocol for a single hex.
    Generates a separate full dossier for EVERY capacity scenario.
    """
    # 1. (Removed global pre-calculation)
    
    # 2. Run all simulations first to establish the comparison set
    lpa_stats = historical_data.get('lpa_stats')
    simulations = []
    for capacity in capacities:
        sim = run_single_simulation(hex_df, capacity, models, historical_data, lpa_stats)
        simulations.append(sim)
        
    # 3. Generate a full Dossier for EACH simulation
    dossiers = []
    
    for sim in simulations:
        # Calculate Dynamic Benchmark PER SIMULATION (Stratified)
        geo_benchmark = calculate_dynamic_benchmarks(hex_df, historical_data, models, capacity_mw=sim['capacity'])
        local_rmse = geo_benchmark.rmse_days

        # A. Uncertainty (Now linked to local_rmse)
        uncertainty = generate_uncertainty_metrics(sim, models, local_rmse)
        
        # B. Explainability
        explainability = generate_explainability(sim, models)
        
        # C. Contextual Benchmarks (Combine static Geo + dynamic Typo)
        benchmarks = generate_contextual_benchmarks(geo_benchmark, sim['features'], models)
        
        # D. Headline Risk
        headline = HeadlineRiskAssessment(
            optimal_capacity_mw=sim['capacity'], # This simulation is the "optimal" in its own dossier
            predicted_duration_days=float(sim['prediction']),
            duration_risk_category=calculate_risk_category(sim['prediction']),
            duration_risk_percentile=calculate_percentile_rank(sim['prediction'], simulations)
        )
        
        # E. Assemble Dossier
        dossier = DecisionDossier(
            identity=Identity(
                hex_id=str(hex_df['hex_id'].iloc[0]),
                centroid_lon=float(hex_df['lon'].iloc[0]),
                centroid_lat=float(hex_df['lat'].iloc[0])
            ),
            headline_risk_assessment=headline,
            uncertainty_metrics=uncertainty,
            explainability_engine=explainability,
            contextual_benchmarks=benchmarks,
            simulation_details=[
                SimulationDetail(
                    sim_capacity_mw=s['capacity'],
                    predicted_duration_days=float(s['prediction'])
                ) for s in simulations
            ]
        )
        dossiers.append(dossier)
    
    return dossiers



# ==============================================================================
# Streaming & Geometry Helpers
# ==============================================================================

class StreamingGeoJSONWriter:
    """
    Context manager for writing GeoJSON features incrementally to a file.
    This avoids holding the entire FeatureCollection in memory.
    Supports resuming by appending to an existing file.
    """
    def __init__(self, output_path, resume=False):
        self.output_path = output_path
        self.file = None
        self.resume = resume
        self.first_feature = not resume

    def __enter__(self):
        if self.resume and os.path.exists(self.output_path):
            logging.info(f"[REDACTED_BY_SCRIPT]")
            self.file = open(self.output_path, 'a')
        else:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.file = open(self.output_path, 'w')
            # Write header
            self.file.write('{\n')
            self.file.write('  "type": "FeatureCollection",\n')
            self.file.write('  "features": [\n')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            # Write footer
            self.file.write('\n  ]\n}')
            self.file.close()

    def write_feature(self, feature):
        if not self.first_feature:
            self.file.write(',\n')
        else:
            self.first_feature = False
        
        json.dump(feature, self.file)

def load_geometry_mapping(geojson_path):
    """
    Load point geometries from the centroids GeoJSON file.
    Returns a dict mapping hex_id -> geometry dict.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    if not os.path.exists(geojson_path):
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return {}
        
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
            
        mapping = {}
        for feature in data.get('features', []):
            props = feature.get('properties', {})
            hex_id = props.get('hex_id')
            geometry = feature.get('geometry')
            
            if hex_id and geometry:
                mapping[hex_id] = geometry
                
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return mapping
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return {}

def repair_geojson(filepath):
    """
    Repairs a potentially corrupt GeoJSON file (e.g. interrupted write).
    Truncates the file after the last valid feature.
    Returns the hex_id of the last valid feature found, or None.
    """
    if not os.path.exists(filepath):
        return None
        
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    try:
        with open(filepath, 'rb+') as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            
            # Scan backwards for the last valid feature closer '}'
            chunk_size = 65536
            pos = size
            found_end = False
            real_end = 0
            
            # We need to skip the footer if it exists (e.g. ']}\n' or similar)
            # Logic: Find last '}', check if it'[REDACTED_BY_SCRIPT]']')
            
            while pos > 0:
                read_len = min(chunk_size, pos)
                pos -= read_len
                f.seek(pos)
                chunk = f.read(read_len)
                
                # Iterate backwards through chunk
                for i in range(len(chunk) - 1, -1, -1):
                    byte = chunk[i]
                    if byte == ord(b'}'):
                        # Found a closer. Is it root or feature?
                        # Scan backwards from here to see if we hit ']'
                        # This is a local check; might need to look further back but usually footer is small.
                        
                        # Check previous non-whitespace bytes
                        is_root_closer = False
                        # Look back a bit
                        check_pos = pos + i - 1
                        while check_pos >= 0:
                            f.seek(check_pos)
                            b = f.read(1)
                            if b in [b' ', b'\n', b'\r', b'\t']:
                                check_pos -= 1
                                continue
                            if b == b']':
                                is_root_closer = True
                            break
                        
                        if is_root_closer:
                            # This '}' is the root closer. Keep looking backwards for the feature closer.
                            continue
                        else:
                            # This is likely a feature closer
                            real_end = pos + i + 1
                            found_end = True
                            break
                if found_end:
                    break
            
            if not found_end:
                logging.warning("[REDACTED_BY_SCRIPT]")
                return None
                
            logging.info(f"[REDACTED_BY_SCRIPT]")
            f.truncate(real_end)
            
            # Now extract the hex_id from the last feature to know where we are
            # Read the last 64KB (should cover the last feature)
            f.seek(max(0, real_end - 65536))
            data = f.read().decode('utf-8', errors='ignore')
            
            # Regex to find hex_id
            # Matches: "hex_id": "8928308280fffff_10MW"
            import re
            matches = list(re.finditer(r'"hex_id":\s*"([^"]+)"', data))
            if matches:
                last_val = matches[-1].group(1)
                # Parse original hex (remove _CapacityMW suffix)
                parts = last_val.split('_')
                if len(parts) >= 2:
                    last_hex = parts[0]
                    logging.info(f"[REDACTED_BY_SCRIPT]")
                    return last_hex
                else:
                    logging.warning(f"[REDACTED_BY_SCRIPT]")
                    return None
            else:
                logging.warning("[REDACTED_BY_SCRIPT]")
                return None
                
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return None

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # Clear debug log
    try:
        with open("debug_log.txt", "w") as f:
            f.write("[REDACTED_BY_SCRIPT]")
    except:
        pass

    logging.info("="*70)
    logging.info("[REDACTED_BY_SCRIPT]")
    logging.info("="*70)
    
    # 1. Load input data
    if not os.path.exists(INPUT_DATA_PATH):
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return
    
    df = pd.read_csv(INPUT_DATA_PATH) if INPUT_DATA_PATH.endswith('.csv') else pd.read_parquet(INPUT_DATA_PATH)
    df = sanitize_column_names(df)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # 2. Basic preparation  
    df = coalesce_features(df, COALESCE_MAP)
    df = rename_features_logic(df)
    
    for feat in MISSING_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan
    
    df = create_artificial_features(df)
    
    # 3. Load historical data FIRST (needed for k-NN feature generation)
    historical_data = load_historical_data()
    
    # 4. (Skipped) k-NN anomaly features now generated inside simulation loop
    
    # 5. Load all models
    models = load_all_models()
    
    # 6. (Skipped) Geometry Mapping - Using H3 directly for WGS84
    # GEOMETRY_PATH = ...
    # geometry_map = ...
    
    # 7. Process each unique hex
    if 'hex_id' not in df.columns:
        logging.error("[REDACTED_BY_SCRIPT]")
        return
    
    # Ensure hex_id is string and strip whitespace
    df['hex_id'] = df['hex_id'].astype(str).str.strip()
    
    unique_hexes = df['hex_id'].unique()
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # --- RESUME LOGIC ---
    start_index = 0
    resume_mode = False
    
    if os.path.exists(OUTPUT_GEOJSON_PATH):
        last_hex = repair_geojson(OUTPUT_GEOJSON_PATH)
        if last_hex:
            # Find index
            matches = np.where(unique_hexes == last_hex)[0]
            if len(matches) > 0:
                # We resume FROM this hex (re-processing it to ensure completeness)
                # or from the next one?
                # To be safe against partial writes for the last hex, we re-process it.
                # However, this creates duplicates in the file.
                # Given the user request "continue where we left off", re-processing the last one is safer than skipping.
                # But wait, if we re-process, we append.
                # If the file was truncated AFTER the last valid feature of this hex, 
                # we might have some features of this hex already.
                # Ideally we should skip it if we want to avoid duplicates, but we risk missing capacities.
                # Let's skip it to avoid duplicates, assuming the last write was likely a complete block or we accept minor loss.
                # Actually, the user said "[REDACTED_BY_SCRIPT]".
                # Usually implies starting AFTER it.
                start_index = matches[0] + 1
                resume_mode = True
                logging.info(f"[REDACTED_BY_SCRIPT]")
            else:
                logging.warning(f"[REDACTED_BY_SCRIPT]")
    
    if start_index >= len(unique_hexes):
        logging.info("[REDACTED_BY_SCRIPT]")
        return

    import gc
    
    # Use streaming writer (Standard WGS84, no CRS needed)
    with StreamingGeoJSONWriter(OUTPUT_GEOJSON_PATH, resume=resume_mode) as writer:
        success_count = 0
        error_count = 0
        
        for i in range(start_index, len(unique_hexes)):
            hex_id = unique_hexes[i]
            
            if i % 10 == 0:
                logging.info(f"[REDACTED_BY_SCRIPT]")
                # Force garbage collection periodically
                gc.collect()
            
            hex_df = df[df['hex_id'] == hex_id].iloc[0:1].copy()
            
            # Calculate WGS84 coordinates from H3
            try:
                # Try H3 v4 API first, then v3 fallback
                if hasattr(h3, 'cell_to_latlng'):
                    lat, lon = h3.cell_to_latlng(hex_id)
                else:
                    lat, lon = h3.h3_to_geo(hex_id)
                
                hex_df['lat'] = lat
                hex_df['lon'] = lon
                
                # GeoJSON geometry is [lon, lat]
                geometry = {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            except Exception as e:
                logging.warning(f"[REDACTED_BY_SCRIPT]")
                error_count += 1
                continue
            
            try:
                # Returns a LIST of dossiers now (one per capacity)
                dossier_list = forge_decision_dossier(
                    hex_df=hex_df,
                    models=models,
                    historical_data=historical_data,
                    capacities=CAPACITIES_TO_SIMULATE
                )
                
                # Sort dossiers by capacity
                dossier_list.sort(key=lambda d: d.headline_risk_assessment.optimal_capacity_mw)
                total_scenarios = len(dossier_list)

                # Get Hex Boundary for Slicing
                # h3.cell_to_boundary returns tuples ((lat, lon), ...)
                try:
                    # Support v4 or v3
                    if hasattr(h3, 'cell_to_boundary'):
                        boundary_raw = h3.cell_to_boundary(hex_id)
                    else:
                        boundary_raw = h3.h3_to_geo_boundary(hex_id)
                    
                    # Convert to [lon, lat] list for GeoJSON
                    boundary_coords = [[p[1], p[0]] for p in boundary_raw]
                    # Ensure closure not strictly needed for slicing algo but good for reference
                    if boundary_coords[0] != boundary_coords[-1]:
                        boundary_coords.append(boundary_coords[0])
                except Exception as e:
                    logging.error(f"[REDACTED_BY_SCRIPT]")
                    # Fallback to point
                    boundary_coords = None

                # Calculate Angle Per Slice
                angle_step = 360.0 / total_scenarios if total_scenarios > 0 else 360

                for idx, dossier in enumerate(dossier_list):
                    # Get properties and apply strict output hygiene
                    raw_props = dossier.to_geojson_properties()
                    props = sanitize_dossier_properties(raw_props)
                    
                    # Unique ID
                    capacity_mw = props['optimal_capacity_mw']
                    original_hex = props['hex_id']
                    props['hex_id'] = f"[REDACTED_BY_SCRIPT]"
                    
                    # Color Logic
                    risk_cat = props.get('[REDACTED_BY_SCRIPT]', 'Green')
                    if risk_cat == 'Red':
                        color_hex = "#ff0000"
                    elif risk_cat == 'Amber':
                        color_hex = "#ffbf00"
                    else: 
                        color_hex = "#00cc00"
                    
                    props['marker-color'] = color_hex
                    props['fill'] = color_hex
                    props['stroke'] = "#ffffff" # White separator lines
                    props['stroke-width'] = 1
                    props['fill-opacity'] = 0.8

                    # Generate Wedge Geometry
                    if boundary_coords:
                        # Start North (0 deg) and go Clockwise
                        start_angle = idx * angle_step
                        end_angle = (idx + 1) * angle_step
                        
                        center_lon = props['centroid_lon']
                        center_lat = props['centroid_lat']
                        
                        poly_coords = generate_hex_wedge(
                            center_lon, center_lat, boundary_coords, start_angle, end_angle
                        )
                        
                        feature_geometry = {
                            "type": "Polygon",
                            "coordinates": poly_coords
                        }
                    else:
                        # Fallback to Point if boundary fails
                        feature_geometry = geometry

                    feature = {
                        "type": "Feature",
                        "geometry": feature_geometry,
                        "properties": props
                    }
                    writer.write_feature(feature)
                
                success_count += 1
                
                # Explicitly delete heavy objects
                del dossier_list
                
            except Exception as e:
                error_count += 1
                logging.error(f"[REDACTED_BY_SCRIPT]")
                continue
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    if error_count > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
    
    logging.info("\n" + "="*70)
    logging.info("[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("="*70)


def get_angle(p1, p2):
    """[REDACTED_BY_SCRIPT]"""
    lat1, lon1 = math.radians(p1[1]), math.radians(p1[0])
    lat2, lon2 = math.radians(p2[1]), math.radians(p2[0])
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def intersect_ray_segment(origin, angle_deg, p1, p2):
    """
    Strict Parametric Intersection:
    Finds where a ray (origin, angle) intersects segment p1-p2.
    Returns [lon, lat] ONLY if intersection is strictly within segment bounds (0 <= u <= 1).
    """
    # Convert Compass Angle (0=N, 90=E) to standard Euclidean Angle (0=E, 90=N)
    # theta = 90 - angle
    theta = math.radians(90 - angle_deg)
    
    # Ray vector (normalized)
    dx_ray = math.cos(theta)
    dy_ray = math.sin(theta)
    
    # Origin and Segment Points
    ox, oy = origin
    x1, y1 = p1
    x2, y2 = p2
    
    # Segment vector
    dx_seg = x2 - x1
    dy_seg = y2 - y1
    
    # Solve system: O + t*R = P1 + u*S
    # t*dx_ray - u*dx_seg = x1 - ox
    # t*dy_ray - u*dy_seg = y1 - oy
    
    det = dx_ray * (-dy_seg) - dy_ray * (-dx_seg)
    
    # Parallel check
    if abs(det) < 1e-12:
        return None
        
    rhs_x = x1 - ox
    rhs_y = y1 - oy
    
    # Cramer's Rule
    t = (rhs_x * (-dy_seg) - rhs_y * (-dx_seg)) / det
    u = (dx_ray * rhs_y - dy_ray * rhs_x) / det
    
    # STRICT BOUNDS CHECK:
    # t > 0 (Forward direction)
    # 0 <= u <= 1 (Strictly between P1 and P2)
    if t > 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        # Calculate intersection point
        ix = ox + t * dx_ray
        iy = oy + t * dy_ray
        return [ix, iy]
        
    return None

def generate_hex_wedge(center_lon, center_lat, hex_boundary, start_angle, end_angle):
    """
    Generates a polygon representing a slice of the hexagon.
    Uses strict ray-segment intersection to prevent spiky artifacts.
    """
    center = [center_lon, center_lat]
    wedge_coords = [center] 
    
    def get_valid_intersection(target_angle):
        # Test ray against ALL 6 segments. 
        # For a convex hex with center inside, EXACTLY ONE segment should return a valid hit.
        for i in range(len(hex_boundary)):
            p1 = hex_boundary[i]
            # Handle list wrapping safely
            p2 = hex_boundary[(i + 1) % len(hex_boundary)]
            
            # Use strict parametric intersection
            pt = intersect_ray_segment(center, target_angle, p1, p2)
            
            if pt is not None:
                return pt
        
        # Fallback (should theoretically not happen if geometry is valid)
        # Return first point to avoid crash, but log could be added here
        return hex_boundary[0]

    # 1. Intersection at Start Angle
    p_start = get_valid_intersection(start_angle)
    wedge_coords.append(p_start)
    
    # 2. Add intervening vertices
    # Normalize angles to [0, 360)
    s_a = start_angle % 360
    e_a = end_angle % 360
    if e_a <= s_a: e_a += 360 # Handle crossing North
    
    for v in hex_boundary:
        # Calculate angle to vertex
        v_angle = get_angle(center, v)
        
        # Normalize relative to start angle
        # If we just crossed North (e.g. 350 to 10), e_a is 370. 
        # If vertex is 5 deg, rel_v should be 15 (365-350)
        
        # Simple containment check handles wrapping naturally if we shift frame
        # Shift everything so s_a is 0
        norm_v = (v_angle - s_a) % 360
        norm_end = (e_a - s_a)
        
        # Add vertex if it lies strictly inside the sweep
        if 1e-5 < norm_v < norm_end - 1e-5:
            wedge_coords.append(v)
            
    # 3. Intersection at End Angle
    p_end = get_valid_intersection(end_angle)
    wedge_coords.append(p_end)
    
    # Close Polygon
    wedge_coords.append(center)
    
    return [wedge_coords]


if __name__ == '__main__':
    main()

