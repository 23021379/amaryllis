import os
import joblib
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import re
import optuna

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GroupKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
# --- AD-AM-33: New imports for k-NN Anomaly Detection Sub-System ---
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

from sklearn.decomposition import PCA

# --- Project Artifacts & Constants (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Defining new paths for the v3.5 cohort refactoring build
OUTPUT_DIR_V35 = r"[REDACTED_BY_SCRIPT]"
HEADS_V35_DIR = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
MODEL_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
DOSSIER_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
# --- AD-AM-33: Artifact paths for the k-NN Anomaly Detection Sub-System ---
KNN_SCALER_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
KNN_ENGINE_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
KNN_IMPUTER_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
ORACLE_MODEL_PATH = os.path.join(OUTPUT_DIR_V35, "oralce_v3.5.joblib")

# --- AD-AM-47: New Artifact Paths for Nested Specialists ---
SOLAR_STRATIFIED_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
GM_SPECIALIST_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
GM_STRATIFIED_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")

# --- AD-AM-45: Solar Bridge Artifacts ---
SOLAR_BRIDGE_PCA_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
SOLAR_BRIDGE_SCALER_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
SOLAR_BRIDGE_TOP_FEATURES_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")



X_PATH = r"[REDACTED_BY_SCRIPT]"
#Y_REG_PATH = r"[REDACTED_BY_SCRIPT]"
Y_REG_PATH = r"[REDACTED_BY_SCRIPT]"

RANDOM_STATE = 42

SPLITS_GLOBAL=50
TRIALS_GLOBAL=10
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Mandated Semantic Cohort: CORE_ATTRIBUTES (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# REMEDIATION: Removed 'solar_site_area_sqm' due to 100% missing values
COHORT_CORE_ATTRIBUTES = [
    # Core Physical Attributes
    '[REDACTED_BY_SCRIPT]', 
    # 'solar_site_area_sqm',  <-- REMOVED
    'chp_enabled',
    '[REDACTED_BY_SCRIPT]',
]

def engineer_lpa_risk_features(X_train, y_train, X_val, X_test, m_factor=10):
    """
    Architectural Directive 005: The "Shape of Chaos" & Bayesian Smoothing.
    HARDENED: Forces float types to survive artifact purge.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # 1. Forensic Recovery of LPA Name
    def recover_lpa(df):
        if 'lpa_name' in df.columns and df['lpa_name'].notna().any():
            return df
        
        # Attempt OHE recovery
        lpa_cols = [c for c in df.columns if c.startswith('lpa_name_')]
        if lpa_cols:
            print(f"[REDACTED_BY_SCRIPT]")
            df['lpa_name'] = df[lpa_cols].idxmax(axis=1).apply(lambda x: str(x).replace('lpa_name_', ''))
        else:
            print("[REDACTED_BY_SCRIPT]'Unknown'.")
            df['lpa_name'] = 'Unknown'
        return df

    X_train = recover_lpa(X_train)
    X_val = recover_lpa(X_val)
    X_test = recover_lpa(X_test)

    # 2. Calculate Statistics on Training Data
    df_train = X_train.copy()
    df_train['[REDACTED_BY_SCRIPT]'] = y_train
    
    global_mean = df_train['[REDACTED_BY_SCRIPT]'].mean()
    global_p90 = df_train['[REDACTED_BY_SCRIPT]'].quantile(0.90)
    
    stats = df_train.groupby('lpa_name')['[REDACTED_BY_SCRIPT]'].agg(
        n_samples='count',
        lpa_mean='mean',
        lpa_p90=lambda x: x.quantile(0.90),
        lpa_extension_propensity=lambda x: (x > 180).mean()
    ).reset_index()
    
    # IQR Stability
    iqr = df_train.groupby('lpa_name')['[REDACTED_BY_SCRIPT]'].agg(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index().rename(columns={'[REDACTED_BY_SCRIPT]': 'LPA_Stability_Index'})
    
    stats = pd.merge(stats, iqr, on='lpa_name', how='left')
    
    # Bayesian Smoothing Calculation
    stats['[REDACTED_BY_SCRIPT]'] = (
        (stats['n_samples'] * stats['lpa_mean']) + (m_factor * global_mean)
    ) / (stats['n_samples'] + m_factor)
    
    stats['[REDACTED_BY_SCRIPT]'] = (
        (stats['n_samples'] * stats['lpa_p90']) + (m_factor * global_p90)
    ) / (stats['n_samples'] + m_factor)
    
    lpa_std = df_train.groupby('lpa_name')['[REDACTED_BY_SCRIPT]'].std().fillna(global_mean*0.2)
    stats['lpa_std'] = lpa_std.values
    stats['LPA_Chaos_Index'] = stats['lpa_std'] / stats['[REDACTED_BY_SCRIPT]']
    
    # 3. Map Features & FORCE FLOAT
    stats.set_index('lpa_name', inplace=True)
    
    def apply_map(X_df):
        out = X_df.copy()
        # Explicitly cast to float64 to prevent 'object' dtype
        out['[REDACTED_BY_SCRIPT]'] = out['lpa_name'].map(stats['[REDACTED_BY_SCRIPT]']).fillna(global_mean).astype('float64')
        out['[REDACTED_BY_SCRIPT]'] = out['lpa_name'].map(stats['[REDACTED_BY_SCRIPT]']).fillna(global_p90).astype('float64')
        out['LPA_Chaos_Index'] = out['lpa_name'].map(stats['LPA_Chaos_Index']).fillna(0.5).astype('float64')
        out['LPA_Stability_Index'] = out['lpa_name'].map(stats['LPA_Stability_Index']).fillna(global_mean/2).astype('float64')
        out['[REDACTED_BY_SCRIPT]'] = out['lpa_name'].map(stats['[REDACTED_BY_SCRIPT]']).fillna(0.5).astype('float64')
        return out

    X_train_out = apply_map(X_train)
    X_val_out = apply_map(X_val)
    X_test_out = apply_map(X_test)
    
    print(f"[REDACTED_BY_SCRIPT]'LPA_Bayesian_Mean_Duration'].iloc[0]}")
    return X_train_out, X_val_out, X_test_out



def engineer_friction_matrix_features(X_train, X_val, X_test):
    """
    Architectural Directive 006: Feature Interaction & The "Friction" Matrix.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    def _apply_interactions(df):
        X = df.copy()
        
        # Hardening: Ensure required columns exist, default to 0.0 if missing
        X['LPA_Chaos_Index'] = X.get('LPA_Chaos_Index', 0.5)
        X['[REDACTED_BY_SCRIPT]'] = X.get('[REDACTED_BY_SCRIPT]', 180.0)
        X['[REDACTED_BY_SCRIPT]'] = X.get('[REDACTED_BY_SCRIPT]', 180.0)

        # 1. Risk Amplification
        X['[REDACTED_BY_SCRIPT]'] = X['LPA_Chaos_Index'] * X.get('[REDACTED_BY_SCRIPT]', 0)
        
        sssi_dist = X.get('sssi_dist_to_nearest_m', 5000).fillna(5000)
        X['FI_Paranoia_Factor_SSSI'] = X['[REDACTED_BY_SCRIPT]'] * (1000 / (sssi_dist + 100))
        
        X['[REDACTED_BY_SCRIPT]'] = X['LPA_Chaos_Index'] * X.get('[REDACTED_BY_SCRIPT]', 0)

        # 2. Grid Confusion
        X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X.get('[REDACTED_BY_SCRIPT]', 0)
        
        dnoa_years = X.get('[REDACTED_BY_SCRIPT]', 10).fillna(10)
        X['[REDACTED_BY_SCRIPT]'] = (1 / (dnoa_years + 1)) * X['[REDACTED_BY_SCRIPT]']

        # 3. Scale Shock
        workload = X.get('[REDACTED_BY_SCRIPT]', 100).fillna(100)
        X['[REDACTED_BY_SCRIPT]'] = X.get('[REDACTED_BY_SCRIPT]', 0) / (workload + 1)
        X['[REDACTED_BY_SCRIPT]'] = X.get('[REDACTED_BY_SCRIPT]', 0) * X['LPA_Chaos_Index']

        # 4. NIMBY Weaponization
        nimby_proxy = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
        X['[REDACTED_BY_SCRIPT]'] = nimby_proxy * X['LPA_Chaos_Index']
        X['[REDACTED_BY_SCRIPT]'] = nimby_proxy * X.get('lpa_withdrawal_rate', 0)

        # 5. Stability & Variance Interactions
        X['[REDACTED_BY_SCRIPT]'] = X.get('LPA_Stability_Index', 0) * X.get('[REDACTED_BY_SCRIPT]', 0)
        X['[REDACTED_BY_SCRIPT]'] = X.get('[REDACTED_BY_SCRIPT]', 0) * X.get('[REDACTED_BY_SCRIPT]', 0)
        X['SIC_LPA_CHAOS_APPROVAL'] = X['LPA_Chaos_Index'] * X.get('lpa_major_commercial_approval_rate', 0)
        X['[REDACTED_BY_SCRIPT]'] = X.get('[REDACTED_BY_SCRIPT]', 0) * X.get('LPA_Stability_Index', 0)
        X['[REDACTED_BY_SCRIPT]'] = X.get('[REDACTED_BY_SCRIPT]', 0) * X.get('[REDACTED_BY_SCRIPT]', 0)
            
        return X

    X_train_out = _apply_interactions(X_train)
    X_val_out = _apply_interactions(X_val)
    X_test_out = _apply_interactions(X_test)
    
    return X_train_out, X_val_out, X_test_out

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
    # Robust construction handling potential missing days/months (default to mid-point if missing)
    X['temp_day'] = X['submission_day'].fillna(15).astype(int)
    X['temp_month'] = X['submission_month'].fillna(6).astype(int)
    X['submission_date'] = pd.to_datetime(dict(year=X['submission_year'], month=X['temp_month'], day=X['temp_day']))
    X.drop(columns=['temp_day', 'temp_month'], inplace=True)

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
    # Mapping dates to distinct policy environments.
    # 0: Pre-2015 (Stable)
    # 1: 2015-2019 (Subsidy Removal / Low Volume)
    # 2: 2020-2021 (COVID Administrative Lag)
    # 3: 2022-2023 (BMV Scrutiny / Mini-Budget Chaos)
    # 4: 2024+ (Transition / Election Flux)
    
    def get_policy_regime(year):
        if year < 2015: return 0
        elif 2015 <= year <= 2019: return 1
        elif 2020 <= year <= 2021: return 2
        elif 2022 <= year <= 2023: return 3
        else: return 4

    X['SIC_POLICY_REGIME_ID'] = X['submission_year'].apply(get_policy_regime)
    
    # Directive 3: Election Proximity Cycle (Refinement)
    # Already calculated '[REDACTED_BY_SCRIPT]', adding the Purdah Flag.
    X['SIC_IS_PURDAH_PERIOD'] = (X['[REDACTED_BY_SCRIPT]'] < 45).astype(int) # Approx 6 weeks

    # Step 2: Engineer the Comprehensive Interaction Set (Refactored to use Regime/Cycle)
    # Replacing 'year_norm' with 'SIC_POLICY_REGIME_ID' to capture non-linear shifts.
    
    # Grid Saturation Interactions
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * X['[REDACTED_BY_SCRIPT]']
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    # Grid Headroom Decay: Headroom is less valuable in later regimes due to queue congestion.
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * (1 / (X['[REDACTED_BY_SCRIPT]'] + 1))

    # Environmental Policy Hardening Interactions
    # BMV Conflict is highest in Regime 3 (2022-23). We create a specific interaction for this peak.
    X['[REDACTED_BY_SCRIPT]'] = (X['SIC_POLICY_REGIME_ID'] == 3).astype(int) * X['alc_is_bmv_at_site']
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * X['cs_on_site_bool']
    # Ancient Woodland Veto is constant/increasing, Regime ID works as linear proxy here.
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] / (X['aw_dist_to_nearest_m'] + 100)

    # LPA Behavior & Precedent Interactions
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * X['lpa_major_commercial_approval_rate']
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] / (X['[REDACTED_BY_SCRIPT]'] + 1)
    # Election Paralysis Impact on Workload
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_IS_PURDAH_PERIOD'] * X['lpa_workload_trend']

    # Project Scale & Economics Interactions
    # REMOVED: X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['solar_site_area_sqm']

    # Socio-Economic & Cumulative Impact Interactions
    # REMOVED: X['TFI_NIMBY_Amplifier'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']
    
    # Step 3: Mandated Sanitization and Normalization of New Features.
    tfi_features = [col for col in X.columns if col.startswith('TFI_')]
    
    # HARDENING GATE: Replace non-finite values created by division operations.
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    if imputer is None: # Training mode: fit new imputer and scaler
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        # Fit and transform in training mode
        X[tfi_features] = imputer.fit_transform(X[tfi_features])
        X[tfi_features] = scaler.fit_transform(X[tfi_features])
    else: # Testing/inference mode: use pre-fitted transformers
        fittable_tfi_features = [f for f in tfi_features if f in X.columns]
        
        # Transform using fitted objects
        X[fittable_tfi_features] = imputer.transform(X[fittable_tfi_features])
        X[fittable_tfi_features] = scaler.transform(X[fittable_tfi_features])
        
    return X, baseline_year, imputer, scaler


def engineer_lpa_congestion_metrics(df: pd.DataFrame, target_col='[REDACTED_BY_SCRIPT]') -> pd.DataFrame:
    """
    REMEDIATED: Input-Only Congestion Metrics.
    Removes all dependencies on 'target_col' (planning_duration_days).
    Calculates congestion based purely on the VOLUME of submissions, which is known at T0.
    """
    print("[REDACTED_BY_SCRIPT]")
    X = df.copy()
    
    # Ensure temporal sort
    if 'submission_date' not in X.columns:
        X['submission_date'] = pd.to_datetime(X[['submission_year', 'submission_month', 'submission_day']])
    
    # Sort is crucial for the index-based rolling trick to align
    X.sort_values(['lpa_name', 'submission_date'], inplace=True)

    # 1. Rolling Submission Volume (The "In-Tray" Proxy)
    # How many applications did this LPA receive in the last 6 months?
    # This proxies the workload without knowing when they finish.
    
    # Manual iteration to avoid groupby().apply() DataFrame return issues and FutureWarnings
    rolling_vol_series = pd.Series(index=X.index, dtype='float64')
    
    for _, sub_df in X.groupby('lpa_name'):
        # Create a Series with DatetimeIndex. Values don't matter for count.
        ts = pd.Series(1, index=sub_df['submission_date'])
        # Rolling count over 180 days, closed='left' to exclude self
        rolled = ts.rolling('180D', closed='left').count()
        # Assign back using original index
        rolling_vol_series.loc[sub_df.index] = rolled.values

    X['[REDACTED_BY_SCRIPT]'] = rolling_vol_series.fillna(0)

    # 2. Congestion Severity (Volume / Experience)
    # High volume is only bad if the LPA is inexperienced.
    epsilon = 1e-6
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['lpa_total_experience'] + epsilon)

    # 3. Relative Scale Attention
    # Am I bigger than the average thing currently landing on their desk?
    # We need a rolling average of capacity, NOT duration.
    rolling_cap_series = pd.Series(index=X.index, dtype='float64')
    
    for _, sub_df in X.groupby('lpa_name'):
        ts = pd.Series(sub_df['[REDACTED_BY_SCRIPT]'].values, index=sub_df['submission_date'])
        rolled = ts.rolling('180D', closed='left').mean()
        rolling_cap_series.loc[sub_df.index] = rolled.values
    
    # Fill NaNs (first row of each LPA) with global mean
    global_mean_cap = X['[REDACTED_BY_SCRIPT]'].mean()
    rolling_cap_series.fillna(global_mean_cap, inplace=True)
    
    X['SIC_LPA_QUEUE_RELATIVE_SCALE'] = X['[REDACTED_BY_SCRIPT]'] / (rolling_cap_series + epsilon)

    return X


def engineer_temporal_velocity_features(df: pd.DataFrame, target_col='[REDACTED_BY_SCRIPT]') -> pd.DataFrame:
    """
    REMEDIATED: Input-Only Velocity Metrics.
    Removes rolling averages of the target variable to prevent leakage.
    """
    print("[REDACTED_BY_SCRIPT]")
    X = df.copy()
    
    if 'submission_date' not in X.columns:
        X['submission_date'] = pd.to_datetime(X[['submission_year', 'submission_month', 'submission_day']])
    X.sort_values('submission_date', inplace=True)

    # 1. National System Load (Global Pulse)
    # Instead of rolling duration, we check rolling VOLUME of applications nationally.
    # High volume often correlates with slower times (system stress).
    X['[REDACTED_BY_SCRIPT]'] = X.rolling('90D', on='submission_date', closed='left')['submission_date'].count()
    
    # Normalize by a baseline (e.g., mean volume) to make it a ratio
    baseline_vol = X['[REDACTED_BY_SCRIPT]'].mean()
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (baseline_vol + 1e-6)

    # 2. Seasonal Index (Month-based, not target-based)
    # We simply use the month sine/cosine which you likely already have, 
    # or a simple mapping if you want to encode "Winter Slump".
    # We DO NOT calculate the average duration of previous Junes.
    X['SIC_TEMP_IS_WINTER'] = X['submission_date'].dt.month.isin([11, 12, 1, 2]).astype(int)

    return X




# --- Mandated Semantic Cohort: LPA_ALL (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Purpose: Models the behavioral signatures, historical performance, and local precedent landscape of the human decision-making body (the LPA).
COHORT_LPA_ALL = [
    # LPA Workload & Efficiency Metrics
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    'lpa_withdrawal_rate',
    '[REDACTED_BY_SCRIPT]',
    'lpa_total_experience',
    'lpa_workload_trend',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    # LPA Disposition & Bias Metrics
    'lpa_major_commercial_approval_rate',
    'lpa_approval_rate_cps2',
    '[REDACTED_BY_SCRIPT]',
    # Local Historical Precedent (Legacy & Dynamic)
    'nearby_legacy_count',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',

    # --- AD-AM-05: Bayesian Risk & Stability ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'LPA_Chaos_Index', 
    'LPA_Stability_Index', '[REDACTED_BY_SCRIPT]',

    # --- AD-AM-19: Congestion & Queue Dynamics ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'SIC_LPA_QUEUE_RELATIVE_SCALE',

    # --- Behavioral Interaction Constructs (SICs) ---
    '[REDACTED_BY_SCRIPT]', 'SIC_LPA_RENEWABLE_BIAS', 'SIC_LPA_INSTABILITY_RISK', 
    '[REDACTED_BY_SCRIPT]', 'SIC_LPA_DECISION_CHAOS', '[REDACTED_BY_SCRIPT]',
    'SIC_LPA_POLITICAL_VOLATILITY', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    
    # --- Infrastructure Competence & Profiling ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'SIC_LPA_SPEED_VS_CAPACITY',
    'SIC_LPA_EFFICIENCY_INDEX', '[REDACTED_BY_SCRIPT]', 'SIC_LPA_POLITICAL_RESISTANCE',

    # --- Ground Mount Specific Variance ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    
    # --- Stratified Regime Interactions ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',

    # --- Aggregated Socio-Economic Context ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
]

# --- Mandated Semantic Cohort: GRID_ALL (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Purpose: A complete digital twin of the physical grid. Models the unforgiving laws of physics and electrical engineering.
COHORT_GRID_ALL = [
    # Substation Proximity, Capacity & Stability
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'avg_max_ef_fault_5nn_ka',
    # DER Density (Proxy for LV Saturation & Sentiment)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # Physical Asset Base (Legacy Proximity-Based)
    '[REDACTED_BY_SCRIPT]', 'avg_total_kva_5nn', '[REDACTED_BY_SCRIPT]',
    # IDNO Precedent (Proxy for Private Development Tolerance)
    'idno_is_within', '[REDACTED_BY_SCRIPT]', 'idno_count_in_5km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'lpa_idno_area_as_percent_of_total_area', '[REDACTED_BY_SCRIPT]',
    # Forward-Looking Strategic Context (LTDS)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'ltds_count_in_10km',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # LCT Saturation & Synthesis (Ground-Truth Socio-Technical Layer)
    '[REDACTED_BY_SCRIPT]', 'lct_total_connections_in_5km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'lct_ev_connections_in_5km', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'lct_reconciliation_delta_connections', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # LV Constraints (DNOA - Known Future Problems)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    'dnoa_avg_dist_knn_m', 'dnoa_avg_deferred_kva_knn', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # Power Quality (Network Stability)
    'pq_idw_thd_knn5', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    'pq_max_thd_in_knn5', 'pq_std_thd_in_knn5',
    # Physical Asset Reality (Primary & Secondary Substations)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'sitevoltage', 'powertransformercount', 'transratingwinter', 'transratingsummer',
    'maxdemandsummer', 'maxdemandwinter', 'resistance_ohm', 'reversepower_encoded', 'substation_age_at_submission', 'is_hot_site',
    '[REDACTED_BY_SCRIPT]', 'has_demand_data', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'kva_per_transformer',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'sec_sub_gov_area_sqm',
    # OHL Topology (Connection Friction & Visual Amenity)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    'ohl_local_structure_count', 'ohl_local_tower_count', 'ohl_local_tower_ratio', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'ohl_nearest_voltage', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # Deterministic Operational Reality (Service Area Polygons)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    # --- Phase V Addendum Features ---
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
]

# --- Mandated Semantic Cohort: ENV_ALL (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Purpose: Models the hard constraints and policy friction from statutory environmental law and land use designations.
COHORT_ENV_ALL = [
    # Statutory Designations (SSSIs, National Parks, AONBs, SACs, SPAs)
    'sssi_dist_to_nearest_m', 'sssi_is_within', 'sssi_count_in_2km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'sssi_unit_dist_to_nearest_m', 'sssi_unit_worst_condition_in_2km',
    'np_dist_to_nearest_m', 'np_is_within', 'np_count_in_2km', '[REDACTED_BY_SCRIPT]', 'lpa_np_coverage_pct',
    'aonb_dist_to_nearest_m', 'aonb_is_within', 'aonb_count_in_2km', 'aonb_total_area_in_5km', '[REDACTED_BY_SCRIPT]',
    'sac_dist_to_nearest_m', 'sac_is_within', 'sac_count_in_2km', '[REDACTED_BY_SCRIPT]', 'lpa_sac_coverage_pct',
    'spa_dist_to_nearest_m', 'spa_is_within', 'spa_count_in_2km', '[REDACTED_BY_SCRIPT]', 'lpa_spa_coverage_pct',
    # Land Use Policy (Agricultural Land Classification - ALC)
    'alc_grade_at_site', 'alc_is_bmv_at_site', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # Agri-Environment Schemes (Countryside Stewardship - CS, Environmental Stewardship - ES)
    'cs_on_site_bool', 'cs_on_site_pct_area', 'cs_on_site_total_value', 'cs_count_2km', 'cs_density_2km', 'cs_density_5km', 'cs_density_10km', 'cs_density_20km', '[REDACTED_BY_SCRIPT]', 'cs_avg_value_20km',
    'es_on_site_bool', 'es_count_2km', 'es_total_area_ha_2km', 'es_hls_on_site_bool', 'es_hls_on_site_pct_area', 'es_hls_density_2km', 'es_hls_density_5km', 'es_hls_density_10km', 'es_hls_density_20km',
    # Ecological Fabric (Ancient Woodland - AW, Priority Habitats - PH)
    'aw_dist_to_nearest_m', 'aw_is_within', 'aw_count_in_2km', 'aw_total_area_in_5km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'aw_AWP_count_in_2km',
    'ph_dist_to_nearest_m', 'ph_is_within', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'ph_nearest_is_priority',
    # Public Amenity & Heritage (National Trails - NT, CRoW Access Land, Historic Parkland - HP)
    'nt_dist_to_nearest_m', 'nt_intersection_count_in_2km', 'nt_length_in_2km',
    '[REDACTED_BY_SCRIPT]', 'crow_is_within', '[REDACTED_BY_SCRIPT]', 'crow_nearest_is_rcl', 'crow_nearest_is_s15',
    'hp_dist_to_nearest_m', 'hp_is_within', 'hp_count_in_2km', 'hp_total_area_in_5km_ha', 'hp_count_in_5km',
    'hp_count_in_10km', 'hp_count_in_20km', 'hp_total_area_in_2km_ha', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'hp_nearest_area_ha',
]

# --- Mandated Semantic Cohort: SOCIO_ECONOMIC_ALL (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Purpose: Models the human landscape, including demographics, urban pressure, and proxies for social opposition.
COHORT_SOCIO_ECONOMIC_ALL = [
    # Rural-Urban Classification & Scores
    'site_lsoa_ruc_rural_score', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    # Output Area Classification (Demographic Personas)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'site_lsoa_oac_e-Veterans', '[REDACTED_BY_SCRIPT]',
    # Pre-calculated Strategic Interaction Constructs (SICs)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # Deprivation & Value Deltas
    'delta_property_value', 'delta_env_health_disadvantage',
    # High-Resolution Land Cover (Human Settlement Proxy)
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # --- Phase V Addendum Features ---
    '[REDACTED_BY_SCRIPT]',

    # --- Expanded Socio-Economic Indicators (Wealth, Education, Housing) ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 
    
    # --- Comparative Context (Site vs LPA) ---
    '[REDACTED_BY_SCRIPT]', 'lpa_avg_property_value', 
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    
    # --- Settlement Dynamics & Sprawl ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    
    # --- Advanced Socio-Political SICs ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
]

# --- Mandated Semantic Cohort: NATIONAL_PROXY_ALL (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Purpose: Models the national physical fabric from OS & other national datasets. The foundation for the Amaryllis Bridge, retained for context.
COHORT_NATIONAL_PROXY_ALL = [
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'railway_length_1km', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'railway_length_2km', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'railway_length_5km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'railway_length_10km', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # --- Phase V Addendum Features ---
    '[REDACTED_BY_SCRIPT]',

    # --- Expanded National Fabric (Topography & Settlement) ---
    'mean_terrain_gradient_1km', 'mean_terrain_gradient_2km', 'mean_terrain_gradient_5km', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    
    # --- Macro-Spatial Context ---
    '[REDACTED_BY_SCRIPT]', 
    '[REDACTED_BY_SCRIPT]', # Key national infrastructure node
    '[REDACTED_BY_SCRIPT]', # National grid visibility
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',

    # --- Agricultural Land Classification (National Dataset) ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
]



# --- AD-AM-20: New Specialist Cohorts ---
COHORT_PUBLIC_AMENITY = [
    'nt_dist_to_nearest_m', 'nt_intersection_count_in_2km', 'nt_length_in_2km', 'nt_intersection_count_in_5km',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'nt_length_in_5km', 'nt_length_in_10km',
    'nt_length_in_20km', 'nt_intersects_site_bool',
    '[REDACTED_BY_SCRIPT]', 'crow_is_within', '[REDACTED_BY_SCRIPT]', 'crow_nearest_is_rcl', 'crow_nearest_is_s15',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    'hp_dist_to_nearest_m', 'hp_is_within', 'hp_count_in_2km', 'hp_total_area_in_5km_ha', 'hp_count_in_5km',
    'hp_count_in_10km', 'hp_count_in_20km', 'hp_total_area_in_2km_ha', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'hp_nearest_area_ha',

    # --- Expanded Amenity Context (Visual & Landscape) ---
    'aonb_dist_to_nearest_m', 'aonb_is_within', 'aonb_count_in_2km', 'aonb_total_area_in_5km',
    'np_dist_to_nearest_m', 'np_is_within', 'np_count_in_2km', '[REDACTED_BY_SCRIPT]', 'lpa_np_coverage_pct',
    'aw_dist_to_nearest_m', 'aw_is_within', # Ancient Woodland is a key public amenity
    
    # --- Visual Intrusion & Clutter (Negative Amenity) ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
    '[REDACTED_BY_SCRIPT]', # High gradient = high visibility of the site
    
    # --- Population Pressure (The "Eyes" on the Amenity) ---
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', # Wealthy areas defend amenity more vigorously
    
    # --- Strategic Interaction Constructs (SICs) - Amenity Specific ---
    '[REDACTED_BY_SCRIPT]',
    'SIC_DEMO_AMENITY_DEFENCE_INTENSITY',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
]

COHORT_AGRI_ENVIRONMENTAL_SCHEMES = [
    # --- Base Metrics (Existing) ---
    'cs_on_site_bool', 'cs_on_site_pct_area', 'cs_on_site_total_value', 'cs_count_2km', 'cs_density_2km',
    'cs_density_5km', 'cs_density_10km', 'cs_density_20km', '[REDACTED_BY_SCRIPT]', 'cs_avg_value_20km',
    'cs_on_site_area_ha', 'cs_on_site_highest_tier_Mid_Tier', 'cs_count_5km', 'cs_count_10km', 'cs_count_20km',
    'cs_total_area_ha_2km', 'cs_total_area_ha_5km', '[REDACTED_BY_SCRIPT]', 'cs_avg_value_2km',
    'cs_avg_value_5km', 'cs_avg_value_10km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
    'es_on_site_bool', 'es_count_2km', 'es_total_area_ha_2km', 'es_hls_on_site_bool', 'es_hls_on_site_pct_area',
    'es_hls_density_2km', 'es_hls_density_5km', 'es_hls_density_10km', 'es_hls_density_20km',
    'es_count_5km', 'es_count_10km', 'es_count_20km', 'es_total_area_ha_5km', 'es_total_area_ha_10km',
    'es_total_area_ha_20km', 'es_hls_count_2km', 'es_hls_count_5km', 'es_hls_count_10km', 'es_hls_count_20km',
    'es_hls_avg_cost_2km', 'es_hls_avg_cost_5km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',

    # --- Expanded Feature Set (New Additions) ---
    
    # 1. Value Intensity Ratios (Cost per Hectare)
    # Captures the "richness" of the schemes. High value/ha implies complex, high-biodiversity agreements.
    'cs_value_per_ha_2km', 'cs_value_per_ha_5km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',

    # 2. Scheme Density Gradients (Focus Ratios)
    # Measures if schemes are clustered locally (high friction) or diffuse regionally.
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',

    # 3. Tier Composition Ratios
    # What % of local schemes are High Tier (Strict) vs Mid Tier (Flexible)?
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',

    # 4. Area Saturation Metrics
    # What percentage of the surrounding land is locked up in schemes?
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', # Sum of CS and ES area

    # 5. Scheme Overlap & Complexity
    # Does the area have both CS and ES schemes? (Legacy + New overlap = deep entrenchment)
    '[REDACTED_BY_SCRIPT]', # (cs_count + es_count)
    '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',

    # 6. On-Site Specific Interactions
    # Interaction of on-site scheme value with project scale (Capacity)
    'SIC_CS_VALUE_X_CAPACITY', 
    'SIC_ES_HLS_AREA_X_CAPACITY',
    
    # 7. Regional Value Context
    # Is the local area (2km) more expensive than the wider region (20km)?
    '[REDACTED_BY_SCRIPT]', 
    '[REDACTED_BY_SCRIPT]',

    # 8. Count vs Area Discrepancy (Fragmentation Proxy)
    # High count but low area = fragmented small farms. Low count high area = large estates.
    'cs_avg_area_per_agreement_2km', '[REDACTED_BY_SCRIPT]',
    'es_avg_area_per_agreement_2km', 'es_avg_area_per_agreement_10km'
]




# ==============================================================================
# PHASE 0: GLOBAL CONTEXT VECTOR FORGING (AD-AMARYLLIS-MOD-11)
# ==============================================================================

def phase_0_tune_and_train_oracle_compressed(X_train, y_train, n_trials=30):
    """
    Forges the Global Solar Oracle using a Hybrid Compressed Architecture.
    
    Architectural Logic:
    1. PROBE: Determines Feature Importance using a raw LightGBM scout.
    2. SPLIT: Separates the feature space into 'Signal Core' (Top N) and 'Noise Tail' (The Rest).
    3. COMPRESS: Applies Optimized PCA to the 'Noise Tail' to extract latent variance without overfitting.
    4. FUSE: Concatenates Raw Signal + Compressed Latent features.
    5. TUNE: Optimizes N, PCA Components, and Regressor Hyperparameters simultaneously.
    """
    
    X_solar = X_train.copy()
    y_solar = y_train.copy()

    # 2. The Probe: Establish Feature Hierarchy
    print("[REDACTED_BY_SCRIPT]")
    probe = lgb.LGBMRegressor(random_state=67, n_jobs=-1, verbosity=-1, min_split_gain=0.01, n_estimators=1250, learning_rate=0.001, num_leaves=11, max_depth=7, subsample=0.8, colsample_bytree=0.8, min_child_samples=20)
    probe.fit(X_solar, y_solar)
    
    importances = pd.DataFrame({
        'feature': X_solar.columns,
        'gain': probe.feature_importances_
    }).sort_values('gain', ascending=False)
    
    ranked_features = importances['feature'].tolist()
    print(f"[REDACTED_BY_SCRIPT]")

    # 3. Optimization Loop: Architecture + Hyperparameters
    def objective(trial):
        # --- A. Architectural Hyperparameters ---
        # How many features to keep Raw? (The Signal Core)
        n_top = trial.suggest_int('n_top_features', 100, 400)
        # How much to compress the Tail? (The Latent Space)
        n_components = trial.suggest_int('n_pca_components', 30, 80)
        
        # Guard clause: Ensure we don't ask for more PCA components than tail features
        n_tail_available = len(ranked_features) - n_top
        if n_tail_available < n_components:
            n_components = max(1, n_tail_available) # Compress to available or 1

        # --- B. Data Transformation ---
        top_cols = ranked_features[:n_top]
        tail_cols = ranked_features[n_top:]
        
        # Split
        X_top = X_solar[top_cols].values
        X_tail = X_solar[tail_cols].values
        
        # Pipeline for Tail: Impute -> Scale -> PCA
        # Note: PCA requires scaling. Imputation required for PCA (unlike LGBM).
        tail_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler(quantile_range=(5.0, 95.0))), # Ignore extreme 5% outliers
            ('pca', PCA(n_components=20, random_state=67, whiten=True)) # Whiten to normalize variance
        ])
        
        try:
            X_tail_compressed = tail_pipeline.fit_transform(X_tail)
        except ValueError:
            # Fallback if PCA fails (e.g. zero variance), return bad score
            return float('inf')
            
        # Fuse: Concatenate Raw Top + Compressed Tail
        X_fused = np.hstack([X_top, X_tail_compressed])
        
        # --- C. Model Hyperparameters ---
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 0.1, log=True)
        }
        
        # --- D. Evaluation ---
        # Using TimeSeriesSplit to respect temporal causality
        score = cross_val_score(
            lgb.LGBMRegressor(**params), 
            X_fused, y_solar, 
            cv=TimeSeriesSplit(n_splits=SPLITS_GLOBAL), 
            scoring='neg_root_mean_squared_error'
        ).mean()
        
        return -score

    # Execute Optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=(n_trials+10))
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]'n_top_features'[REDACTED_BY_SCRIPT]'n_pca_components']} PCA Components")

    # 4. Forge Final Artifacts
    best_params = study.best_params
    n_top_final = best_params.pop('n_top_features')
    n_pca_final = best_params.pop('n_pca_components')
    
    # Re-construct the winning pipeline
    top_cols_final = ranked_features[:n_top_final]
    tail_cols_final = ranked_features[n_top_final:]
    
    tail_pipeline_final = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler(quantile_range=(5.0, 95.0))), # Ignore extreme 5% outliers
        ('pca', PCA(n_components=n_pca_final, random_state=67, whiten=True)) # Whiten to normalize variance
    ])
    
    # Fit Transformations on Full Solar Set
    X_tail_final = X_solar[tail_cols_final].values
    X_tail_compressed_final = tail_pipeline_final.fit_transform(X_tail_final)
    X_top_final = X_solar[top_cols_final].values
    X_fused_final = np.hstack([X_top_final, X_tail_compressed_final])
    
    # Train Final Oracle
    final_oracle = lgb.LGBMRegressor(**best_params, random_state=67, n_jobs=-1, verbosity=-1)
    final_oracle.fit(X_fused_final, y_solar)
    

    # top_cols_final = top_cols_final[:int(len(top_cols_final)/4)]  # passes 25% of features to inference
    # tail_cols_final = tail_cols_final[:int(len(tail_cols_final)/4)]
    


    # Bundle Artifacts for Inference
    compression_artifacts = {
        'top_features': top_cols_final,
        'tail_features': tail_cols_final,
        'tail_pipeline': tail_pipeline_final
    }
    
    return final_oracle, compression_artifacts

def apply_oracle_compression(X_df, compression_artifacts):
    """
    Helper function to apply the learned compression scheme to Validation/Test data.
    """
    top_cols = compression_artifacts['top_features']
    tail_cols = compression_artifacts['tail_features']
    pipeline = compression_artifacts['tail_pipeline']
    
    # Check for missing columns (resilience)
    available_top = [c for c in top_cols if c in X_df.columns]
    available_tail = [c for c in tail_cols if c in X_df.columns]
    
    # Extract
    X_top = X_df[available_top].values
    X_tail = X_df[available_tail].values
    
    # Handle case where test set might have missing columns by padding? 
    # For now, assuming rigorous schema alignment from previous steps.
    
    # Transform
    X_tail_compressed = pipeline.transform(X_tail)
    
    # Fuse
    X_fused = np.hstack([X_top, X_tail_compressed])
    return X_fused


def phase_0b_identify_and_excise_unlearnable_samples(X, y, groups, outlier_threshold=300, n_splits=5):
    """
    Architectural Mandate: Implements a robust, cross-validation-based protocol
    to identify and flag unlearnable samples (i.e., those with consistently high
    prediction errors) before they contaminate the training process. This replaces
    the flawed post-hoc cleaning of the test set.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    
    oof_predictions = pd.Series(index=X.index, dtype=float)
    
    # Hardening Gate: Ensure all columns are numeric for the probe model to prevent crashes.
    X_probe = X.select_dtypes(include=np.number)

    # Robust Group Handling
    n_groups = groups.nunique()
    if n_groups < n_splits:
        if n_groups <= 1:
            print(f"[REDACTED_BY_SCRIPT]")
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            splitter = cv.split(X_probe, y)
        else:
            print(f"[REDACTED_BY_SCRIPT]")
            n_splits = n_groups
            cv = GroupKFold(n_splits=n_splits)
            splitter = cv.split(X_probe, y, groups=groups)
    else:
        cv = GroupKFold(n_splits=n_splits)
        splitter = cv.split(X_probe, y, groups=groups)
    
    # Use a simple, robust probe model. No need for extensive tuning.
    probe_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    
    for fold, (train_idx, val_idx) in enumerate(splitter):
        print(f"[REDACTED_BY_SCRIPT]")
        X_train_fold, X_val_fold = X_probe.iloc[train_idx], X_probe.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        
        probe_model.fit(X_train_fold, y_train_fold)
        preds = probe_model.predict(X_val_fold)
        oof_predictions.iloc[val_idx] = preds
        
    # Calculate out-of-fold errors
    oof_errors = (y - oof_predictions).abs()
    
    # Identify indices of unlearnable samples
    unlearnable_indices = oof_errors[oof_errors > outlier_threshold].index
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    return unlearnable_indices


# ==============================================================================
# PHASE 1: DATA PREPARATION & STRATEGIC FEATURE ENGINEERING
# ==============================================================================

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df

def engineer_post_knn_sics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate AD-AM-43:
    Engineers SICs that have a dependency on the features generated by the
    k-NN Anomaly Detection Sub-System. This function is executed post-kNN
    to resolve procedural dependency violations.
    """
    X = df.copy()
    
    # SIC-LPA-04 (Relocated): Precedent Uncertainty
    # Combines k-NN signals to create a measure of how reliable local precedent is. High entropy (diverse neighboring LPAs)
    # and high variance in outcomes for similar projects signals an unpredictable environment.
    if 'knn_lpa_entropy_gcv' in X.columns and '[REDACTED_BY_SCRIPT]' in X.columns:
        X['[REDACTED_BY_SCRIPT]'] = X['knn_lpa_entropy_gcv'] * X['[REDACTED_BY_SCRIPT]']
    else:
        # Failsafe: if k-NN features are missing, create a null column to prevent downstream errors.
        X['[REDACTED_BY_SCRIPT]'] = 0
    
    return X


def engineer_strategic_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate AD-AMARYLLIS-MOD-07 & Phase V Addendum:
    Engineers high-order interaction features to test specific, real-world hypotheses.
    This is the core of the Specialist Interaction Cohort (SIC) strategy and advanced intelligence synthesis.
    """
    X = df.copy()
    epsilon = 1e-6 # Defensive constant to prevent division by zero.

    # --- Legacy SICs (AD-AMARYLLIS-MOD-07) ---
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['lpa_major_commercial_approval_rate'] + epsilon)
    
    # REMOVED: sic_affluent_nimby_risk is 100% missing
    # X['SIC_NIMBY_AMPLIFIER'] = (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]']) / (X['nt_dist_to_nearest_m'] + 100)
    
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])

    # REMOVED: aonb_is_within is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (raw_visual_clutter * landscape_sensitivity_multiplier) / industrial_mitigation_denominator


    # --- AD-AM-23: Modulated Scale Penalty Stabilization ---
    # The previous multiplicative scale penalties (AD-AM-22) proved too volatile.
    # This intervention replaces them with stable, conditional risk flags. The signal
    # is now a binary indicator of a confluence of risk factors, modulated by time.
    
    # Define robust, data-driven thresholds to avoid magic numbers.
    large_scale_threshold = X['[REDACTED_BY_SCRIPT]'].quantile(0.75)
    grid_stress_threshold = X['[REDACTED_BY_SCRIPT]'].quantile(0.75)
    env_proximity_threshold = X['[REDACTED_BY_SCRIPT]'].quantile(0.25) # Lower quantile means closer/higher risk
    tough_lpa_threshold = X['lpa_major_commercial_approval_rate'].quantile(0.25)   # Lower quantile means tougher LPA
    
    # Create binary flags based on these dynamic thresholds.
    is_large_scale = (X['[REDACTED_BY_SCRIPT]'] > large_scale_threshold)
    is_grid_stressed = (X['[REDACTED_BY_SCRIPT]'] > grid_stress_threshold)
    is_env_proximate = (X['[REDACTED_BY_SCRIPT]'] < env_proximity_threshold)
    is_tough_lpa = (X['lpa_major_commercial_approval_rate'] < tough_lpa_threshold)

    # Engineer the new, stable, time-modulated interaction features.
    # REPLACEMENT: 'year_norm' replaced with 'SIC_POLICY_REGIME_ID' per AD-AM-19.
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * (is_large_scale & is_grid_stressed)
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * (is_large_scale & is_env_proximate)
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * (is_large_scale & is_tough_lpa)

    # --- AD-AM-24: Stabilize installed_capacity_mwelec via Amenity Context ---
    # Hypothesis: The impact of project scale is conditional on its proximity to public amenity.
    # These SICs provide that context, stabilizing the raw signal.
    
    # 1. Amenity Conflict Intensity: Models the acute risk from a large project being close to any valued amenity.
    # The signal is amplified exponentially as distance to the nearest amenity receptor decreases.
    amenity_proximity_denominator = X['nt_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100 # Defensive epsilon
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / amenity_proximity_denominator

    # 2. Cumulative Amenity Pressure: Models the chronic risk of adding a large project to an area already dense with amenities.
    amenity_density = X['hp_count_in_5km'] + X['nt_intersection_count_in_5km'] + 1 # Defensive epsilon
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * amenity_density

    # --- Phase V Mandated SICs ---
    
    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] * X['[REDACTED_BY_SCRIPT]']) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    
    # SIC-07: Grid Reality Gap
    #X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']
    # SIC-08: LPA Precedent Bias
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] - X['lpa_major_commercial_approval_rate']) * (1 - X['lpa_workload_trend'])
    
    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * (X['cs_density_5km'] + X['[REDACTED_BY_SCRIPT]'])
    # X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']
    
    # --- Phase VI Addendum: Grid-Focused SICs ---
    # Hypothesis: Model the administrative and forward-looking grid pressures not captured by static headroom metrics.
    # SIC-GRID-01: Grid Queue Pressure Proxy
    # A proxy for the connection queue. Pressure is high when many large-scale developments are competing for limited headroom.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    
    # SIC-GRID-02: Forward-Looking Risk
    # Models risk for projects connecting to a grid known to be constrained in the near future. High loading plus a long wait for upgrades is a major red flag.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + 1)
    
    # SIC-GRID-03: Stability Vulnerability
    # Models the risk of connecting a large project to a "soft" part of the grid with low fault level tolerance.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    
    # SIC-GRID-04: DNOA Imminence Penalty
    # A direct penalty for projects near an area with an imminent Distribution Network Operator Assessment (DNOA) constraint.
    X['[REDACTED_BY_SCRIPT]'] = 1 / ((X['[REDACTED_BY_SCRIPT]'] + 100) * (X['[REDACTED_BY_SCRIPT]'] + 1))
    
    # SIC-GRID-05: Primary Substation Saturation
    # Models the stress on the entire primary substation zone, not just the nearest connection point.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    
    # SIC-GRID-06: LCT Integration Stress
    # Models the stress from a high density of Low Carbon Technologies (LCTs) like EVs and heat pumps at the primary substation level.
    X['SIC_LCT_INTEGRATION_STRESS'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # --- Phase VI Addendum: LPA-Focused SICs ---
    # Hypothesis: Model the intra-LPA variance and behavioral nuances not captured by high-level statistics.
    # SIC-LPA-01: Decision Fatigue
    # Models the risk that an overworked LPA (high workload trend) that is already slow (high avg decision days) will delay or refuse complex projects.
    X['[REDACTED_BY_SCRIPT]'] = X['lpa_workload_trend'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-LPA-02: Renewable Bias
    # Isolates an LPA's specific disposition towards renewables by comparing its historical renewable approval rate to its general commercial approval rate.
    X['SIC_LPA_RENEWABLE_BIAS'] = X['[REDACTED_BY_SCRIPT]'] - X['lpa_major_commercial_approval_rate']
    
    # REMOVED: sic_affluent_nimby_risk is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['lpa_total_experience'] + epsilon)

    # SIC-LPA-04: Precedent Uncertainty
    # DECOMMISSIONED from this function and relocated to `engineer_post_knn_sics` to resolve procedural dependency violation.

    # SIC-LPA-05: Instability Risk
    # Models political or procedural instability. An LPA that frequently fails to meet statutory decision deadlines is a higher risk, especially for large-scale projects.
    X['SIC_LPA_INSTABILITY_RISK'] = (1 - X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']
    
    # SIC-LPA-06: Procedural Friction Proxy
    # A high withdrawal rate can be a sign that an LPA encourages applicants to withdraw difficult applications to avoid a formal refusal. This risk is hypothesized to be higher for projects near sensitive constraints.
    X['[REDACTED_BY_SCRIPT]'] = X['lpa_withdrawal_rate'] / ((X['[REDACTED_BY_SCRIPT]'] / 1000) + epsilon) # distance in km

    # --- Phase VI Addendum: ENV-Focused SICs ---
    # Hypothesis: Model the complex conflicts between project scale, landscape sensitivity, and cumulative ecological impact.
    # REMOVED: solar_site_area_sqm and aonb_is_within are 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (X['aonb_dist_to_nearest_m'] + X['np_dist_to_nearest_m'] + 100)

    # SIC-ENV-02: Ecological Fragmentation Risk
    # Penalizes large projects located in an ecological "pinch point" between multiple sensitive habitat types like Ancient Woodland and SSSIs.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['aw_dist_to_nearest_m'] + X['sssi_dist_to_nearest_m'] + X['ph_dist_to_nearest_m'] + 100)

    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * (X['cs_density_5km'] + X['es_hls_density_5km'])
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * X['alc_is_bmv_at_site'] * X['[REDACTED_BY_SCRIPT]']
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (X['nt_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100)
    
    # SIC-ENV-06: Heritage Asset Conflict
    # A focused signal representing the tension between large-scale modern development and the setting of historic parklands.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['hp_dist_to_nearest_m'] + 100)

    # --- Phase VI Addendum: SOCIO-ECONOMIC-Focused SICs ---
    # Hypothesis: Model specific social flashpoints where project characteristics conflict with the local human landscape.
    
    # REMOVED: site_lsoa_ruc_rural_score and solar_site_area_sqm are 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] / 10000) * X['site_lsoa_ruc_rural_score'] * X['[REDACTED_BY_SCRIPT]']
    
    # SIC-SOCIO-02: Urban Fringe Land Use Tension
    # Models the conflict on the edge of settlements, where the pressure to protect green space for amenity meets the pressure for development.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['alc_is_bmv_at_site'] * X['[REDACTED_BY_SCRIPT]']
    
    # REMOVED: sic_affluent_nimby_risk is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['solar_site_area_sqm'] + (X['[REDACTED_BY_SCRIPT]'] * 10000))
    
    # SIC-SOCIO-04: Regeneration Potential
    # The inverse of a conflict. A large project on industrial land in an area with lower property values may be viewed as positive investment, reducing friction.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]']) / (X['delta_property_value'] + epsilon)
    
    # SIC-SOCIO-05: Digital Divide Engagement Risk
    # Models the risk that in areas with high concentrations of "offline" residents, conventional engagement may fail, leading to late-stage objections and delays.
    X['SIC_SOCIO_DIGITAL_DIVIDE_RISK'] = X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]']
    
    # SIC-SOCIO-06: Settlement Proximity Conflict
    # A direct measure of conflict based on the density of human settlement immediately surrounding the site, weighted by the project's scale.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # --- Phase VI Addendum: Cross-Domain "Perfect Storm" SICs ---
    # Hypothesis: Model scenarios where risks from different domains intersect and amplify each other.
    # SIC-XD-01: Procedural Gridlock (LPA x GRID)
    # Models the extreme delay risk when a procedurally inefficient LPA must handle a project on a highly constrained part of the grid.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] * (1 - X['[REDACTED_BY_SCRIPT]'])) * X['[REDACTED_BY_SCRIPT]']
    
    # SIC-XD-02: Environmental Policy Amplification (LPA x ENV)
    # Models how a politically tough LPA (low approval rate) can weaponize a nearby environmental constraint, amplifying its impact.
    X['[REDACTED_BY_SCRIPT]'] = (1 / (X['lpa_major_commercial_approval_rate'] + epsilon)) / (X['[REDACTED_BY_SCRIPT]'] + 100)
    
    # REMOVED: aonb_is_within is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + 1) * (1 + X['aonb_is_within'] + X['np_is_within'])

    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['solar_site_area_sqm'] * X['alc_is_bmv_at_site']

    
    
    # SIC-XD-06: Cumulative Development Pressure (GRID x ENV x SOCIO)
    # An ultimate cumulative impact signal. It combines the density of existing large developments with the density of environmental schemes and human settlement.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + 1) * (X['cs_density_10km'] + 1) * (X['[REDACTED_BY_SCRIPT]'] + 1)

    # --- Phase VII Addendum: Temporal Strategic Interaction Constructs (TSICs) ---
    # Hypothesis: Model the non-linear acceleration of risk factors over time.
    # REPLACEMENT: 'year_norm' replaced with 'SIC_POLICY_REGIME_ID' per AD-AM-19.

    # TSIC-01: Grid Saturation Acceleration
    # Models the accelerating penalty of connecting to a stressed grid as saturation approaches non-linearly.
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] * (X['[REDACTED_BY_SCRIPT]'] ** 2)

    # TSIC-02: Policy Hardening on Veto Constraints
    # Models how protections for "veto" assets (Ancient Woodland, Grade 1 Land) have become nearly absolute over time.
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_POLICY_REGIME_ID'] / (X['aw_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100)

    # TSIC-06: The Erosion of Precedent
    # Models the idea that old planning approvals (legacy successes) become less relevant over time as the policy and physical environment changes.
    X['TSIC_PRECEDENT_EROSION'] = X['SIC_POLICY_REGIME_ID'] / (X['[REDACTED_BY_SCRIPT]'] + 1)

    # --- Phase VIII Addendum: Core Project Characterization SICs ---
    # Hypothesis: Model the intrinsic physical suitability and logistical viability of the project itself.
    
    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['solar_site_area_sqm'] / 10000 + epsilon) # MW per hectare
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-CORE-03: Logistical Access Challenge
    # Models the construction traffic friction. A large-scale project with poor access to major road networks is a significant source of local objection and delay.
    X['SIC_CORE_LOGISTICAL_ACCESS_CHALLENGE'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (industrial_area_sqm + epsilon)

    # SIC-CORE-05: Mounting System vs. Land Quality Conflict
    # A more advanced mounting system (e.g., trackers, often coded as a higher integer) on prime agricultural land represents a more intensive, industrial-style conversion, potentially increasing planning friction.
    # Note: Assumes mounting_type_for_solar is numerically ordered by intensity.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['alc_is_bmv_at_site']

    # SIC-CORE-06: Dislocated CHP Anomaly
    # CHP (Combined Heat and Power) enabled projects are typically co-located with an industrial heat user. A CHP project in a non-industrial area is a significant anomaly that warrants flagging.
    X['[REDACTED_BY_SCRIPT]'] = X['chp_enabled'] * (1 - X['[REDACTED_BY_SCRIPT]'])

    # --- Phase IX Addendum: Advanced Conflict Synthesis SICs ---
    # Hypothesis: Model specific, high-level planning arguments by synthesizing multiple cross-domain features.
    
    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = ((X['solar_site_area_sqm'] + (X['[REDACTED_BY_SCRIPT]'] * 10000)) / (X['nt_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + 100)) * (1 + X['aonb_is_within'] + X['np_is_within'])

    # SIC-SYNTH-02: Grid Reality Gap Synthesis
    # Models the dissonance between theoretical grid capacity and the practical, forward-looking reality. Risk is high when the wait for crucial upgrades is long and known future constraints are imminent.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'] + 1) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-SYNTH-03: LPA Precedent Paradox Synthesis
    # Captures the uncertainty when an LPA's general disposition (e.g., pro-commercial) contradicts its specific historical precedent with renewables. This uncertainty is amplified by high workload.
    X['[REDACTED_BY_SCRIPT]'] = (X['lpa_major_commercial_approval_rate'] - X['[REDACTED_BY_SCRIPT]']).abs() * (1 + X['lpa_workload_trend'])
    
    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * X['[REDACTED_BY_SCRIPT]']
    
    # REMOVED: sic_affluent_nimby_risk is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]'] / (X['sssi_dist_to_nearest_m'] + 100)

    # --- Phase X Addendum: Contextual Anomaly & Ratio SICs ---
    # Hypothesis: Model relative context and flag anomalous project configurations.
    # SIC-CONTEXT-01: Scale vs. Local Grid Capacity Ratio
    # A measure of local grid impact. A high ratio indicates a project that is very large relative to the primary substation it connects to, signaling a potentially disruptive scheme.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / ((X['[REDACTED_BY_SCRIPT]'] / 1000) + epsilon) # MWelec vs MVA

    # SIC-CONTEXT-02: Connection Path Inefficiency
    # Models the inefficiency of requiring a long connection path for a relatively small project. This can indicate suboptimal siting or significant wayleave challenges.
    X['SIC_CONTEXT_CONNECTION_PATH_INEFFICIENCY'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # REMOVED: aonb_is_within is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['aonb_is_within'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-CONTEXT-04: Headroom Data Reliability Risk
    # Uses the discrepancy between forecasted and empirical headroom as a proxy for data uncertainty. High uncertainty on a large project is a significant risk multiplier.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]'].abs()

    # SIC-CONTEXT-05: LPA Inexperience vs. Project Complexity
    # Models the delay risk when a relatively inexperienced LPA has to handle a project with numerous complex environmental constraints.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['lpa_total_experience'] + epsilon)
    
    # SIC-CONTEXT-06: Isolation Anomaly
    # Models the anomaly of a very large project being sited far from any major transport links or urban centers, suggesting potential logistical or socio-economic disconnects.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # --- Phase XI Addendum: Data Uncertainty & High-Resolution Context SICs ---
    # Hypothesis: Model the reliability of input data and the fine-grained topology of risks.
    
    # REMOVED: tx_count_mismatch_flag is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-UNC-02: LCT Data Uncertainty Penalty
    # Models the risk of planning in an area where Low Carbon Technology (LCT) data is incomplete. This uncertainty is most dangerous in grids already under stress.
    X['[REDACTED_BY_SCRIPT]'] = (1 - X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']

    # SIC-GRID-TOPOLOGY-01: Governing vs. Nearest Connection Delta
    # A crucial topological signal. A large difference between the physically nearest substation and the electrically governing substation implies a complex, long, and likely contentious connection path.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] - X['[REDACTED_BY_SCRIPT]']).abs()

    # SIC-IMPACT-01: Ecological Constraint Focus
    # Models the high-resolution clustering of risk. A high value indicates that key ecological constraints (SSSI, Ancient Woodland) are intensely concentrated around the site rather than being diffuse, creating a formidable barrier.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]']

    # SIC-IMPACT-02: High-Voltage Character
    # Differentiates between landscapes with many small lines versus those dominated by large, visually intrusive 132kV towers. The latter often generates more significant landscape character objections.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    
    # SIC-IMPACT-03: Amenity Saturation
    # Models "destination landscapes" - areas with a high density of public amenities relative to the local population, suggesting their value extends beyond immediate residents and thus have a larger pool of potential objectors.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + X['nt_length_in_10km']) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # --- Phase XII Addendum: Operational Reality & Competing Policy SICs ---
    # Hypothesis: Model the dynamic operational state of the grid and the friction from conflicting policy objectives.
    # SIC-GRID-OPS-01: Reverse Power Connection Risk
    # A project connecting to a substation already experiencing reverse power flows is entering a complex, bi-directional grid environment, significantly increasing DNO scrutiny.
    X['SIC_GRID_OPS_REVERSE_POWER_RISK'] = X['reversepower_encoded'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-GRID-OPS-02: Power Quality Degradation Risk
    # Models the risk that a large new generator will exacerbate pre-existing power quality issues (e.g., high harmonic distortion) on the local network, a major concern for DNOs.
    X['[REDACTED_BY_SCRIPT]'] = X['pq_idw_thd_knn5'] * X['[REDACTED_BY_SCRIPT]']

    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['alc_is_bmv_at_site'] * X['cs_on_site_bool'] * X['solar_site_area_sqm']

    # SIC-LOGISTICS-01: Connection Path Tortuosity
    # Models the logistical and legal complexity of the connection path. A high number of intersections per kilometer suggests a tortuous, difficult, and expensive wayleave process.
    X['SIC_LOGISTICS_CONNECTION_TORTUOSITY'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-ECOLOGY-01: Dense Ecological Network Conflict
    # Differentiates between proximity to a single constraint and siting within a dense network of ecological assets. Risk is amplified when a project is near multiple constraints in a concentrated area.
    X['[REDACTED_BY_SCRIPT]'] = (X['sssi_count_in_2km'] + X['aw_count_in_2km'] + X['[REDACTED_BY_SCRIPT]']) / (X['[REDACTED_BY_SCRIPT]'] + 100)

    # REMOVED: cs_on_site_pct_area is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['cs_on_site_total_value'] * X['cs_on_site_pct_area']

    # --- Phase XIII Addendum: Advanced Topological & Political SICs ---
    # Hypothesis: Model the most subtle risks arising from political context, infrastructure age, and complex spatial relationships.
    # SIC-POL-01: LPA Political Volatility Proxy
    # Models risk from procedural chaos. An LPA whose workload is increasing while its decision timing becomes more erratic is a high-risk environment for any major application.
    X['SIC_POL_LPA_POLITICAL_VOLATILITY'] = X['lpa_workload_trend'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-INFRA-01: Aging Asset Risk Compounder
    # A critical operational signal. The planning and engineering risk of connecting to a heavily loaded substation is compounded significantly if that asset is also old.
    X['SIC_INFRA_AGING_ASSET_RISK'] = X['substation_age_at_submission'] * X['[REDACTED_BY_SCRIPT]']

    # REMOVED: sic_affluent_nimby_risk is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (X['nt_intersection_count_in_5km'] + X['hp_count_in_5km'] + 1) * X['[REDACTED_BY_SCRIPT]']

    # SIC-GRID-TOPO-02: Cumulative Generation Pressure
    # Differentiates between a grid constrained by demand vs. one saturated with generators. This models the political "cumulative impact" argument by measuring the new project'[REDACTED_BY_SCRIPT]'s physical limits.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    
    # SIC-LOGISTICS-02: Dense Corridor Friction
    # Models the "[REDACTED_BY_SCRIPT]" connection path. This measures the density of sensitive intersections (roads, rail, etc.) and amplifies the risk if the corridor passes through protected natural areas.
    X['SIC_LOGISTICS_DENSE_CORRIDOR_FRICTION'] = (X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)) * (1 + X['[REDACTED_BY_SCRIPT]'])

    # SIC-10: Economic vs. Ecological Tension
    # Architectural Interpretation: Penalty is inverse distance to nearest veto constraint (Ancient Woodland).
    ecological_penalty = 1 / (X['aw_dist_to_nearest_m'] + 100) # Add 100m to prevent extreme values at close proximity
    X['SIC_ECONOMIC_VS_ECOLOGICAL_TENSION'] = X['[REDACTED_BY_SCRIPT]'] / (X['alc_is_bmv_at_site'] + ecological_penalty + epsilon)

    # --- Phase XIV Addendum: LPA Behavioral Nuance SICs ---
    # Hypothesis: Model the specific, observable behaviors and pressures within an LPA that are not captured by high-level annual statistics.
    # This directly addresses the "Intra-LPA Variance" signal gap identified in the v3.3 diagnostic dossier.

    # SIC-LPA-07: Decision Chaos Proxy
    # Models the risk from an LPA that is both overworked (high workload trend) and procedurally erratic (high decision speed variance).
    # This combination suggests an environment where standard timelines are unreliable.
    X['SIC_LPA_DECISION_CHAOS'] = X['lpa_workload_trend'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-LPA-08: Infrastructure Aversion Signal
    # Isolates an LPA's potential bias against large-scale industrial projects by comparing its specific industrial approval rate
    # to its general commercial approval rate. A large negative value indicates a potential aversion to projects of our type.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] - X['lpa_major_commercial_approval_rate']

    # REMOVED: sic_affluent_nimby_risk is 100% missing
    # X['SIC_LPA_INEXPERIENCE_UNDER_PRESSURE'] = X['[REDACTED_BY_SCRIPT]'] / (X['lpa_total_experience'] + epsilon)

    # SIC-LPA-10: Procedural Instability Risk
    X['[REDACTED_BY_SCRIPT]'] = (1 - X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']

    # [NEW] SIC-LPA-13: Political Volatility Proxy
    # Captures the "Chaos Factor". An LPA that is non-compliant AND has high variance in decision speeds is politically unstable.
    X['SIC_LPA_POLITICAL_VOLATILITY'] = (1 - X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']

    # SIC-LPA-11: "Quiet Refusal" Friction
    X['[REDACTED_BY_SCRIPT]'] = X['lpa_withdrawal_rate'] / (X['[REDACTED_BY_SCRIPT]'] + 100)

    # [NEW] SIC-GRID-13: Administrative Queue Friction
    # A Proxy for the Connection Queue. High Loading (Physical) + Long LTDS Wait (Planning) = High Queue Risk.
    # This differentiates "physically full" from "[REDACTED_BY_SCRIPT]".
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + 1)

    # SIC-LPA-12: Precedent vs. Policy Conflict
    # Captures the tension when an LPA's historical actions (legacy approval rate for renewables) conflict with its current general disposition
    # (overall major commercial approval rate). A large delta suggests that past performance is not a reliable indicator of future results.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] - X['lpa_major_commercial_approval_rate']).abs()

    # --- Phase XV Addendum: Grid Administrative Reality SICs ---
    # Hypothesis: Model the administrative, temporal, and stability-related grid pressures not captured by simple headroom or loading metrics.
    # This directly addresses the "[REDACTED_BY_SCRIPT]" signal gap identified in the v3.3 diagnostic dossier.

    # SIC-GRID-07: Queue Pressure Proxy
    # Models the competition for grid access. High pressure occurs when a large project competes with many other large projects
    # for limited physical headroom at the nearest substation.
    X['SIC_GRID_QUEUE_PRESSURE'] = (X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-GRID-08: Forward-Looking Constraint Risk
    # Models the risk of connecting to a grid known to be constrained in the near future. A project connecting to a heavily loaded
    # substation with a long wait time for planned upgrades is a major red flag for delays.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + 1)

    # SIC-GRID-09: Network Stability Vulnerability
    # Models the engineering risk of connecting a large generator to a "soft" part of the grid with low fault level tolerance.
    # DNOs are extremely cautious about projects that could threaten network stability.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-GRID-10: DNOA Imminence Penalty
    # A direct penalty for projects sited near an area with an imminent Distribution Network Operator Assessment (DNOA) constraint.
    # This feature weaponizes forward-looking DNO data, penalizing proximity to known, near-term problem areas.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / ((X['[REDACTED_BY_SCRIPT]'] + 100) * (X['[REDACTED_BY_SCRIPT]'] + 1))

    # SIC-GRID-11: Primary Substation Saturation
    # Models the stress on the entire primary substation zone, not just the nearest physical connection point. A high ratio indicates
    # the substation is already managing a high load from its downstream network relative to its total capacity.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-GRID-12: LCT Integration Friction
    # Models the complexity of integrating a large generator into a network already saturated with bi-directional Low Carbon
    # Technologies (LCTs). A high generation-to-demand ratio from LCTs at the primary sub level indicates a complex and volatile network.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # --- Phase XVI Addendum: Environmental Conflict Archetype SICs ---
    # Hypothesis: Model the complex conflicts between project scale, landscape sensitivity, and cumulative ecological impact,
    # moving beyond simple proximity to model specific planning "storylines".

    # REMOVED: solar_site_area_sqm and aonb_is_within are 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (X['aonb_dist_to_nearest_m'] + X['np_dist_to_nearest_m'] + 100)

    # SIC-ENV-08: Ecological Fragmentation Risk
    # Penalizes large projects located in an ecological "pinch point" between multiple sensitive habitat types (Ancient Woodland, SSSIs, Priority Habitats),
    # modeling the risk of severing ecological corridors.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['aw_dist_to_nearest_m'] + X['sssi_dist_to_nearest_m'] + X['ph_dist_to_nearest_m'] + 100)

    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * (X['cs_density_5km'] + X['es_hls_density_5km'])
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * X['alc_is_bmv_at_site'] * X['[REDACTED_BY_SCRIPT]']
    # X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (X['nt_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100)

    # SIC-ENV-12: Heritage Setting Conflict
    # A focused signal representing the tension between large-scale modern development and the legally protected "setting" of historic parklands.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['hp_dist_to_nearest_m'] + 100)

    # --- Phase XVII Addendum: Socio-Economic Flashpoint SICs ---
    # Hypothesis: Model specific social flashpoints where project characteristics conflict with the local human landscape,
    # moving beyond generic demographic labels to capture actionable behavioral patterns.

    # REMOVED: solar_site_area_sqm and site_lsoa_ruc_rural_score are 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] / 10000) * X['site_lsoa_ruc_rural_score'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-SOCIO-08: Urban Fringe Land Use Tension
    # Models the conflict on the edge of settlements, where the pressure to protect green space for amenity meets the pressure for development.
    # The tension is highest when the land is also high-quality agricultural land.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['alc_is_bmv_at_site'] * X['[REDACTED_BY_SCRIPT]']

    # REMOVED: sic_affluent_nimby_risk and solar_site_area_sqm are 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['solar_site_area_sqm'] + (X['[REDACTED_BY_SCRIPT]'] * 10000))

    # SIC-SOCIO-10: Regeneration Potential
    # The inverse of a conflict. A large project on industrial land in an area with lower property values may be viewed as a positive investment,
    # creating a "brownfield bonus" that reduces planning friction.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]']) / (X['delta_property_value'] + epsilon)

    # SIC-SOCIO-11: Digital Divide Engagement Risk
    # Models the risk that in areas with high concentrations of "offline" or "withdrawn" residents, conventional digital-first engagement
    # may fail, leading to late-stage objections from a community that feels it was not properly consulted.
    X['SIC_SOCIO_DIGITAL_DIVIDE_RISK'] = X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]']

    # SIC-SOCIO-12: Settlement Proximity Conflict
    # A direct measure of conflict based on the density of human settlement immediately surrounding the site, weighted by the project's scale.
    # This captures the simple reality that large projects next to many houses face more scrutiny.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # --- Phase XVIII Addendum: Explicit Capacity Friction & NSIP Handling ---
    # Hypothesis: Friction is not linear. 50MW in a city is infinitely harder than 5MW.
    
    # SIC-CAP-01: Urban Denial (The "You Can't Build That Here" Factor)
    # Explicitly models the impossibility of building large scale solar in dense urban cells.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-CAP-02: Priority Habitat Variance Amplifier
    # Priority Habitats cause massive delays for large sites due to survey requirements.
    # Small sites often slip through.
    X['SIC_CAPACITY_X_PRIORITY_HABITAT'] = X['[REDACTED_BY_SCRIPT]'] / (X['ph_dist_to_nearest_m'] + epsilon)

    # SIC-CAP-03: Infrastructure Crossing Complexity
    # The complexity of wayleaves scales with power output (bigger cables) and number of crossings.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-CAP-04: Brownfield Scale Opportunity
    # Large projects on brownfield get a "[REDACTED_BY_SCRIPT]" bonus that small ones don't.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])

    # SIC-NSIP-01: National Infrastructure Regime Flag
    # Projects > 50MW fall under a completely different legal regime (DCO vs TCPA).
    # The model MUST know this is a categorical shift, not just a linear increase.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] > 49.9).astype(int)

    # SIC-BESS-01: Co-Location Complexity Proxy
    # High density of sub-1MW DERs often indicates battery storage presence.
    # Interaction with solar capacity suggests a complex "Energy Hub" application.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # --- Phase V Mandated Standalone Features ---
    # Note: '[REDACTED_BY_SCRIPT]' is not available. Using available OHL data as the primary friction source.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    
    # REMOVED: aonb_is_within is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (1 + X['aonb_is_within'] + X['np_is_within']) * (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])
    
    # REMOVED: site_lsoa_ruc_rural_score is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['site_lsoa_ruc_rural_score'] * X['[REDACTED_BY_SCRIPT]']

    # --- AD-AM-15: Fortified NIMBY Interaction Constructs (RELOCATED) ---
    # REMOVED: solar_site_area_sqm is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] + (X['[REDACTED_BY_SCRIPT]'] * 10000)) / (X['nt_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100)
    
    # REMOVED: aonb_is_within is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + 1) * X['[REDACTED_BY_SCRIPT]'] * (X['cs_density_5km'] + 0.1) * (1 + X['aonb_is_within'])
    # X['[REDACTED_BY_SCRIPT]'] = np.log1p(X['[REDACTED_BY_SCRIPT]'])
    # X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']
    
    # REMOVED: sic_affluent_nimby_risk is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['sssi_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + 100)

    # REMOVED: sic_affluent_nimby_risk is 100% missing
    # X['[REDACTED_BY_SCRIPT]'] = (1 - X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']
    # X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']


    # --- Final Normalization Step ---
    # Consolidate all interaction features for robust scaling.
    interaction_features = [f for f in X.columns if f.startswith('SIC_') or f.startswith('feat_')]
    if interaction_features:
        # Normalize interactions to prevent scale dominance.
        scaler = StandardScaler()
        X[interaction_features] = scaler.fit_transform(X[interaction_features])
    return X

def define_all_cohorts(X: pd.DataFrame):
    """
    Defines cohorts.
    REMEDIATION: Sanitized to remove references to dropped features (Area, AONB, etc).
    """
    def _sanitize(features):
        return [re.sub(r'[^A-Za-z0-9_]+', '_', f) for f in features]

    # --- v3.5 Definitive Cohort Manifest ---
    # REMOVED: 'submission_year', 'year_norm' (Toxic Recency Bias)
    temporal_base_features = [
        'SIC_POLICY_REGIME_ID', 'SIC_IS_PURDAH_PERIOD', # New Political/Regime Features
        'submission_month', 'submission_day', 'submission_month_sin', 'submission_month_cos',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
        # 'aonb_is_within', <-- REMOVED
        'alc_is_bmv_at_site', 'cs_on_site_bool', 'aw_dist_to_nearest_m', 'lpa_major_commercial_approval_rate',
        '[REDACTED_BY_SCRIPT]', 'lpa_workload_trend', '[REDACTED_BY_SCRIPT]', 
        # 'solar_site_area_sqm', <-- REMOVED
        # '[REDACTED_BY_SCRIPT]' <-- REMOVED
    ]
    
    lpa_additions = [f for f in X.columns if f.startswith('knn_') or f.startswith('lpa_legacy_')]
    knn_anomaly_features = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]', 'knn_lpa_entropy_gcv'
    ]
    lpa_additions.extend(knn_anomaly_features)
    
    # Directive 006: Add Friction Matrix Features to LPA Cohort
    fi_additions = [f for f in X.columns if f.startswith('FI_')]
    lpa_additions.extend(fi_additions)
    
    flag_additions = [f for f in X.columns if f.startswith('[REDACTED_BY_SCRIPT]') or '_nearest_name_' in f]
    # Removed: '[REDACTED_BY_SCRIPT]' (likely dropped), keeping if present handled by logic below.

    env_additions = [f for f in X.columns if any(p in f for p in ['aonb_', 'aw_', 'np_', 'ph_', 'sac_', 'spa_', 'sssi_']) and f not in flag_additions]
    # Add new Ecological Drag SICs to environment list
    env_additions.extend([f for f in X.columns if f.startswith('SIC_ECO_')])

    grid_additions = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]

    socio_additions = [f for f in X.columns if f.startswith('nhlc_') or f.startswith('site_lsoa_')]

    FORTIFIED_SIC_GRID_POLICY = [
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'pq_idw_thd_knn5',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'lpa_major_commercial_approval_rate', '[REDACTED_BY_SCRIPT]', 'lpa_workload_trend',
        'lpa_legacy_approval_rate',
        '[REDACTED_BY_SCRIPT]', 
        # 'solar_site_area_sqm', <-- REMOVED
        '[REDACTED_BY_SCRIPT]'
    ]
    
    FORTIFIED_SIC_NIMBY = [
        # Vector 1
        # 'SIC_NIMBY_AMPLIFIER', <-- REMOVED
        # '[REDACTED_BY_SCRIPT]', <-- REMOVED
        # '[REDACTED_BY_SCRIPT]', <-- REMOVED
        # '[REDACTED_BY_SCRIPT]', <-- REMOVED

        # Vector 2
        # '[REDACTED_BY_SCRIPT]', <-- REMOVED
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',

        # Vector 3
        # 'aonb_is_within', <-- REMOVED
        'nt_dist_to_nearest_m', '[REDACTED_BY_SCRIPT]', 'sssi_dist_to_nearest_m',
        'hp_dist_to_nearest_m', 'cs_density_5km', 
        # '[REDACTED_BY_SCRIPT]', <-- REMOVED

        # Vector 4
        # 'solar_site_area_sqm', <-- REMOVED
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    
    # --- CONSOLIDATED SPECIALIST COHORT ---
    consolidated_features = []
    consolidated_features.extend(_sanitize(COHORT_CORE_ATTRIBUTES))
    consolidated_features.extend([f for f in X.columns if f.startswith('TFI_')] + temporal_base_features)
    consolidated_features.extend(FORTIFIED_SIC_GRID_POLICY)
    consolidated_features.extend(FORTIFIED_SIC_NIMBY)
    consolidated_features.extend(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'])
    # REMOVED references to SICs dependent on area/missing features from consolidated list
    consolidated_features.extend(['SIC_ECONOMIC_VS_ECOLOGICAL_TENSION', '[REDACTED_BY_SCRIPT]', 'alc_is_bmv_at_site', 'aw_dist_to_nearest_m'])
    # Add Ecological Drag features to consolidated list
    consolidated_features.extend([f for f in X.columns if f.startswith('SIC_ECO_')])
    # Add Friction Matrix features to consolidated list
    consolidated_features.extend([f for f in X.columns if f.startswith('FI_')])
    
    # Add Speculator Detection features to consolidated list
    speculator_features = [
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(speculator_features)

    # Add NSIP & Cliff features (V6.0)
    v6_features = [
        '[REDACTED_BY_SCRIPT]',
        'SIC_NSIP_LPA_BYPASS_FLAG',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v6_features)

    # Add Evasion & Step-Up features (V7.0)
    v7_features = [
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v7_features)

    # Add Rhythm & Agitation features (V8.0)
    v8_features = [
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v8_features)

    # Add Structural Friction features (V9.0)
    v9_features = [
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'SIC_GRID_CONNECTION_EFFICIENCY',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v9_features)

    # Add Common Sense features (V10.0)
    v10_features = [
        'SIC_SOCIAL_OCCUPATION_RATIO',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'SIC_AGRI_SCARCITY_RATIO',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v10_features)

    # Add Mid-Scale Rescue features (V11.0)
    v11_features = [
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'SIC_MID_LPA_NOVICE_PANIC',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'SIC_MID_ROAD_ACCESS_CONSTRICTION',
        '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v11_features)

    # Add Distribution Grit features (V12.0)
    v12_features = [
        '[REDACTED_BY_SCRIPT]', 'SIC_GRIT_RANSOM_STRIP_RISK',
        'SIC_GRIT_COMMITTEE_CLIFF_EDGE', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'SIC_GRIT_BEST_FIELD_SACRIFICE', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v12_features)

    # Add Grid Edge & Inertia features (V13.0)
    v13_features = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v13_features)

    # Add Hazards & Margins features (V14.0)
    v14_features = [
        'SIC_HAZARD_RAIL_GLARE_RISK', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    consolidated_features.extend(v14_features)

    all_cohorts = {
        "[REDACTED_BY_SCRIPT]": _sanitize([REDACTED_BY_SCRIPT]),
        "[REDACTED_BY_SCRIPT]": _sanitize([REDACTED_BY_SCRIPT]),
        "COHORT_LPA_ALL": _sanitize(COHORT_LPA_ALL + lpa_additions),
        "COHORT_GRID_ALL": _sanitize(COHORT_GRID_ALL + grid_additions),
        "COHORT_ENV_ALL": _sanitize(COHORT_ENV_ALL + env_additions),
        "COHORT_SOCIO_ECONOMIC_ALL": _sanitize(COHORT_SOCIO_ECONOMIC_ALL + socio_additions + [f for f in X.columns if f.startswith('lpa_lsoa_agg_')]),
        "[REDACTED_BY_SCRIPT]": _sanitize([REDACTED_BY_SCRIPT]),
        "[REDACTED_BY_SCRIPT]": list(dict.fromkeys(consolidated_features))
    }

    print("[REDACTED_BY_SCRIPT]")
    final_cohorts = {}
    for cohort_name, features in all_cohorts.items():
        cohort_features_in_df = [f for f in features if f in X.columns]
        if not cohort_features_in_df:
            print(f"[REDACTED_BY_SCRIPT]'{cohort_name}'[REDACTED_BY_SCRIPT]")
            continue
        deduplicated_features = list(dict.fromkeys(cohort_features_in_df))
        final_cohorts[cohort_name] = deduplicated_features
        print(f"[REDACTED_BY_SCRIPT]")
        
    return final_cohorts

def phase_D_forge_project_regimes(X_train, X_val, X_test, n_clusters=5):
    """
    AD-AM-37: Forging Project Regimes.
    REMEDIATION: Removed '[REDACTED_BY_SCRIPT]'.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 1. Curate the feature set for defining strategic typologies.
    regime_features = [
        '[REDACTED_BY_SCRIPT]', 'lpa_major_commercial_approval_rate',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
        # '[REDACTED_BY_SCRIPT]' <-- REMOVED
    ]
    regime_features = [f for f in regime_features if f in X_train.columns]
    print(f"[REDACTED_BY_SCRIPT]")

    X_train_regime = X_train[regime_features]
    X_val_regime = X_val[regime_features]
    X_test_regime = X_test[regime_features]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_regime)
    X_val_scaled = scaler.transform(X_val_regime)
    X_test_scaled = scaler.transform(X_test_regime)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
    kmeans.fit(X_train_scaled)
    print("[REDACTED_BY_SCRIPT]")
    
    X_train_out = X_train.copy()
    X_val_out = X_val.copy()
    X_test_out = X_test.copy()
    X_train_out['project_regime_id'] = kmeans.predict(X_train_scaled)
    X_val_out['project_regime_id'] = kmeans.predict(X_val_scaled)
    X_test_out['project_regime_id'] = kmeans.predict(X_test_scaled)
    
    print("[REDACTED_BY_SCRIPT]")
    return X_train_out, X_val_out, X_test_out, scaler, kmeans

def phase_1_load_and_sanitize_data():
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    X_raw = pd.read_csv(X_PATH, index_col=0)
    y_reg = pd.read_csv(Y_REG_PATH, index_col=0).squeeze("columns")

    # Mandated Purge of non-predictive metadata (AD-AMARYLLIS-MOD-05)
    metadata_to_purge = [c for c in X_raw.columns if 'join_method' in c or 'application_reference' in c or 'application_id' in c]
    X_purged = X_raw.drop(columns=metadata_to_purge)
    print(f"[REDACTED_BY_SCRIPT]")

    # --- AD-AM-42: Removal of 100% Missing Features ---
    # These features have been identified as having 100% missing values and must be removed.
    missing_features_to_remove = [
        'solar_site_area_sqm', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]_stddev', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'site_lsoa_ruc_rural_score', '[REDACTED_BY_SCRIPT]', 'application_reference',
        'aw_nearest_area_ha', 'aw_is_within', 'aonb_nearest_area_sqkm', 'aonb_is_within', 'aw_AWP_count_in_2km',
        'aw_AWP_count_in_5km', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'sssi_unit_worst_condition_in_2km',
        '[REDACTED_BY_SCRIPT]', 'np_nearest_area_sqkm', 'np_is_within', '[REDACTED_BY_SCRIPT]',
        'np_count_in_2km', 'lpa_np_coverage_pct', 'lpa_sac_coverage_pct', 'spa_nearest_area_ha', 'lpa_spa_coverage_pct',
        'nt_intersects_site_bool', 'application_id', 'cs_on_site_area_ha', 'cs_on_site_pct_area', 'es_hls_on_site_pct_area',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'nt_nearest_name_Offa\'s Dyke Path', 'nt_nearest_name_Peddars Way and Norfolk Coast Path', 'nt_nearest_name_Pennine Bridleway',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'nt_nearest_name_Thames Path',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'es_on_site_highest_tier_HLS', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'spa_nearest_name_Broadland', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'spa_nearest_name_Greater Wash', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'spa_nearest_name_Lee Valley',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'spa_nearest_name_Nene Washes',
        'spa_nearest_name_Ouse Washes', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'spa_nearest_name_The Wash', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    
    # Filter to only those present in the dataframe to avoid errors
    features_to_drop = [f for f in missing_features_to_remove if f in X_purged.columns]
    X_purged = X_purged.drop(columns=features_to_drop)
    print(f"[REDACTED_BY_SCRIPT]")

    # Mandated Sanitization Gate
    X_sanitized = sanitize_column_names(X_purged)

    # --- FORENSIC IDENTITY RECOVERY (AD-AM-19 Support) ---
    # The 'Advanced LPA Queue' engine requires the 'lpa_name' grouping key.
    # If OHE has occurred, we must reconstruct the categorical identity from the fragments.
    if 'lpa_name' not in X_sanitized.columns:
        print("  - WARNING: 'lpa_name'[REDACTED_BY_SCRIPT]")
        # Identify OHE columns (sanitized, so likely '[REDACTED_BY_SCRIPT]')
        lpa_ohe_cols = [c for c in X_sanitized.columns if c.startswith('lpa_name_')]
        
        if lpa_ohe_cols:
            # Reconstruct by finding the column with value 1
            # We use idxmax to find the column name, then strip the prefix
            X_sanitized['lpa_name'] = X_sanitized[lpa_ohe_cols].idxmax(axis=1).apply(lambda x: str(x).replace('lpa_name_', ''))
            print(f"[REDACTED_BY_SCRIPT]'lpa_name'[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]'lpa_name'[REDACTED_BY_SCRIPT]")
            # Fallback: Create a dummy column to prevent immediate crash, though metrics will be meaningless globally
            X_sanitized['lpa_name'] = 'Unknown_LPA'

    # Final data alignment for regression task.
    #  CRITICAL: Only train on successful applications with a valid, positive planning duration.
    # This prevents data poisoning from failed (0, -1), invalid (NaN), or rejected (10000.0) cases.
    valid_indices = y_reg[(y_reg > 25) & (y_reg < 900.0)].index
    y_reg_clean = y_reg.loc[valid_indices]
    X_final = X_sanitized.loc[valid_indices]

    # --- OUTLIER REMOVAL: Capacity Cap ---
    print("[REDACTED_BY_SCRIPT]")
    mask_cap = X_final['[REDACTED_BY_SCRIPT]'] <= 75
    removed_count = (~mask_cap).sum()
    X_final = X_final[mask_cap]
    y_reg_clean = y_reg_clean[mask_cap]
    print(f"[REDACTED_BY_SCRIPT]")

    print(f"[REDACTED_BY_SCRIPT]")
    return X_final, y_reg_clean


def phase_1b_engineer_knn_anomaly_features(X_train, y_train, X_val, X_test, gcv_features, k_neighbors=15):
    """
    AD-AM-33 (Remediated for Mandate 41.4): Implements the k-NN Anomaly Detection Sub-System.
    This stateful process is executed post-split, fitting on train and transforming all sets.
    """
    print("[REDACTED_BY_SCRIPT]")

    # --- 2.1: Isolate the "Success Atlas" ---
    X_train_successful = X_train.copy()
    y_train_successful = y_train.copy()
    X_train_atlas_gcv = X_train_successful[gcv_features]
    print(f"[REDACTED_BY_SCRIPT]")

    # --- 2.2 & Risk Mitigation: Instantiate, Fit, and Persist Imputer, Scaler & k-NN Engine ---
    # SURGICAL INTERVENTION: An imputer is now mandatory to prevent NaN contamination.
    imputer = SimpleImputer(strategy='median')
    X_train_atlas_gcv_imputed = imputer.fit_transform(X_train_atlas_gcv)
    
    # MANDATORY SCALING to prevent silent failure of distance metrics.
    scaler = StandardScaler()
    X_train_atlas_scaled = scaler.fit_transform(X_train_atlas_gcv_imputed)
    
    # Fit the core k-NN engine ONLY on the sanitized and scaled atlas.
    knn_engine = NearestNeighbors(n_neighbors=k_neighbors, leaf_size=10, algorithm='ball_tree', n_jobs=-1, metric='euclidean')
    knn_engine.fit(X_train_atlas_scaled)
    print(f"[REDACTED_BY_SCRIPT]")

    # --- 2.3 & 2.4: The Feature Forging Mandate ---
    def _forge_features(X_target: pd.DataFrame, is_training_set: bool = False) -> pd.DataFrame:
        """[REDACTED_BY_SCRIPT]"""
        # Ensure target has the GCV columns, then impute and scale using the FITTED transformers.
        X_target_gcv_imputed = imputer.transform(X_target[gcv_features])
        X_target_gcv_scaled = scaler.transform(X_target_gcv_imputed)
        
        # Find the k nearest neighbors in the Success Atlas for each target sample.
        # LEAKAGE FIX: If training set, query k+1 neighbors and drop the first one (self).
        n_query = k_neighbors + 1 if is_training_set else k_neighbors
        distances, indices = knn_engine.kneighbors(X_target_gcv_scaled, n_neighbors=n_query)
        
        if is_training_set:
            # Drop the first column (0th neighbor is self with dist=0)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        
        # Map the returned indices back to the original atlas data to get neighbor properties.
        neighbor_durations = y_train_successful.iloc[indices.flatten()].values.reshape(indices.shape)
        
        # --- LPA Entropy Calculation ---
        lpa_proxy_identifier = 'lpa_major_commercial_approval_rate'
        if lpa_proxy_identifier not in X_train_successful.columns:
            print(f"[REDACTED_BY_SCRIPT]'{lpa_proxy_identifier}' not found. Skipping 'knn_lpa_entropy_gcv' feature.")
            X_target['knn_lpa_entropy_gcv'] = 0
        else:
            neighbor_lpas = X_train_successful[lpa_proxy_identifier].iloc[indices.flatten()].values.reshape(indices.shape)
            X_target['knn_lpa_entropy_gcv'] = pd.DataFrame(neighbor_lpas).nunique(axis=1).values

        X_target['[REDACTED_BY_SCRIPT]'] = distances.mean(axis=1)
        X_target['[REDACTED_BY_SCRIPT]'] = pd.DataFrame(neighbor_durations).var(axis=1).values
        X_target['[REDACTED_BY_SCRIPT]'] = pd.DataFrame(neighbor_durations).mean(axis=1).values

        # Sanitize any potential NaNs created by variance calculation on single-value groups.
        X_target.fillna(0, inplace=True)
        
        return X_target

    print("[REDACTED_BY_SCRIPT]")
    X_train_augmented = _forge_features(X_train.copy(), is_training_set=True)
    print("[REDACTED_BY_SCRIPT]")
    X_val_augmented = _forge_features(X_val.copy(), is_training_set=False)
    print("[REDACTED_BY_SCRIPT]")
    X_test_augmented = _forge_features(X_test.copy(), is_training_set=False)
    
    print("[REDACTED_BY_SCRIPT]")
    return X_train_augmented, X_val_augmented, X_test_augmented, scaler, knn_engine, imputer


# ==============================================================================
# PHASE 1: ARCHITECTING THE UNIMPEACHABLE METRIC PIPELINE (AD-AM-17)
# ==============================================================================

def asymmetric_overprediction_objective(y_true, y_pred):
    """
    AD-AM-34: A custom LightGBM objective function that penalizes OVERprediction
    more severely. This is the surgical remediation for the v3.9 model's
    systematic overprediction bias.
    """
    residual = (y_true - y_pred).astype("float")
    # Define the penalty multiplier for overprediction (real-world error < 0)
    overprediction_penalty = 2.0

    grad = np.where(residual < 0, -2.0 * residual * overprediction_penalty, -2.0 * residual)
    hess = np.where(residual < 0, 2.0 * overprediction_penalty, 2.0)

    return grad, hess


def phase_B_train_catastrophe_classifier(X_train, y_train, oracle_preds_train):
    """
    AD-AM-34: Decommissions the failed Anti-Oracle regressor and replaces it with
    a robust Catastrophe Risk Classifier.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # 1. Define the target: What constitutes a "catastrophe"?
    # A data-driven approach: an error in the top 10% of absolute errors.
    error = (y_train - oracle_preds_train).abs()
    catastrophe_threshold = error.quantile(0.90)
    y_catastrophe = (error > catastrophe_threshold).astype(int)
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 2. Define the specialist feature set: Arm the classifier with "perfect storm" signals.
    classifier_features = [
        # The original k-NN anomaly signals
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'knn_lpa_entropy_gcv',
        # High-order conflict SICs
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    classifier_features = [f for f in classifier_features if f in X_train.columns]
    X_classifier_train = X_train[classifier_features]
    print(f"[REDACTED_BY_SCRIPT]")

    # 3. Train the classifier, handling the extreme class imbalance.
    neg, pos = np.bincount(y_catastrophe)
    scale_pos_weight = neg / pos if pos > 0 else 1

    classifier = lgb.LGBMClassifier(
        objective='binary',
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_samples=10,
        min_data_in_leaf=10,
        min_split_gain=0.1,
        num_leaves=31,
        n_jobs=-1,
        reg_alpha=5,
        reg_lambda=5,
        verbosity=-1
    )
    classifier.fit(X_classifier_train, y_catastrophe)
    print("[REDACTED_BY_SCRIPT]")
    
    return classifier, classifier_features

def phase_C_train_quantile_specialists(X_train, oracle_error_train):
    """
    AD-AM-35: Forges the Quantile Specialist regressors to predict the
    bounds of the Oracle's error distribution.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # Define common parameters for the quantile models. They need to be flexible.
    quantile_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'random_state': RANDOM_STATE,
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 5,
        'num_leaves': 31,
        'n_jobs': -1,
        'reg_alpha': 5,
        'reg_lambda': 5,
        'verbosity': -1,
        'min_split_gain': 0.01
    }
    
    # --- Train the 10th Percentile (Plausible Best-Case Error) Specialist ---
    print("[REDACTED_BY_SCRIPT]")
    p10_model = lgb.LGBMRegressor(**quantile_params, alpha=0.10)
    p10_model.fit(X_train, oracle_error_train)
    
    # --- Train the 90th Percentile (Plausible Worst-Case Error) Specialist ---
    print("[REDACTED_BY_SCRIPT]")
    p90_model = lgb.LGBMRegressor(**quantile_params, alpha=0.90)
    p90_model.fit(X_train, oracle_error_train)
    
    print("[REDACTED_BY_SCRIPT]")
    return p10_model, p90_model


from sklearn.cluster import KMeans

def phase_D_forge_project_regimes(X_train, X_val, X_test, n_clusters=5):
    """
    AD-AM-37 (Remediated for Mandate 41.4): Implements "[REDACTED_BY_SCRIPT]"
    by fitting on the training set and applying the learned regimes to all data partitions.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 1. Curate the feature set for defining strategic typologies.
    regime_features = [
        '[REDACTED_BY_SCRIPT]', 'lpa_major_commercial_approval_rate',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    regime_features = [f for f in regime_features if f in X_train.columns]
    print(f"[REDACTED_BY_SCRIPT]")

    X_train_regime = X_train[regime_features]
    X_val_regime = X_val[regime_features]
    X_test_regime = X_test[regime_features]
    
    # 2. MANDATORY SCALING: k-Means is distance-based and requires scaled features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_regime)
    X_val_scaled = scaler.transform(X_val_regime)
    X_test_scaled = scaler.transform(X_test_regime)
    
    # 3. Fit the clustering algorithm on the training data ONLY to prevent leakage.
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
    kmeans.fit(X_train_scaled)
    print("[REDACTED_BY_SCRIPT]")
    
    # 4. Append the new regime ID feature to the main dataframes.
    X_train_out = X_train.copy()
    X_val_out = X_val.copy()
    X_test_out = X_test.copy()
    X_train_out['project_regime_id'] = kmeans.predict(X_train_scaled)
    X_val_out['project_regime_id'] = kmeans.predict(X_val_scaled)
    X_test_out['project_regime_id'] = kmeans.predict(X_test_scaled)
    
    print("[REDACTED_BY_SCRIPT]")
    return X_train_out, X_val_out, X_test_out, scaler, kmeans

def phase_E_train_failure_regime_specialist(X_train, oracle_error_train, catastrophe_classifier, catastrophe_features):
    """
    AD-AM-38: Forges a specialist regressor trained only on the samples
    identified as belonging to the "failure regime" (i.e., catastrophe risks).
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # 1. Isolate the failure regime within the training set.
    X_classifier_predict = X_train[catastrophe_features]
    is_catastrophe_train = (catastrophe_classifier.predict(X_classifier_predict) == 1)
    
    X_failure_regime = X_train[is_catastrophe_train]
    y_failure_regime = oracle_error_train[is_catastrophe_train]
    
    if X_failure_regime.empty:
        print("[REDACTED_BY_SCRIPT]")
        return None

    print(f"[REDACTED_BY_SCRIPT]")
    
    # 2. Train a small, regularized expert on this specific data subset.
    #    This model's only job is to predict error for high-risk cases.
    failure_specialist = lgb.LGBMRegressor(
        objective='regression_l1',
        random_state=RANDOM_STATE,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_samples=10,
        min_data_in_leaf=10,
        min_split_gain=0.1,
        num_leaves=31,
        n_jobs=-1,
        reg_alpha=5,
        reg_lambda=5,
        verbosity=-1
    )
    failure_specialist.fit(X_failure_regime, y_failure_regime)
    print("[REDACTED_BY_SCRIPT]")
    
    return failure_specialist


def create_encapsulated_model(regressor_params):
    """
    MANDATE 17.1: Factory to build the architecturally-mandated model object.
    This prevents the Log-Transform Catastrophe by design.
    """
    regressor = lgb.LGBMRegressor(**regressor_params)
    
    # The transformer now lives with the model, ensuring .predict() is always correct.
    encapsulated_model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )
    return encapsulated_model

def generate_performance_report(model, X_test, y_test, title_prefix=""):
    """
    MANDATE 17.2: The single, canonical function for generating a real-world performance report.
    """
    # .predict() on the encapsulated model automatically applies the inverse transform.
    y_pred = model.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    report_str = (
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
    )
    print(report_str)
    
    # The mandatory Error Distribution "Apology Plot"
    error = y_pred - y_test
    plt.figure(figsize=(12, 7))
    sns.histplot(error, kde=True, bins=50)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='[REDACTED_BY_SCRIPT]')
    plt.title(f"[REDACTED_BY_SCRIPT]", fontsize=16)
    plt.xlabel("[REDACTED_BY_SCRIPT]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    plot_path = os.path.join(OUTPUT_DIR_V35, f"[REDACTED_BY_SCRIPT]' ', '_')}_error_dist.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    return {"rmse": rmse, "mae": mae, "r2": r2}, report_str, plot_path


# ==============================================================================
# PHASE 2: RE-EXECUTING FINE-TUNING WITH A VALID OBJECTIVE (AD-AM-17)
# ==============================================================================

def phase_A_tune_individual_heads(X_train, y_head_target_train, sample_weights, cohorts, groups, n_trials=TRIALS_GLOBAL):
    """
    AD-AM-41.2: Individually tunes specialist head REGRESSORS to predict Oracle
    error regimes. The search space is constrained to enforce parsimony and prevent
    overfitting on high-frequency noise. Constraints scale with cohort size.
    """
    print("[REDACTED_BY_SCRIPT]'Anti-Oracle'[REDACTED_BY_SCRIPT]")
    all_best_params = {}

    for cohort_name, features in cohorts.items():
        print(f"[REDACTED_BY_SCRIPT]")
        
        specialist_features = [f for f in features if f in X_train.columns]
        if not specialist_features:
            print(f"[REDACTED_BY_SCRIPT]")
            continue

        X_head_tune = X_train[specialist_features]
    
        def objective(trial):
            n_features = X_head_tune.shape[1]
            
            # Param Grid: Standardized Constraints
            if n_features < 20: 
                max_depth = trial.suggest_int('max_depth', 2, 4)
                num_leaves = trial.suggest_int('num_leaves', 4, 12)
                min_child_samples = trial.suggest_int('min_child_samples', 30, 60)
                n_estimators = trial.suggest_int('n_estimators', 100, 300) # Higher max, let early stopping limit it
            elif n_features < 80: 
                max_depth = trial.suggest_int('max_depth', 3, 5)
                num_leaves = trial.suggest_int('num_leaves', 8, 24)
                min_child_samples = trial.suggest_int('min_child_samples', 50, 100)
                n_estimators = trial.suggest_int('n_estimators', 200, 500)
            else: 
                max_depth = trial.suggest_int('max_depth', 3, 6)
                num_leaves = trial.suggest_int('num_leaves', 16, 48)
                min_child_samples = trial.suggest_int('min_child_samples', 75, 150)
                n_estimators = trial.suggest_int('n_estimators', 300, 800)

            params = {
                'objective': 'regression_l1', 'metric': 'mae', 'random_state': RANDOM_STATE, 'verbosity': -1, 'n_jobs': -1,
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
                'min_child_samples': min_child_samples,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                
                # High Regularization Preserved
                'reg_alpha': trial.suggest_float('reg_alpha', 5.0, 50.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 50.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 0.1, log=True)
            }
            
            # Construct Dataset with weights ONCE. lgb.cv handles splitting internally.
            dtrain = lgb.Dataset(X_head_tune, label=y_head_target_train, weight=sample_weights)
            
            # Use GroupKFold to prevent leakage
            n_groups = groups.nunique()
            if n_groups <= 1:
                # Fallback to KFold if groups are insufficient
                folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            else:
                n_splits_head = min(5, n_groups)
                gkf = GroupKFold(n_splits=n_splits_head)
                folds = gkf.split(X_head_tune, y_head_target_train, groups=groups)

            # Execute Native CV with Early Stopping
            cv_results = lgb.cv(
                params,
                dtrain,
                num_boost_round=n_estimators,
                folds=folds, # Use GroupKFold
                stratified=False,
                shuffle=False, # GroupKFold handles shuffling if needed (but it doesn't shuffle groups by default, usually fine)
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )
            
            # Return best MAE from history
            if 'valid l1-mean' in cv_results:
                return cv_results['valid l1-mean'][-1]
            elif 'valid mae-mean' in cv_results:
                return cv_results['valid mae-mean'][-1]
            else:
                # Fallback for safety, though l1-mean is standard for regression_l1
                keys = list(cv_results.keys())
                print(f"[REDACTED_BY_SCRIPT]")
                return cv_results[keys[0]][-1]

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"[REDACTED_BY_SCRIPT]")
        all_best_params[cohort_name] = study.best_params
        
    print("[REDACTED_BY_SCRIPT]")
    return all_best_params

def phase_E_train_persist_and_analyze_heads(X_train, y_head_target_train, sample_weights, cohorts, all_head_params, output_dir, top_n=10):
    """
    AD-AM-41 (Revised): Forges, persists, and analyzes specialist REGRESSORS trained to
    predict Oracle residual errors directly.
    """
    os.makedirs(output_dir, exist_ok=True)
    top_head_features_path = os.path.join(output_dir, "[REDACTED_BY_SCRIPT]")
    top_head_features = {}

    print(f"[REDACTED_BY_SCRIPT]'Anti-Oracle' Regressor Heads ---")
    for cohort_name, features in cohorts.items():
        if cohort_name not in all_head_params:
            print(f"  Skipping '{cohort_name}'[REDACTED_BY_SCRIPT]")
            continue
            
        model_features = [f for f in features if f in X_train.columns]
        X_head_train = X_train[model_features]
        head_specific_params = all_head_params[cohort_name]
        
        print(f"[REDACTED_BY_SCRIPT]")
        
        # SURGICAL INTERVENTION: Enforce Robust Regression Objective
        # Must match the tuning phase (regression / RMSE)
        head_specific_params['objective'] = 'regression_l1'
        head_specific_params['metric'] = 'mae'
        if 'num_class' in head_specific_params: del head_specific_params['num_class'] # Cleanup
        
        model = lgb.LGBMRegressor(**head_specific_params, random_state=RANDOM_STATE, verbosity=-1)
        
        # Train the specialist regressor on the continuous residual target.
        model.fit(X_head_train, y_head_target_train, sample_weight=sample_weights)
        joblib.dump(model, os.path.join(output_dir, f"[REDACTED_BY_SCRIPT]"))

        print(f"[REDACTED_BY_SCRIPT]")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_head_train)
        
        # SURGICAL INTERVENTION: Simplified SHAP processing for Regression.
        # Regression SHAP is always a 2D array (samples, features).
        if isinstance(shap_values, list):
            # Fallback in case of unexpected library behavior
            mean_abs_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))
        else:
            # Standard Regression Case
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': X_head_train.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        top_features = importance_df['feature'].head(top_n).tolist()
        top_head_features[cohort_name] = top_features

    import json
    with open(top_head_features_path, 'w') as f:
        json.dump(top_head_features, f, indent=4)
    print(f"[REDACTED_BY_SCRIPT]")
    return top_head_features


def phase_J_train_stratified_capacity_residuals(X_train, y_residual_train, groups, n_trials=20):
    """
    AD-AM-46: Robust Stratified Residuals (Protocol v5.0).
    Reinstates stratification with '[REDACTED_BY_SCRIPT]' to fix the
    over-prediction on small projects.
    
    UPDATED: Now uses Optuna to optimize hyperparameters per stratum.
    """
    print("[REDACTED_BY_SCRIPT]")
    from sklearn.model_selection import train_test_split, GroupKFold
    from sklearn.metrics import root_mean_squared_error
    
    # ROBUST BINS: Defined to isolate distinct regulatory/physics regimes.
    # Updated to granular strata: 0-1, 1-5, 5-10, 10-15, 15-20, 20-30, 30-40, 40-50, 50-100, 100+
    bins = [0, 1, 5, 10, 15, 20, 30, 40, 50, 100, float('inf')]
    labels = [
        'Micro_0_1', 'Small_1_5', 
        'Mid_5_10', 'Mid_10_15', 'Mid_15_20', 
        'Large_20_30', 'Large_30_40', 'Large_40_50', 
        'Major_50_100', 'Major_100_Plus'
    ]
    
    # Ensure capacity exists (Guaranteed by Phase H change)
    if '[REDACTED_BY_SCRIPT]' not in X_train.columns:
        raise ValueError("[REDACTED_BY_SCRIPT]")

    X_train_strata = X_train.copy()
    X_train_strata['stratum_id'] = pd.cut(
        X_train_strata['[REDACTED_BY_SCRIPT]'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    stratified_models = {}
    
    for stratum in labels:
        mask = X_train_strata['stratum_id'] == stratum
        X_subset = X_train[mask]
        y_subset = y_residual_train[mask]
        groups_subset = groups[mask]
        
        n_samples = len(X_subset)
        print(f"[REDACTED_BY_SCRIPT]")
        
        if n_samples < 10:
            print(f"[REDACTED_BY_SCRIPT]")
            stratified_models[stratum] = None
            continue

        # --- OUTLIER REMOVAL (User Request) ---
        # Remove top and bottom 5% of residuals to stabilize training
        lower_bound = y_subset.quantile(0.02)
        upper_bound = y_subset.quantile(0.98)
        mask_clean = (y_subset >= lower_bound) & (y_subset <= upper_bound)
        
        X_subset = X_subset[mask_clean]
        y_subset = y_subset[mask_clean]
        groups_subset = groups_subset[mask_clean]
        
        
        n_samples_clean = len(X_subset)
        print(f"[REDACTED_BY_SCRIPT]")
        n_samples = n_samples_clean

        if n_samples < 10:
            print(f"[REDACTED_BY_SCRIPT]")
            stratified_models[stratum] = None
            continue
                    
        # Create local validation split for Optuna
        # Note: Ideally this should be group-aware too, but for simplicity we keep random split here
        # as the main leakage concern is in the feature selection loop below.
        X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=67
        )

        # --- FEATURE DIET PROTOCOL (Phase J) ---
        max_features = int(n_samples / 3)
        max_features = max(5, min(max_features, 100)) # Slightly tighter cap for generic Solar (80)
        
        print(f"[REDACTED_BY_SCRIPT]")
        
        # --- ROBUST FEATURE SELECTION (Group K-Fold) ---
        n_groups_subset = groups_subset.nunique()
        feature_importances = pd.DataFrame(index=X_subset.columns, data={'gain': 0.0})

        if n_groups_subset <= 1:
             # Fallback to KFold if only 1 group exists
             cv_select = KFold(n_splits=5, shuffle=True, random_state=67)
             split_gen = cv_select.split(X_subset, y_subset)
        else:
             n_splits_dynamic = min(SPLITS_GLOBAL, n_groups_subset)
             cv_select = GroupKFold(n_splits=n_splits_dynamic)
             split_gen = cv_select.split(X_subset, y_subset, groups=groups_subset)
        
        for train_idx, _ in split_gen:
            X_fold = X_subset.iloc[train_idx]
            y_fold = y_subset.iloc[train_idx]
            
            # Hardened Probe: Prevents selecting features that only work for 1-2 samples.
            probe = lgb.LGBMRegressor(
                random_state=67, n_jobs=-1, verbosity=-1, 
                max_depth=2, n_estimators=500, 
                min_child_samples=max(5, int(len(X_fold) * 0.1)), # Probe must respect data density
                reg_alpha=10.0 # High regularization for selection
            )
            probe.fit(X_fold, y_fold)
            feature_importances['gain'] += probe.feature_importances_
            
        top_stratum_features = feature_importances.sort_values('gain', ascending=False).head(max_features).index.tolist()
        X_subset_diet = X_subset[top_stratum_features]

        # --- ROBUST TUNING (K-Fold Objective) ---
        def objective(trial):
            # Dynamic Leaf Floor: 20% of data or 10, whichever is higher.
            # This prevents N=30 strata from having min_child=5 (too small).
            # For N=30, floor is 10. For N=100, floor is 20.
            dynamic_min_child = max(10, int(n_samples * 0.2))
            
            # Clamp upper bound to avoid invalid range if N is small
            upper_min_child = max(dynamic_min_child + 1, int(n_samples * 0.5))
            
            params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1,
                'n_estimators': trial.suggest_int('n_estimators', 50, 350),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                
                # STRUCTURAL LOCK: Depth 2 (Stumps) prevents complex interaction overfitting
                'max_depth': 2, 
                'num_leaves': trial.suggest_int('num_leaves', 2, 12), 
                
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                
                # REGULARIZATION WALL
                'reg_alpha': trial.suggest_float('reg_alpha', 10.0, 50.0, log=True), 
                'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 50.0, log=True),
                
                # THE STARVATION FLOOR
                'min_child_samples': trial.suggest_int('min_child_samples', dynamic_min_child, upper_min_child),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 1.0, log=True) # High split cost
            }
            
            dtrain = lgb.Dataset(X_subset_diet, label=y_subset)
            
            cv_results = lgb.cv(
                params, dtrain, num_boost_round=params['n_estimators'], nfold=5,
                stratified=False, shuffle=True, callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            if 'valid l1-mean' in cv_results:
                return cv_results['valid l1-mean'][-1]
            elif 'valid mae-mean' in cv_results:
                return cv_results['valid mae-mean'][-1]
            else:
                # Fallback for safety, though l1-mean is standard for regression_l1
                keys = list(cv_results.keys())
                print(f"[REDACTED_BY_SCRIPT]")
                return cv_results[keys[0]][-1]

        # Run Optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"[REDACTED_BY_SCRIPT]")

        # Retrain on FULL stratum data (DIET only)
        best_params = study.best_params
        best_params.update({
            'objective': 'regression_l1',
            'metric': 'mae',
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        })
        
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X_subset_diet, y_subset)
        final_model.feature_names_diet_ = top_stratum_features
        stratified_models[stratum] = final_model
        
    return stratified_models, bins, labels



def phase_K_train_gm_specialist(X_train_comp, y_residual, groups, sample_weights=None, n_trials=30):
    """
    AD-AM-47.1: Ground Mount Specialist (Protocol v5.0 - Stratified).
    Trains specialized regressors for Ground Mount Solar projects, stratified by capacity.
    
    UPDATED: Now uses Optuna to optimize hyperparameters per stratum.
    """
    print("[REDACTED_BY_SCRIPT]")
    from sklearn.model_selection import train_test_split, GroupKFold
    from sklearn.metrics import root_mean_squared_error
    
    # Reuse the granular bins from Phase J for consistency
    bins = [0, 1, 5, 10, 15, 20, 30, 40, 50, 100, float('inf')]
    labels = [
        'Micro_0_1', 'Small_1_5', 
        'Mid_5_10', 'Mid_10_15', 'Mid_15_20', 
        'Large_20_30', 'Large_30_40', 'Large_40_50', 
        'Major_50_100', 'Major_100_Plus'
    ]

    # Ensure capacity exists
    if '[REDACTED_BY_SCRIPT]' not in X_train_comp.columns:
        raise ValueError("[REDACTED_BY_SCRIPT]")

    X_train_strata = X_train_comp.copy()
    X_train_strata['stratum_id'] = pd.cut(
        X_train_strata['[REDACTED_BY_SCRIPT]'], 
        bins=bins, 
        labels=labels, 
        right=False
    )

    stratified_models = {}

    for stratum in labels:
        mask = X_train_strata['stratum_id'] == stratum
        X_subset = X_train_comp[mask]
        y_subset = y_residual[mask]
        groups_subset = groups[mask]
        
        if sample_weights is not None:
            w_subset = sample_weights[mask]
        else:
            w_subset = None
        
        n_samples = len(X_subset)
        print(f"[REDACTED_BY_SCRIPT]")
        
        if n_samples < 10:
            print(f"[REDACTED_BY_SCRIPT]")
            stratified_models[stratum] = None
            continue

        # --- OUTLIER REMOVAL (User Request) ---
        # Remove top and bottom 5% of residuals to stabilize training
        lower_bound = y_subset.quantile(0.02)
        upper_bound = y_subset.quantile(0.98)
        mask_clean = (y_subset >= lower_bound) & (y_subset <= upper_bound)
        
        X_subset = X_subset[mask_clean]
        y_subset = y_subset[mask_clean]
        groups_subset = groups_subset[mask_clean]
        if w_subset is not None:
            w_subset = w_subset[mask_clean]
        
        n_samples_clean = len(X_subset)
        print(f"[REDACTED_BY_SCRIPT]")
        n_samples = n_samples_clean

        if n_samples < 10:
            print(f"[REDACTED_BY_SCRIPT]")
            stratified_models[stratum] = None
            continue
        
        # Split for Optuna
        if w_subset is not None:
            X_opt_train, X_opt_val, y_opt_train, y_opt_val, w_opt_train, w_opt_val = train_test_split(
                X_subset, y_subset, w_subset, test_size=0.2, random_state=67
            )
        else:
            X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
                X_subset, y_subset, test_size=0.2, random_state=67
            )
            w_opt_train, w_opt_val = None, None

        # --- FEATURE DIET PROTOCOL (AD-AM-48) ---
        # Calculate Max Permissible Features (MPF) based on sample size N to prevent overfitting.
        # Rule of thumb: N > 10 * P (features). 
        # For N=100, we allow max 10 features. For N=1000, max 50 (cap).
        max_features = int(n_samples / 10)
        max_features = max(3, min(max_features, 70)) # Clamp between 5 and 50
        
        print(f"[REDACTED_BY_SCRIPT]")
        
        # --- ROBUST FEATURE SELECTION (Group K-Fold) ---
        # Instead of one probe, we average importance across 50 folds to find STABLE features.
        n_groups_subset = groups_subset.nunique()
        feature_importances = pd.DataFrame(index=X_subset.columns, data={'gain': 0.0})

        if n_groups_subset <= 1:
             # Fallback to KFold if only 1 group exists
             cv_select = KFold(n_splits=5, shuffle=True, random_state=67)
             split_gen = cv_select.split(X_subset, y_subset)
        else:
             n_splits_dynamic = min(SPLITS_GLOBAL, n_groups_subset)
             cv_select = GroupKFold(n_splits=n_splits_dynamic)
             split_gen = cv_select.split(X_subset, y_subset, groups=groups_subset)
        
        for train_idx, _ in split_gen:
            # Safe slicing
            X_fold = X_subset.iloc[train_idx]
            y_fold = y_subset.iloc[train_idx]
            w_fold = w_subset.iloc[train_idx] if w_subset is not None else None
            
            # Hardened Probe: Prevents selecting features that only work for 1-2 samples.
            probe = lgb.LGBMRegressor(
                random_state=67, n_jobs=-1, verbosity=-1, 
                max_depth=2, n_estimators=100, 
                min_child_samples=max(5, int(len(X_fold) * 0.1)), # Probe must respect data density
                reg_alpha=10.0 # High regularization for selection
            )
            probe.fit(X_fold, y_fold, sample_weight=w_fold)
            feature_importances['gain'] += probe.feature_importances_
            
        top_stratum_features = feature_importances.sort_values('gain', ascending=False).head(max_features).index.tolist()
        X_subset_diet = X_subset[top_stratum_features]

        # --- ROBUST TUNING (K-Fold Objective) ---
        # Optuna now minimizes the CV score, not a single split score.
        
        def objective_small(trial):
            # Dynamic Leaf Floor: 20% of data or 10, whichever is higher.
            # This prevents N=30 strata from having min_child=5 (too small).
            # For N=30, floor is 10. For N=100, floor is 20.
            dynamic_min_child = max(10, int(n_samples * 0.2))
            
            # Clamp upper bound to avoid invalid range if N is small
            upper_min_child = max(dynamic_min_child + 1, int(n_samples * 0.5))

            params = {
                'objective': 'regression', 'metric': 'rmse', 'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 5, 31),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 30.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 30.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', dynamic_min_child, upper_min_child),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 0.1, log=True)
            }
            
            # Using LGBM's internal CV for speed and correctness
            dtrain = lgb.Dataset(X_subset_diet, label=y_subset, weight=w_subset)
            
            # 5-Fold CV within the trial
            cv_results = lgb.cv(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                nfold=5,
                stratified=False,
                shuffle=True,
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            # Return the best CV RMSE from history
            return cv_results['valid rmse-mean'][-1]
        
        def objective_rest(trial):

            # Dynamic Leaf Floor: 20% of data or 10, whichever is higher.
            # This prevents N=30 strata from having min_child=5 (too small).
            # For N=30, floor is 10. For N=100, floor is 20.
            dynamic_min_child = max(10, int(n_samples * 0.2))
            
            # Clamp upper bound to avoid invalid range if N is small
            upper_min_child = max(dynamic_min_child + 1, int(n_samples * 0.5))
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1,
                'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                
                # STRUCTURAL LOCK: Depth 2 (Stumps) prevents complex interaction overfitting
                'max_depth': 2, 
                'num_leaves': trial.suggest_int('num_leaves', 2, 6), 
                
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                
                # REGULARIZATION WALL
                'reg_alpha': trial.suggest_float('reg_alpha', 5.0, 50.0, log=True), 
                'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 50.0, log=True),
                
                # THE STARVATION FLOOR
                'min_child_samples': trial.suggest_int('min_child_samples', dynamic_min_child, upper_min_child),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 1.0, log=True) # High split cost
            }
            dtrain = lgb.Dataset(X_subset_diet, label=y_subset, weight=w_subset)
            cv_results = lgb.cv(
                params, dtrain, num_boost_round=params['n_estimators'], nfold=5,
                stratified=False, shuffle=True, callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            return cv_results['valid rmse-mean'][-1]

        study = optuna.create_study(direction='minimize')
        if stratum == 'Micro_0_1' or stratum == 'Small_1_5' or stratum == 'Mid_5_10' or stratum == 'Mid_10_15':
            study.optimize(objective_small, n_trials=n_trials)
        else:
            study.optimize(objective_rest, n_trials=n_trials)
        
        print(f"[REDACTED_BY_SCRIPT]")
        
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        })
        
        final_model = lgb.LGBMRegressor(**best_params)
        # CRITICAL: Train on DIET features only
        final_model.fit(X_subset_diet, y_subset, sample_weight=w_subset)
        
        # Save the feature list inside the model object for inference usage
        final_model.feature_names_diet_ = top_stratum_features
        stratified_models[stratum] = final_model

    print("[REDACTED_BY_SCRIPT]")
    return stratified_models, bins, labels


def phase_L_train_gm_scale_corrector(X_train, y_residual):
    """
    AD-AM-47.2: Ground Mount Scale Corrector (Protocol v4.9).
    Replaces the failed 'Stratified' binning with a continuous residual model.
    Focuses purely on the interaction between Project Scale and LPA Workload.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # Isolate the specific features that drive scale-based errors
    scale_features = [
        '[REDACTED_BY_SCRIPT]', 
        'lpa_workload_trend', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'SIC_LPA_POLITICAL_VOLATILITY'
    ]
    
    # Hardening: Ensure features exist
    available_features = [f for f in scale_features if f in X_train.columns]
    
    # [CRITICAL FIX] Deduplicate columns in X_train before selection to prevent LightGBM error
    # "[REDACTED_BY_SCRIPT]"
    X_train_deduped = X_train.loc[:, ~X_train.columns.duplicated()]
    
    # Select features from the deduplicated dataframe
    X_scale = X_train_deduped[available_features].copy()
    
    # INFLECTION ENABLED: Depth 2 allows for a "Sweet Spot" correction.
    # It can learn that both Micro AND Major projects have delays, while Mid are fast.
    params = {
        'objective': 'regression_l1', # MAE for robustness
        'metric': 'mae',
        'random_state': 42,
        'verbosity': -1,
        'n_estimators': 100,          # Increased slightly
        'learning_rate': 0.05,
        'num_leaves': 4,              # Max leaves for Depth 2
        'max_depth': 2,               # One inflection point allowed
        'min_child_samples': 50,      
        'colsample_bytree': 0.8,
        'min_split_gain': 0.01,
        'reg_alpha': 10.0,            
        'n_jobs': -1,
        'min_split_gain': 0.01
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_scale, y_residual)
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    return model



def generate_stratified_predictions(X, stratified_models, bins, labels):
    """
    Routes each sample to its appropriate capacity model.
    Accepts dynamic bin definitions to match training.
    """
    # Identify strata for incoming data
    strata_series = pd.cut(
        X['[REDACTED_BY_SCRIPT]'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    predictions = pd.Series(0.0, index=X.index)
    
    for stratum, model in stratified_models.items():
        if model is None:
            continue
            
        mask = strata_series == stratum
        if mask.any():
            X_subset = X.loc[mask]
            
            # AD-AM-48: Handle Feature Diet Slicing
            if hasattr(model, 'feature_names_diet_'):
                # Model was trained on a strict subset
                needed_cols = model.feature_names_diet_
                # Hardening: Ensure all needed cols exist (fill 0 if missing)
                missing = [c for c in needed_cols if c not in X_subset.columns]
                if missing:
                    for c in missing: X_subset[c] = 0
                X_input = X_subset[needed_cols]
            else:
                # Fallback for legacy models or Solar Stratified (if not updated yet)
                X_input = X_subset
                
            subset_preds = model.predict(X_input)
            predictions.loc[mask] = subset_preds
            
    return predictions


def phase_F_generate_ridge_arbiter_features(X_base, oracle_preds, head_residual_preds, top_head_features, stratified_residual_preds=None, scaler=None):
    """
    Architectural Mandate: Forges the feature matrix for the final Arbiter model.
    UPDATED: Restores raw head predictions but maintains the 'Context Diet' to focus
    the Arbiter on weighing experts rather than learning new patterns.
    """
    print("[REDACTED_BY_SCRIPT]")
    arbiter_features = pd.DataFrame(index=X_base.index)
    arbiter_features['[REDACTED_BY_SCRIPT]'] = oracle_preds
    
    # AD-AM-44 Integration: Stratified Residuals
    if stratified_residual_preds is not None:
        arbiter_features['[REDACTED_BY_SCRIPT]'] = stratified_residual_preds
    else:
        arbiter_features['[REDACTED_BY_SCRIPT]'] = 0.0

    # 1. Add Aggregated Residual Signals (The "Conclave Consensus")
    all_head_preds = pd.DataFrame(head_residual_preds) # Columns are cohort names
    
    # Feature: Mean Residual Opinion (The consensus direction of error)
    arbiter_features['[REDACTED_BY_SCRIPT]'] = all_head_preds.mean(axis=1)
    
    # Feature: Residual Uncertainty (The disagreement between specialists)
    arbiter_features['std_specialist_residual'] = all_head_preds.std(axis=1)
    
    # Feature: Max Absolute Residual (The "Alarmist" signal)
    arbiter_features['[REDACTED_BY_SCRIPT]'] = all_head_preds.abs().max(axis=1)

    # 2. Add Raw Residual Predictions from each head (Restored)
    for cohort_name in head_residual_preds.keys():
        arbiter_features[f"[REDACTED_BY_SCRIPT]"] = head_residual_preds[cohort_name]

    # 3. CONTEXT DIET: Minimal Context Injection
    # We retain ONLY the minimal 'Selector' signals that help the Arbiter decide WHICH head to trust.
    context_keys = [
        '[REDACTED_BY_SCRIPT]',          # Scale Regime
        'lpa_major_commercial_approval_rate', # Political Regime
        'project_regime_id',                  # Cluster Regime
        'SIC_POLICY_REGIME_ID'                # Temporal Regime
    ]
    
    # Safely select only available context keys
    available_context = [c for c in context_keys if c in X_base.columns]
    arbiter_features = arbiter_features.join(X_base[available_context])
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 4. Final sanitation and scaling
    arbiter_features.fillna(0, inplace=True)
    
    if scaler is None: 
        scaler = StandardScaler()
        arbiter_features_scaled = pd.DataFrame(
            scaler.fit_transform(arbiter_features),
            index=arbiter_features.index,
            columns=arbiter_features.columns
        )
        return arbiter_features_scaled, scaler
    else: 
        arbiter_features_scaled = pd.DataFrame(
            scaler.transform(arbiter_features),
            index=arbiter_features.index,
            columns=arbiter_features.columns
        )
        return arbiter_features_scaled, scaler


def phase_G_tune_and_train_lgbm_arbiter(X_arbiter_train, y_arbiter_train, X_arbiter_val, y_arbiter_val, groups_train, groups_val, n_trials=TRIALS_GLOBAL):
    """
    Tunes and trains the final Arbiter model using LightGBM with Group K-Fold CV.
    Combines train and validation sets to ensure robust hyperparameter selection.
    Updated for v6.0 Overfitting Remediation.
    """
    print("[REDACTED_BY_SCRIPT]")
    from sklearn.model_selection import GroupKFold
    
    # Data Fusion for Cross-Validation
    X_full = pd.concat([X_arbiter_train, X_arbiter_val])
    y_full = pd.concat([y_arbiter_train, y_arbiter_val])
    groups_full = pd.concat([groups_train, groups_val])
    
    # Ensure groups align with X_full (reset index might be needed if indices are not unique, but here we assume they are aligned)
    # Actually, pd.concat preserves index. If indices are unique, we are good.
    
    def objective(trial):
        params = {
            'objective': 'regression_l1', # MAE optimization
            'metric': 'mae',
            'random_state': RANDOM_STATE,
            'verbosity': -1,
            'n_jobs': -1,
            
            # Restrict tree size severely to force parsimony
            'n_estimators': trial.suggest_int('n_estimators', 25, 75),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
            
            # THE LINEAR STRAIGHTJACKET: Depth 1 ensures model is just a weighted sum
            'max_depth': 1,
            'num_leaves': 10,
            
            # INCREASED LEAF FLOOR (Anti-Overfitting)
            # Must see meaningful chunks of data (10%+ of dataset) to split
            'min_child_samples': trial.suggest_int('min_child_samples', 100, 300),
            
            # FEATURE SUBSAMPLING
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'subsample_freq': 1,
            
            # INCREASED REGULARIZATION FLOOR
            # High L1/L2 penalties force the model to ignore weak heads
            'reg_alpha': trial.suggest_float('reg_alpha', 15.0, 150.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 15.0, 150.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 5.0, log=True)
        }
        
        # Use GroupKFold to prevent leakage
        n_groups_full = groups_full.nunique()
        if n_groups_full <= 1:
            # Fallback to KFold
            folds = KFold(n_splits=SPLITS_GLOBAL, shuffle=True, random_state=RANDOM_STATE)
        else:
            n_splits_arbiter = min(SPLITS_GLOBAL, n_groups_full)
            gkf = GroupKFold(n_splits=n_splits_arbiter)
            folds = gkf.split(X_full, y_full, groups=groups_full)
        
        dtrain = lgb.Dataset(X_full, label=y_full)
        
        cv_results = lgb.cv(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            folds=folds, # Pass the iterator directly for robust evaluation
            stratified=False,
            shuffle=False, 
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Return best CV MAWE (Mean of repeats)
        if 'valid l1-mean' in cv_results:
            return cv_results['valid l1-mean'][-1]
        elif 'valid mae-mean' in cv_results:
            return cv_results['valid mae-mean'][-1]
        else:
            # Fallback for safety, though l1-mean is standard for regression_l1
            keys = list(cv_results.keys())
            print(f"[REDACTED_BY_SCRIPT]")
            return cv_results[keys[0]][-1]
            

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[REDACTED_BY_SCRIPT]")
    best_params = study.best_params

    # Retrain on full data with best params
    print("[REDACTED_BY_SCRIPT]")
    
    best_params['objective'] = 'regression_l1'
    best_params['metric'] = 'mae'
    best_params['random_state'] = RANDOM_STATE
    best_params['verbosity'] = -1
    best_params['n_jobs'] = -1
    
    final_arbiter = lgb.LGBMRegressor(**best_params)
    final_arbiter.fit(X_full, y_full)
    print("[REDACTED_BY_SCRIPT]")
    
    return final_arbiter

def phase_H_retune_arbiter_with_valid_objective(X_train, y_train, groups, n_trials=TRIALS_GLOBAL):
    """
    [AD-AM-32 Remediation] Re-tunes the Arbiter directly on its error-prediction task,
    excising the incompatible TransformedTargetRegressor.
    """
    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'random_state': 42, 'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 800, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.005, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 3, 25),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8),
            'subsample': trial.suggest_float('subsample', 0.6, 0.8),
            'n_jobs': -1,
            'verbosity': -1,
            'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 0.1, log=True)
        }
        
        # The Arbiter is now a raw LGBMRegressor, as it predicts error directly.
        # The TransformedTargetRegressor is no longer architecturally valid for this component.
        model = lgb.LGBMRegressor(**params)
        
        try:
            X_train_np = X_train.values
            y_train_np = y_train.values

            score = cross_val_score(
                model, X_train_np, y_train_np, 
                cv=GroupKFold(n_splits=SPLITS_GLOBAL),
                groups=groups,
                scoring='[REDACTED_BY_SCRIPT]'
            ).mean()
            return -score
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            return float('inf')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    print(f"[REDACTED_BY_SCRIPT]")
    return study.best_params


# ==============================================================================
# MAIN REMEDIATION EXECUTION BLOCK
# ==============================================================================

def phase_G_generate_arbiter_features(X_new, head_models_dir, cohorts, oracle_prediction_series, 
                                      catastrophe_classifier, catastrophe_features, 
                                      p10_model, p90_model, top_specialists, failure_specialist):
    """
    AD-AM-38 Remediation: Corrects a procedural flaw to prevent UnboundLocalError
    and properly integrates all new intelligence signals.
    """
    # Block 1: Foundational Signals (Oracle Baseline + Catastrophe Risk)
    arbiter_features = pd.DataFrame(index=X_new.index)
    arbiter_features['[REDACTED_BY_SCRIPT]'] = oracle_prediction_series
    
    X_classifier_predict = X_new[catastrophe_features]
    arbiter_features['[REDACTED_BY_SCRIPT]'] = catastrophe_classifier.predict_proba(X_classifier_predict)[:, 1]

    # Block 2: Uncertainty Signals from Quantile Specialists
    p10_preds = p10_model.predict(X_new)
    p90_preds = p90_model.predict(X_new)
    arbiter_features['[REDACTED_BY_SCRIPT]'] = p90_preds - p10_preds
    arbiter_features['[REDACTED_BY_SCRIPT]'] = (p90_preds + p10_preds) / 2.0
    
    # Block 3: Top Specialist Opinions (Error Predictions)
    specialist_error_predictions = pd.DataFrame(index=X_new.index)
    for cohort_name in top_specialists:
        if cohort_name not in cohorts: continue
        features = cohorts[cohort_name]
        model_path = os.path.join(head_models_dir, f"[REDACTED_BY_SCRIPT]")
        if not os.path.exists(model_path): continue
        
        model = joblib.load(model_path)
        model_features = [f for f in features if f in X_new.columns]
        if not model_features: continue
        
        X_head_predict = X_new[model_features]
        error_pred_col_name = f"[REDACTED_BY_SCRIPT]"
        specialist_error_predictions[error_pred_col_name] = model.predict(X_head_predict)

    # Block 4: AD-AM-36 "[REDACTED_BY_SCRIPT]" Protocol
    abs_error_preds = specialist_error_predictions.abs()
    dominant_source_series = abs_error_preds.idxmax(axis=1)
    dominant_source_dummies = pd.get_dummies(dominant_source_series, prefix='dominant_error_source')
    
    # Block 5: AD-AM-37 "[REDACTED_BY_SCRIPT]" Signal
    project_regime_dummies = pd.get_dummies(X_new['project_regime_id'], prefix='project_regime')

    # Block 6: AD-AM-38 "[REDACTED_BY_SCRIPT]" Signal
    if failure_specialist:
        is_catastrophe_new = (catastrophe_classifier.predict(X_classifier_predict) == 1)
        failure_regime_preds = pd.Series(0.0, index=X_new.index)
        
        if np.any(is_catastrophe_new):
            X_new_failure_regime = X_new[is_catastrophe_new]
            preds = failure_specialist.predict(X_new_failure_regime)
            failure_regime_preds.loc[is_catastrophe_new] = preds
        
        # SURGICAL CORRECTION: Add the new signal to the primary carrier DataFrame.
        arbiter_features['failure_regime_error_pred'] = failure_regime_preds

    # Block 7: Final Assembly of the Arbiter's Decision Matrix
    arbiter_matrix = pd.concat([
        arbiter_features, 
        specialist_error_predictions, 
        dominant_source_dummies,
        project_regime_dummies
    ], axis=1)
    
    # Block 8: The hardening and sanitation gate remains MANDATORY.
    arbiter_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    if arbiter_matrix.isnull().sum().sum() > 0:
        print("[REDACTED_BY_SCRIPT]")
        imputer = SimpleImputer(strategy='median')
        original_dtypes = arbiter_matrix.dtypes
        imputed_data = imputer.fit_transform(arbiter_matrix)
        arbiter_matrix = pd.DataFrame(imputed_data, index=arbiter_matrix.index, columns=arbiter_matrix.columns).astype(original_dtypes)

    return arbiter_matrix

def get_calibrated_prediction(model, arbiter_matrix):
    """
    AD-AM-29: Applies a Positive Bias Calibration Layer to the final prediction,
    enforcing a "business reality" safety margin to counteract underprediction.
    """
    # 1. Get the raw prediction from the encapsulated model (in real-world days)
    raw_prediction_days = model.predict(arbiter_matrix)
    
    # 2. Apply a positive bias based on the magnitude of the prediction.
    #    This accounts for the observed heteroscedasticity.
    bias_percentage = 0.10 # Add a 10% safety margin per directive
    calibrated_prediction = raw_prediction_days * (1 + bias_percentage)
    
    return calibrated_prediction


def phase_D1_arbiter_shap_analysis(model, arbiter_test_matrix, output_dir):
    """
    MANDATE 18.1: Generates a SHAP summary for the final arbiter model to
    determine the influence of its input features (the head model predictions).
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # AD-AM-32 REMEDIATION: The final model is a raw LGBMRegressor, not an encapsulated
    # object. The .regressor_ attribute is no longer present or necessary.
    arbiter_regressor = model
    
    explainer = shap.TreeExplainer(arbiter_regressor)
    shap_values = explainer.shap_values(arbiter_test_matrix)
    
    # Generate and persist the SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, arbiter_test_matrix, show=False, plot_type='dot')
    plt.title("[REDACTED_BY_SCRIPT]", fontsize=16)
    plt.xlabel("[REDACTED_BY_SCRIPT]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plot_path = os.path.join(output_dir, "[REDACTED_BY_SCRIPT]")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"[REDACTED_BY_SCRIPT]")

    # Produce the table of the top 5 most influential heads
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Head': arbiter_test_matrix.columns,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    most_influential_input = importance_df['Head'].iloc[0]
    top_5_inputs = importance_df.head(5)

    report = [
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'expert opinions'[REDACTED_BY_SCRIPT]",
        f"[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        top_5_inputs.to_markdown(index=False),
        "\n### Interpretation\n",
        f"The Arbiter's decisions are most heavily influenced by **`{most_influential_input}`**. This analysis includes both cohort head predictions and raw GCV features passed directly to the arbiter. The high ranking of GCV features confirms the Global Signal Amplification protocol is working as designed. For the forensic deep-dive, we will select the most influential input that represents an actual sub-model.\n"
    ]
    
    return "".join(report), importance_df

def phase_D2_forensic_case_studies(influential_head, results_df, X_test, cohorts, head_models_dir, output_dir, gcv, oracle_preds_test):
    """
    MANDATE 18.2: Conducts forensic case studies by constructing the full augmented feature
    set required by the re-architected head models.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # Load the influential head model
    head_model_path = os.path.join(head_models_dir, f"[REDACTED_BY_SCRIPT]")
    head_model = joblib.load(head_model_path)
    
    # ARCHITECTURAL CORRECTION: Reconstruct the *exact* feature matrix the model was trained on.
    # The heads are trained ONLY on their cohort features in Phase E.
    # GCV and Oracle Prediction are NOT included in the training set for the heads.
    specialist_features = [f for f in cohorts[influential_head] if f in X_test.columns]
    X_test_head_augmented = X_test[specialist_features].copy()
    
    explainer = shap.TreeExplainer(head_model)

    def _generate_case_study(case_type, case_index):
        # The instance for analysis must be sliced from the full augmented matrix.
        instance = X_test_head_augmented.loc[[case_index]]
        shap_values_instance = explainer.shap_values(instance)

        for i in range(len(shap_values_instance)):
            plt.figure()

            # The waterfall plot must also use the full augmented feature set for its labels.
            shap.plots.waterfall(shap.Explanation(
                values=shap_values_instance[i],
                base_values=explainer.expected_value,
                data=instance.iloc[0],
                feature_names=X_test_head_augmented.columns
            ), max_display=15, show=False)
            plt.title(f"[REDACTED_BY_SCRIPT]", fontsize=14)
            plot_filename = f"[REDACTED_BY_SCRIPT]' ', '_')}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

        true_val = results_df.loc[case_index, 'y_true']
        pred_val = results_df.loc[case_index, 'y_pred']
        error_val = results_df.loc[case_index, 'error']

        narrative = [
            f"[REDACTED_BY_SCRIPT]",
            f"[REDACTED_BY_SCRIPT]",
            f"[REDACTED_BY_SCRIPT]",
            f"[REDACTED_BY_SCRIPT]",
            f"[REDACTED_BY_SCRIPT]",
            f"[REDACTED_BY_SCRIPT]'s prediction was driven by specific feature interactions. The waterfall plot above pinpoints the exact features that pushed the prediction away from the base value, providing a granular view of the model'[REDACTED_BY_SCRIPT]"
        ]
        return "".join(narrative)

    # Identify cases
    underestimation_idx = results_df['error'].idxmin()
    overestimation_idx = results_df['error'].idxmax()
    
    print(f"[REDACTED_BY_SCRIPT]")
    report_under = _generate_case_study("[REDACTED_BY_SCRIPT]", underestimation_idx)
    
    print(f"[REDACTED_BY_SCRIPT]")
    report_over = _generate_case_study("[REDACTED_BY_SCRIPT]", overestimation_idx)

    report = [
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'Intra-LPA Variance' Signal is Missing\n",
        f"[REDACTED_BY_SCRIPT]'s single most influential component, yet the model fails consistently on outliers within otherwise predictable LPAs.\n",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'LPA Controversy Score'[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'Grid Connection Queue' Signal is Missing\n",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'Landscape Mitigation Quality' Signal is Missing\n",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'LVIA Mitigation Strategy' (e.g., 'Bund Screening', 'Tree Planting', 'No Action'); (2) A numerical 'LVIA Confidence Score'[REDACTED_BY_SCRIPT]'s consultants.\n"
    ]
    
    return "".join(report)

def phase_D3_bias_investigation(results_df, output_dir):
    """
    MANDATE 18.3: Generates a Residuals vs. Predicted plot to investigate
    systematic bias and other error patterns.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    plt.figure(figsize=(12, 7))
    
    try:
        # Preferred method: Use LOWESS smoothing for a clear trend line.
        # This requires the 'statsmodels' package.
        sns.residplot(x='y_pred', y='residuals', data=results_df, lowess=True, 
                      scatter_kws={'alpha': 0.5}, 
                      line_kws={'color': 'red', 'lw': 2, 'label': 'Trend'})
        plt.legend()
    except RuntimeError as e:
        # Fallback for incomplete environments.
        if "statsmodels" in str(e):
            print("  WARNING: 'statsmodels'[REDACTED_BY_SCRIPT]")
            print("[REDACTED_BY_SCRIPT]")
            sns.residplot(x='y_pred', y='residuals', data=results_df, lowess=False,
                          scatter_kws={'alpha': 0.5})
        else:
            raise e # Re-raise other unexpected runtime errors.

    plt.title("[REDACTED_BY_SCRIPT]", fontsize=16)
    plt.xlabel("[REDACTED_BY_SCRIPT]", fontsize=12)
    plt.ylabel("[REDACTED_BY_SCRIPT]", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linestyle='--', lw=1)
    
    plot_path = os.path.join(output_dir, "[REDACTED_BY_SCRIPT]")
    plt.savefig(plot_path)
    plt.close()
    print(f"[REDACTED_BY_SCRIPT]")

    report = [
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'s error structure.\n",
        f"[REDACTED_BY_SCRIPT]",
        "### Analysis\n",
        "[REDACTED_BY_SCRIPT]'s systematic tendency to **underestimate** planning durations (`True > Predicted`). The LOWESS trend line, if available, would quantify this trend.\n",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'add 50 days to every prediction'[REDACTED_BY_SCRIPT]"
    ]

    return "".join(report)

def phase_D4_generate_hypothesis_report(influential_head):
    """
    MANDATE 18.4: Generates a qualitative report detailing concrete hypotheses
    for the next phase of feature engineering.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    report = [
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'Intra-LPA Variance' Signal is Missing\n",
        f"[REDACTED_BY_SCRIPT]'s single most influential component, yet the model fails consistently on outliers within otherwise predictable LPAs.\n",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'LPA Controversy Score'[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'Grid Connection Queue' Signal is Missing\n",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'Landscape Mitigation Quality' Signal is Missing\n",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]'LVIA Mitigation Strategy' (e.g., 'Bund Screening', 'Tree Planting', 'No Action'); (2) A numerical 'LVIA Confidence Score'[REDACTED_BY_SCRIPT]'s consultants.\n"
    ]
    
    return "".join(report)

def phase_D5_all_heads_shap_analysis(X_test, cohorts, head_models_dir, output_dir, gcv, oracle_preds_test, top_n=20):
    """
    MANDATE 18 ADDENDUM: Generates SHAP summary plots for every specialist head model,
    using the full augmented feature set required by the re-architected models.
    """
    print("[REDACTED_BY_SCRIPT]")
    shap_output_dir = os.path.join(output_dir, "head_shap_summaries")
    os.makedirs(shap_output_dir, exist_ok=True)
    
    report_content = ["[REDACTED_BY_SCRIPT]",
                      "[REDACTED_BY_SCRIPT]"]

    for cohort_name in sorted(cohorts.keys()):
        print(f"[REDACTED_BY_SCRIPT]")
        
        # 1. Load the head model
        model_path = os.path.join(head_models_dir, f"[REDACTED_BY_SCRIPT]")
        if not os.path.exists(model_path):
            print(f"[REDACTED_BY_SCRIPT]")
            continue
            
        try:
            head_model = joblib.load(model_path)
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            continue

        # 2. Reconstruct the EXACT feature matrix used for training
        # Logic must match phase_E_train_persist_and_analyze_heads.
        # The heads are trained ONLY on their cohort features.
        specialist_features = [f for f in cohorts[cohort_name] if f in X_test.columns]
        
        X_test_head_augmented = X_test[specialist_features].copy()
        
        # 3. Calculate SHAP values
        try:
            explainer = shap.TreeExplainer(head_model)
            # Subsample for speed if necessary, but usually X_test is small enough
            shap_values = explainer.shap_values(X_test_head_augmented)
            
            # Handle Classifier Output (List of arrays) -> Select Positive Class (Index 1)
            if isinstance(shap_values, list):
                # For multiclass (5 classes), we might want to aggregate or pick one.
                # Usually summary_plot handles list of arrays by plotting stacked bars or we pick one class.
                # Given this is a diagnostic, let's aggregate absolute impact across all classes
                # to see overall feature importance.
                # However, shap.summary_plot with plot_type='dot' expects a single matrix for color coding.
                # If we pass a list, it does class breakdown.
                # Let's stick to the default behavior of summary_plot which handles lists (usually class 0 or stacked).
                # Or better, calculate mean abs importance for the plot if we want a simple view, 
                # but summary_plot is smarter.
                # Let's pass the raw list, shap handles it.
                shap_values_for_plot = shap_values
            else:
                shap_values_for_plot = shap_values

            # 4. Generate Summary Plot
            # CRITICAL FIX: Must pass X_test_head_augmented (Full Features), not just specialist subset.
            # Passing a subset causes dimension mismatch with shap_values.
            plt.figure(figsize=(10, 6))
            
            # Dynamic max_display to avoid errors if total features < top_n
            dynamic_max_display = min(top_n, X_test_head_augmented.shape[1])
            
            shap.summary_plot(shap_values_for_plot, X_test_head_augmented, show=False, max_display=dynamic_max_display)
            
            plt.title(f"[REDACTED_BY_SCRIPT]", fontsize=14)
            plot_filename = f"[REDACTED_BY_SCRIPT]"
            plot_path = os.path.join(shap_output_dir, plot_filename)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            report_content.append(f"### {cohort_name}\n")
            report_content.append(f"[REDACTED_BY_SCRIPT]")
            
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            continue
        
    return "".join(report_content)

# --- AD-AM-31: Forensic Diagnostic Framework ---

def phase_D6_failure_case_waterfalls(model, arbiter_test_matrix, results_df, output_dir):
    """
    AD-AM-31.1: Generates SHAP waterfall plots for the most extreme overestimation
    and underestimation cases in the holdout set to dissect failure modes.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # AD-AM-32 REMEDIATION: The final model is a raw LGBMRegressor. The .regressor_
    # attribute is no longer present. The model object itself is the regressor.
    explainer = shap.TreeExplainer(model)

    def _generate_case_study(case_type, case_index):
        instance = arbiter_test_matrix.loc[[case_index]]
        shap_values_instance = explainer.shap_values(instance)
        
        plt.figure()
        shap.plots.waterfall(shap.Explanation(
            values=shap_values_instance[0],
            base_values=explainer.expected_value,
            data=instance.iloc[0],
            feature_names=arbiter_test_matrix.columns
        ), max_display=20, show=False)
        
        true_val = results_df.loc[case_index, 'y_true']
        pred_val = results_df.loc[case_index, 'y_pred']
        error_val = results_df.loc[case_index, 'error']

        plt.title(f"[REDACTED_BY_SCRIPT]", fontsize=14)
        plot_filename = f"[REDACTED_BY_SCRIPT]' ', '_')}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"[REDACTED_BY_SCRIPT]")

    # Identify and analyze the worst failure cases
    underestimation_idx = results_df['error'].idxmin()
    overestimation_idx = results_df['error'].idxmax()
    
    _generate_case_study("[REDACTED_BY_SCRIPT]", underestimation_idx)
    _generate_case_study("[REDACTED_BY_SCRIPT]", overestimation_idx)

def phase_D7_expert_correlation_heatmap(X_test, cohorts, head_models_dir, oracle_preds_test, output_dir):
    """
    AD-AM-31.2: Analyzes the Pearson correlation between the Oracle and specialist
    head predictions to identify informational redundancy and uniqueness.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    expert_predictions = pd.DataFrame({'[REDACTED_BY_SCRIPT]': oracle_preds_test})

    for cohort_name, features in cohorts.items():
        model_path = os.path.join(head_models_dir, f"[REDACTED_BY_SCRIPT]")
        if not os.path.exists(model_path): continue
        model = joblib.load(model_path)
        model_features = [f for f in features if f in X_test.columns]
        if not model_features: continue
        
        pred = model.predict(X_test[model_features])
        expert_predictions[f"[REDACTED_BY_SCRIPT]"] = pred

    corr_matrix = expert_predictions.corr(method='pearson')
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("[REDACTED_BY_SCRIPT]", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plot_path = os.path.join(output_dir, "[REDACTED_BY_SCRIPT]")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"[REDACTED_BY_SCRIPT]")

def phase_D8_performance_stratification_report(results_df, X_test):
    """
    AD-AM-31.3: Calculates and reports model performance (RMSE) stratified by
    key business-critical features to identify systematic weaknesses.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    
    data = results_df.join(X_test)
    data['abs_error'] = results_df['error'].abs()
    
    ## Stratify by Project Scale
    print("[REDACTED_BY_SCRIPT]")

    # Create quartiles
    data['scale_quartile'] = pd.qcut(data['[REDACTED_BY_SCRIPT]'], 4, labels=['Q1 (Small)', 'Q2', 'Q3', 'Q4 (Large)'], duplicates='drop')

    # Define metric functions
    def rmse(x):
        return np.sqrt(np.mean(x**2))

    def mae(x):
        return np.mean(np.abs(x))

    def mbe(x):
        return np.mean(x) # Mean Bias Error (if 'abs_error' is actually raw error, otherwise this is just mean absolute error again)

    # Apply multiple aggregations
    scale_performance = data.groupby('scale_quartile')['abs_error'].agg([rmse, mae, 'count'])

    # Rename columns for clarity
    scale_performance.columns = ['RMSE', 'MAE', 'Count']

    print(scale_performance.to_string())

def phase_H_forge_solar_bridge(X_train, y_train, X_val, X_test, n_top_features=10, n_pca_components=5):
    """
    AD-AM-45: Forging Solar Bridge (Protocol v5.0).
    REMEDIATION: Implements 'Protected Features' logic. 
    Exempts critical 'Axes of Variance' from PCA to enable downstream stratification
    and regime-specific corrections.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Filter Training Data to Solar Only (Type 21)
    # Critical: Must filter y_train to match X_train dimensions to prevent LightGBM Error.
    # Using index alignment with global master_df to ensure safety even if column was dropped locally.
    mask = (master_df.loc[X_train.index, 'technology_type'] == 21)
    X_train = X_train.loc[mask]
    y_train = y_train.loc[mask]

    # 1. Identify Top N Features
    print(f"[REDACTED_BY_SCRIPT]")
    probe = lgb.LGBMRegressor(
        random_state=67, n_jobs=-1, verbosity=-1, 
        max_depth=7, n_estimators=750, 
        learning_rate=0.001, min_split_gain=0.01,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=max(5, int(len(X_train) * 0.1)), # Probe must respect data density
        reg_alpha=10.0 # High regularization for selection
    )
    probe.fit(X_train, y_train)
    
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': probe.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance['feature'].head(n_top_features).tolist()
    
    # 2. Define Feature Categories
    # ARCHITECTURAL MANDATE: Protect the 'Axes of Variance' from PCA compression.
    candidates_for_protection = [
        # --- Core Physical & Regulatory Axes ---
        '[REDACTED_BY_SCRIPT]', 'technology_type', '[REDACTED_BY_SCRIPT]',
        'SIC_POLICY_REGIME_ID', 'lpa_major_commercial_approval_rate',
        '[REDACTED_BY_SCRIPT]', 'lpa_workload_trend',
        '[REDACTED_BY_SCRIPT]', 'SIC_LPA_POLITICAL_VOLATILITY',

        # --- V6.0: Speculator & NSIP Detection ---
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'SIC_NSIP_LPA_BYPASS_FLAG',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',

        # --- V7.0: Evasion & Step-Up Risks ---
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',

        # --- V8.0: Rhythm & Agitation ---
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',

        # --- V9.0: Structural Friction & Regional Hostility ---
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'SIC_GRID_CONNECTION_EFFICIENCY', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',

        # --- V10.0: Common Sense & Physics ---
        'SIC_SOCIAL_OCCUPATION_RATIO', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'SIC_AGRI_SCARCITY_RATIO', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',

        # --- V11.0: Mid-Scale Rescue & Grid Edge ---
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'SIC_MID_LPA_NOVICE_PANIC',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'SIC_MID_ROAD_ACCESS_CONSTRICTION', '[REDACTED_BY_SCRIPT]'

        # --- V12.0: Distribution Grit (0-10MW) ---
        '[REDACTED_BY_SCRIPT]', 'SIC_GRIT_RANSOM_STRIP_RISK',
        'SIC_GRIT_COMMITTEE_CLIFF_EDGE', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'SIC_GRIT_BEST_FIELD_SACRIFICE', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
        # --- V13.0: Grid Edge & Social Inertia (0-10MW) ---
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
        # --- V14.0: Hazards & Margins (0-10MW) ---
        'SIC_HAZARD_RAIL_GLARE_RISK', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    
    # Robustly select only features present in the dataset
    protected_features = [f for f in candidates_for_protection if f in X_train.columns]
    
    # Remove protected from top (to avoid duplication) and other
    top_features_clean = [f for f in top_features if f not in protected_features]
    other_features = [c for c in X_train.columns if c not in top_features_clean and c not in protected_features]
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 3. Fit PCA on 'Other' features
    print(f"[REDACTED_BY_SCRIPT]")
    pca = PCA(n_components=n_pca_components, random_state=67)
    scaler_pca = StandardScaler()
    
    X_train_other_scaled = scaler_pca.fit_transform(X_train[other_features])
    pca.fit(X_train_other_scaled)
    
    # 4. Transformation Pipeline with Protection
    def _compress(X_df):
        # A. Protected Features (Must exist)
        X_protected = X_df[protected_features].reset_index(drop=True)
        
        # B. Top Features
        X_top = X_df[top_features_clean].reset_index(drop=True)
        
        # C. PCA Features
        X_other_scaled = scaler_pca.transform(X_df[other_features])
        X_pca = pd.DataFrame(
            pca.transform(X_other_scaled), 
            columns=[f'PCA_{i}' for i in range(n_pca_components)]
        )
        
        # D. Recombine
        X_compressed = pd.concat([X_protected, X_top, X_pca], axis=1)
        X_compressed.index = X_df.index
        return X_compressed

    X_train_compressed = _compress(X_train)
    X_val_compressed = _compress(X_val)
    X_test_compressed = _compress(X_test)
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    return X_train_compressed, X_val_compressed, X_test_compressed, top_features_clean, pca, scaler_pca


def phase_I_train_solar_residual_specialist(X_train_comp, y_residual, n_trials=50):
    """
    AD-AM-45.2: Trains a highly regularized regressor on the Solar Bridge features
    to predict the Arbiter's error.
    
    UPDATED: Now uses Optuna to find the optimal regularization structure for this 
    specific residual surface, rather than hardcoded assumptions.
    """
    print("[REDACTED_BY_SCRIPT]")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import root_mean_squared_error
    
    # Create a local validation split for the optimization process
    # Since N is small (~1000), we use a small validation set (20%)
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_train_comp, y_residual, test_size=0.2, random_state=67
    )

    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1,
            
            # Search Space tailored for "Small N, High Noise" (Regularization Focus)
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 7, 31),
            'max_depth': trial.suggest_int('max_depth', 2, 5), # Keep trees shallow
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0, log=True), # High regularization floor
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 0.1, log=True)
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_opt_train, y_opt_train)
        preds = model.predict(X_opt_val)
        mae = mean_absolute_error(y_opt_val, preds)
        return mae

    # Run Optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Retrain on FULL training data with best params
    best_params = study.best_params
    best_params.update({
        'objective': 'regression_l1',
        'metric': 'mae',
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    })
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train_comp, y_residual)

    print("[REDACTED_BY_SCRIPT]")
    return final_model

def phase_M_train_terminal_calibrator(X_test_comp, y_test, final_predictions):
    """
    AD-AM-FINAL: Trains a Terminal Calibrator on the FINAL residuals.
    This is the only probability score that should be shown to the user.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    # 1. Calculate the FINAL residual (after all 6 layers of correction)
    final_residuals = y_test - final_predictions
    
    # 2. Define Error Buckets for the User Interface
    # We use slightly wider bins because the final model should be more accurate
    bins = [-np.inf, -50, 50, np.inf] 
    labels = [0, 1, 2] # 0: Underpredicted (>50 days), 1: Accurate (+/- 50), 2: Overpredicted (>50 days)
    
    y_calibrator = pd.cut(final_residuals, bins=bins, labels=labels).astype(int)
    
    # 3. Train a lightweight classifier
    # We use the compressed features from the last stage (GM Solar Compressed)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'n_estimators': 200,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': 3,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
        'min_split_gain': 0.01
    }
    
    calibrator = lgb.LGBMClassifier(**params)
    calibrator.fit(X_test_comp, y_calibrator)
    
    print("[REDACTED_BY_SCRIPT]")
    return calibrator

def engineer_bess_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The '[REDACTED_BY_SCRIPT]' Engine.
    Models the high-variance regime caused by Solar + Battery co-location.
    Focuses on Safety Friction (Fire), Grid Complexity (Import/Export), and Cumulative Impact.
    """
    X = df.copy()
    epsilon = 1e-6

    # --- 1. The Co-Location Flag (Implicit) ---
    # High Solar Capacity + High Existing/Planned Storage Density = Likely Energy Hub.
    # We sum sub-1MW and over-1MW storage to get total local battery context.
    total_storage_density_mwh = (
        X.get('[REDACTED_BY_SCRIPT]', 0) + 
        X.get('[REDACTED_BY_SCRIPT]', 0)
    )
    
    # Interaction: Solar Scale * Storage Density
    X['SIC_BESS_CO_LOCATION_INTENSITY'] = X['[REDACTED_BY_SCRIPT]'] * total_storage_density_mwh

    # --- 2. The "Fire Safety" Friction (The Variance Driver) ---
    # Batteries near settlements trigger intense "Thermal Runaway" / Fire Safety objections.
    # This causes massive variance in decision times (consultation loops).
    # Formula: Storage Density * Settlement Density
    X['SIC_BESS_FIRE_SAFETY_FRICTION'] = total_storage_density_mwh * X['[REDACTED_BY_SCRIPT]']

    # --- 3. The Grid "Vampire" Effect (Headroom Conflict) ---
    # Solar needs Export Headroom. Batteries need Import AND Export.
    # Co-location strains the substation in both directions, complicating the DNO/National Grid approval.
    # This models the "Grid Complexity" delay.
    X['[REDACTED_BY_SCRIPT]'] = total_storage_density_mwh / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # --- 4. The "Tech-Clash" Volatility Amplifier ---
    # If an LPA is already volatile (high decision variance) AND has to deal with complex Battery tech,
    # the chaos multiplies.
    if '[REDACTED_BY_SCRIPT]' in X.columns:
        X['[REDACTED_BY_SCRIPT]'] = total_storage_density_mwh * X['[REDACTED_BY_SCRIPT]']
    else:
        # Fallback if volatility metric missing
        X['[REDACTED_BY_SCRIPT]'] = total_storage_density_mwh * X['[REDACTED_BY_SCRIPT]']

    # --- 5. The "[REDACTED_BY_SCRIPT]" Barrier ---
    # A large Solar farm is one thing. Solar + Batteries looks like a "Power Station" to locals.
    # This checks if the site is already in an industrialized area (low friction) or a greenfield (high friction).
    # Greenfields with Batteries face exponential resistance.
    is_greenfield_proxy = 1 - (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_BESS_CO_LOCATION_INTENSITY'] * is_greenfield_proxy

    return X


def engineer_tower_paradox_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Tower Paradox' Engine.
    Decodes why high existing infrastructure density (132kV Towers) leads to LONGER delays.
    Models '[REDACTED_BY_SCRIPT]' (Social), 'Wayleave Complexity' (Technical), and 'Green Belt' conflicts.
    """
    X = df.copy()
    epsilon = 1e-6

    # --- 1. The "[REDACTED_BY_SCRIPT]" Index (Social Driver) ---
    # Hypothesis: Communities with high tower density feel "dumped on".
    # Adding a LARGE new solar farm to an area already full of pylons triggers a "Enough is Enough" reaction.
    # Formula: Solar Scale * Total Tower Density
    total_tower_density = X.get('[REDACTED_BY_SCRIPT]', 0) + X.get('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * total_tower_density

    # --- 2. The "[REDACTED_BY_SCRIPT]" (Technical Driver) ---
    # Hypothesis: It's not political, it's engineering. High tower density means the connection path 
    # has to navigate a minefield of existing high-voltage assets (Wayleaves, Crossing Agreements, Safety Zones).
    # Formula: Connection Path Intersections * 132kV Tower Density
    # High intersections in a high-density zone = Maximum Complexity.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X.get('[REDACTED_BY_SCRIPT]', 0)

    # --- 3. The "Urban Fringe" Gridlock (Socio-Spatial Driver) ---
    # Hypothesis: High tower density often marks the "Urban Fringe" or Green Belt (substations serving cities).
    # These areas are fiercely defended by suburban populations.
    # Formula: Settlement Density * Tower Density
    # This isolates the "[REDACTED_BY_SCRIPT]" effect from the "[REDACTED_BY_SCRIPT]" effect.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * total_tower_density

    # --- 4. The "Sterilization" Constraint ---
    # Hypothesis: Large transmission corridors physically sterilize land, forcing the solar farm into 
    # suboptimal, fragmented layouts that take longer to approve and design.
    # Formula: 132kV Sterilized Area * Site Capacity
    # Note: Using '[REDACTED_BY_SCRIPT]' if available, otherwise proxy with corridor intersection.
    sterilization_factor = X.get('[REDACTED_BY_SCRIPT]', X.get('[REDACTED_BY_SCRIPT]', 0))
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * sterilization_factor

    return X

def engineer_infrastructure_crossing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The '[REDACTED_BY_SCRIPT]' Engine.
    Models the legal and physical friction of building connection lines over existing infrastructure.
    Captures 'Wayleave Volume', 'Path Tortuosity', and '[REDACTED_BY_SCRIPT]'.
    """
    X = df.copy()
    epsilon = 1e-6

    # --- 1. Wayleave Volume (Legal Friction) ---
    # The absolute count of infrastructure intersections (Rail/Road/River).
    # Each intersection represents a legal "Wayleave" or "Easement" agreement, each with its own timeline.
    # We sum High Voltage (DHV) and Low Voltage (DLV) path intersection counts if distinct, or use total.
    # Based on input features, '[REDACTED_BY_SCRIPT]' is the master metric.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]']

    # --- 2. Path Tortuosity (Density of Obstacles) ---
    # A short line with 10 crossings is a nightmare. A long line with 10 crossings is manageable.
    # This feature measures the "Obstacle Density" of the connection route.
    # High density = intense localized engineering and legal complexity.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # --- 3. Capacity-Weighted Crossing Difficulty (Engineering Friction) ---
    # Crossing a motorway with a 132kV tower (Large Project) is infinitely harder than stringing a 11kV wooden pole line (Small Project).
    # We weight the intersection count by the project's installed capacity to model this engineering scaling.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # --- 4. The "Pole-Sprawl" Interaction ---
    # The prompt specifically mentions "number of poles".
    # While we estimate new poles via distance, we also have data on *existing* pole density (`dhv_pole_density_in_2km_per_km2`).
    # Hypothesis: Building new crossings in an area already cluttered with poles increases "Visual Clutter" objections alongside the technical crossing difficulty.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X.get('[REDACTED_BY_SCRIPT]', 0)

    # --- 5. The "Encirclement" Context ---
    # Even if the specific path intersection count is low, being in a site surrounded by major transport links (Rail/Road length)
    # implies a generally difficult logistical environment for construction access (e.g., getting cranes to site).
    surrounding_infra_density = X.get('[REDACTED_BY_SCRIPT]', 0) + X.get('railway_length_2km', 0)
    X['[REDACTED_BY_SCRIPT]'] = surrounding_infra_density * X['[REDACTED_BY_SCRIPT]']

    return X

def engineer_ecological_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Ecological Variance' Engine.
    Models the high-variance delays caused by specific 'Survey Seasons' and 'Veto' constraints.
    Targets Priority Habitats (PH), Special Areas of Conservation (SAC), and Ancient Woodland (AW).
    """
    X = df.copy()
    epsilon = 1e-6

    # --- 1. Priority Habitat: The "Seasonal Survey" Variance ---
    # Hypothesis: Variance is driven by the *Type* of habitat.
    # Wetlands/Heathlands = Birds/Newts = Specific seasonal windows = High Variance.
    # We weight the specific habitat group flag by the proximity to the site.
    
    # Calculate inverse distance (Proximity Score)
    ph_proximity = 1 / (X['ph_dist_to_nearest_m'] + 100) # +100m smoothing
    
    # Habitat-Specific Risk Scores
    # Note: Using .get() to be safe, though columns should exist based on feature set.
    X['SIC_PH_WETLAND_RISK'] = ph_proximity * X.get('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = ph_proximity * X.get('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = ph_proximity * X.get('[REDACTED_BY_SCRIPT]', 0)
    X['SIC_PH_COASTAL_RISK'] = ph_proximity * X.get('[REDACTED_BY_SCRIPT]', 0)
    
    # "Unknown" or "Other" implies poor data quality, which itself is a delay risk.
    X['[REDACTED_BY_SCRIPT]'] = ph_proximity * (X.get('[REDACTED_BY_SCRIPT]', 0) + X.get('[REDACTED_BY_SCRIPT]', 0))

    # --- 2. Ancient Woodland: The "Irreplaceability" Factor ---
    # ASNW (Ancient Semi-Natural Woodland) is legally "irreplaceable". Objections are absolute.
    # PAWS (Plantations on Ancient Woodland) is degraded. Mitigation is possible.
    # We model the distinct friction of ASNW vs PAWS.
    aw_proximity = 1 / (X['aw_dist_to_nearest_m'] + 100)
    
    X['[REDACTED_BY_SCRIPT]'] = aw_proximity * X.get('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = aw_proximity * X.get('[REDACTED_BY_SCRIPT]', 0)

    # --- 3. SAC: The "[REDACTED_BY_SCRIPT]" (HRA) Trigger ---
    # SACs trigger a "[REDACTED_BY_SCRIPT]" (LSE) test and potentially a full HRA.
    # This process is lengthy and scales with the size of the SAC (wider impact zone).
    # Formula: Proximity * SAC Area (Proxy for importance/impact zone)
    sac_proximity = 1 / (X['sac_dist_to_nearest_m'] + 100)
    X['[REDACTED_BY_SCRIPT]'] = sac_proximity * X['sac_nearest_area_ha']

    # --- 4. The "Triple Lock" Cumulative Veto ---
    # If a site is near ALL THREE "Worst Offenders", it is likely a non-starter or will face years of delay.
    # We sum the proximity scores.
    X['SIC_ECO_TRIPLE_LOCK_VETO'] = ph_proximity + aw_proximity + sac_proximity

    # --- 5. Capacity-Weighted Sensitivity ---
    # A 50MW solar farm has a much larger ecological footprint than a 5MW one.
    # The friction of these habitats scales linearly (or exponentially) with project size.
    X['SIC_ECO_SCALE_SENSITIVITY'] = X['[REDACTED_BY_SCRIPT]'] * X['SIC_ECO_TRIPLE_LOCK_VETO']

    return X


def engineer_demographic_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The '[REDACTED_BY_SCRIPT]' Engine.
    Quantifies the 'NIMBY' friction generated by specific local demographic profiles.
    Models 'Resistance Strength', '[REDACTED_BY_SCRIPT]', and 'Amenity Defence'.
    """
    X = df.copy()
    epsilon = 1e-6

    # --- 1. The "Resistance Bloc" Strength ---
    # User-identified high-friction groups: 'e-Veterans' and '[REDACTED_BY_SCRIPT]'.
    # We aggregate their presence scores to create a composite "Opposition Index".
    # Note: Using .get() to handle potential missing columns safely.
    resistance_score = (
        X.get('site_lsoa_oac_e-Veterans', 0) + 
        X.get('[REDACTED_BY_SCRIPT]', 0)
    )
    X['[REDACTED_BY_SCRIPT]'] = resistance_score

    # --- 2. The "Digital Exclusion" Friction ---
    # Groups that are less likely to engage with digital planning portals often result in 
    # late-stage objections ("We weren't consulted") and political intervention.
    # Includes: 'e-Withdrawn', 'Digital Seniors', '[REDACTED_BY_SCRIPT]'.
    digital_exclusion_score = (
        X.get('[REDACTED_BY_SCRIPT]', 0) + 
        X.get('[REDACTED_BY_SCRIPT]', 0) + 
        X.get('[REDACTED_BY_SCRIPT]', 0)
    )
    X['[REDACTED_BY_SCRIPT]'] = digital_exclusion_score

    # --- 3. The "[REDACTED_BY_SCRIPT]" (NIMBY Scalar) ---
    # A small barn conversion might be ignored. A 50MW solar farm triggers the bloc.
    # Interaction: Project Scale * Resistance Strength.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * resistance_score

    # --- 4. The "Amenity Defence" Interaction ---
    # Hypothesis: These specific groups are the primary defenders of local amenities (walking routes, parks).
    # Friction increases exponentially as the site gets closer to National Trails (NT) or Historic Parks (HP).
    # Formula: Resistance Score / Distance to Amenity
    # We construct a composite amenity proximity metric first.
    dist_to_amenity = X[['nt_dist_to_nearest_m', 'hp_dist_to_nearest_m']].min(axis=1)
    X['SIC_DEMO_AMENITY_DEFENCE_INTENSITY'] = resistance_score / (dist_to_amenity + 100)

    # --- 5. The "Rural Character" Defence ---
    # Hypothesis: '[REDACTED_BY_SCRIPT]' in rural areas (Rural-Urban Classification) fight harder to preserve "character".
    # Interaction: Resistance Score * Rural Score (if available) OR Inverse Settlement Density (proxy for rurality).
    # Using inverse settlement density as a robust proxy for rural character value.
    rural_proxy = 1 / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = resistance_score * rural_proxy

    return X

def engineer_physical_footprint_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Physical Footprint' Engine.
    Models the impact of Physical Size (Log Scale) and Power Density (Land Use Efficiency).
    Captures 'Scale Shock', 'Sprawl Friction', and 'Density Efficiency'.
    """
    X = df.copy()
    epsilon = 1e-6

    # --- Data Hardening: Ensure Area Exists ---
    # If solar_site_area_sqm was previously dropped or missing, we must handle it.
    # If missing, we construct a "Theoretical Area" based on standard industry density (approx 1.6 hectares/MW)
    # to prevent pipeline failure, while flagging the synthetic nature.
    if 'solar_site_area_sqm' not in X.columns or X['solar_site_area_sqm'].isnull().all():
        # Fallback: Estimate 16,000 sqm per MW (1.6 hectares/MW)
        X['temp_area_sqm'] = X['[REDACTED_BY_SCRIPT]'] * 16000
    else:
        X['temp_area_sqm'] = X['solar_site_area_sqm'].fillna(X['[REDACTED_BY_SCRIPT]'] * 16000)

    # --- 1. The "Scale Shock" (Log Size) ---
    # As requested: "[REDACTED_BY_SCRIPT]".
    # Log-transformation is crucial because objection volume tends to scale logarithmically with visual magnitude, not linearly.
    X['[REDACTED_BY_SCRIPT]'] = np.log1p(X['temp_area_sqm'])

    # --- 2. Power Density (Land Use Efficiency) ---
    # As requested: "[REDACTED_BY_SCRIPT]".
    # Calculated as Watts per Square Meter. High Density = Efficient. Low Density = Sprawl.
    # MW * 1,000,000 / Sqm
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] * 1e6) / (X['temp_area_sqm'] + epsilon)

    # --- 3. The "Sprawl" Friction ---
    # Projects with low power density (inefficient sprawl) in areas with high settlement density face
    # accusations of "Wasting Land".
    # Formula: (1 / Density) * Settlement Density
    X['SIC_SPRAWL_FRICTION'] = (1 / (X['[REDACTED_BY_SCRIPT]'] + epsilon)) * X['[REDACTED_BY_SCRIPT]']

    # --- 4. "Industrial Scale" Visual Impact Proxy ---
    # Large area + High Terrain Gradient = Highly Visible Scar.
    # Interaction: Log Area * StdDev Terrain Gradient (Visual Exposure)
    if '[REDACTED_BY_SCRIPT]' in X.columns:
        X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + 1)
    
    # Clean up temp column
    X.drop(columns=['temp_area_sqm'], inplace=True)

    return X

def engineer_brownfield_shield_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Brownfield Shield' Engine.
    Models the 'Variance Dampener' effect of siting near existing energy infrastructure or brownfield land.
    Captures 'Legacy Shield', 'Industrial Context', and 'Shielded Capacity'.
    """
    X = df.copy()
    epsilon = 1e-6

    # --- Data Hardening: Handle Missing Legacy Distances ---
    # If no legacy site is nearby, distance might be NaN. We assume a "Safe Distance" of 50km for the math to work.
    legacy_dist = X['[REDACTED_BY_SCRIPT]'].fillna(50)

    # --- 1. Legacy Shield Intensity (The "Incumbent" Effect) ---
    # High count of nearby legacy sites + Close proximity = Strongest Shield.
    # This signals that the area is already accepted as an energy generation zone.
    X['[REDACTED_BY_SCRIPT]'] = X['nearby_legacy_count'] / (legacy_dist + epsilon)

    # --- 2. Brownfield Context Score (The "Industrial" Effect) ---
    # User mentioned "[REDACTED_BY_SCRIPT]".
    # We use the 1km Industrial Area Percentage as a direct proxy for immediate proximity.
    # We boost this if the site itself is flagged as Urban/Industrial (`nhlc_urban_on_site_bool`).
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] + (X['[REDACTED_BY_SCRIPT]'] * 10)

    # --- 3. The "Cluster Bonus" (Variance Dampener) ---
    # User noted "reduces variance".
    # Sites with BOTH legacy neighbors AND high industrial density are "Safe Bets".
    # We create a composite "Safety Score".
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] * 0.5) + (X['[REDACTED_BY_SCRIPT]'] * 0.5)

    # --- 4. Shielded Capacity (The "Hidden Giant") ---
    # A large project (High Capacity) is much more acceptable if it has a High Shield Score.
    # This interaction allows the model to discount the penalty of size in favorable locations.
    X['SIC_SHIELDED_CAPACITY'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

    # --- 5. The "Precedent" Log-Decay ---
    # Modeling the rapid decay of the precedent benefit as distance increases.
    # "Proximity" is key.
    X['[REDACTED_BY_SCRIPT]'] = np.log1p(1 / (legacy_dist + epsilon))

    return X

def engineer_ecological_drag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Ecological Drag' Engine.
    Implements the Stratum-Specific Feature Interactions required to resolve the 
    '[REDACTED_BY_SCRIPT]' paradox.
    
    Directives:
    1. Decay Buffer: Inverse-distance weighting clipped at 2km.
    2. Stratum Gates: Separate features for High-Cap/High-Risk vs Low-Cap/Low-Risk.
    3. IRZ Proxy: Footprint-to-Proximity Ratio.
    4. Uncertainty Score: Scrutiny flags.
    5. Golden Ratio: Capacity Density vs Sensitivity.
    """
    print("[REDACTED_BY_SCRIPT]")
    X = df.copy()
    epsilon = 1e-6

    # Data Hardening: Ensure critical columns exist
    if 'sssi_dist_to_nearest_m' not in X.columns:
        print("  - WARNING: 'sssi_dist_to_nearest_m'[REDACTED_BY_SCRIPT]")
        X['sssi_dist_to_nearest_m'] = 5000
    
    # 1. The "Decay Buffer" Transformation (Directive 1)
    # Inverse-distance weighting with a relevance floor.
    # We use a linear decay from 0m (Score 1.0) to 2000m (Score 0.0).
    # This focuses attention entirely on the "Danger Zone".
    X['temp_sssi_dist_clipped'] = X['sssi_dist_to_nearest_m'].clip(upper=2000)
    X['SIC_ECO_DECAY_BUFFER_SCORE'] = (2000 - X['temp_sssi_dist_clipped']) / 2000.0

    # 2. Stratum-Specific Interaction Gates (Directive 2)
    # Gate A: High Capacity (>25MW) in Danger Zone (<2km).
    # This allows the model to learn a positive coefficient (Penalty).
    mask_high_cap = (X['[REDACTED_BY_SCRIPT]'] > 25) & (X['sssi_dist_to_nearest_m'] < 2000)
    X['[REDACTED_BY_SCRIPT]'] = 0.0
    X.loc[mask_high_cap, '[REDACTED_BY_SCRIPT]'] = X.loc[mask_high_cap, 'SIC_ECO_DECAY_BUFFER_SCORE'] * X.loc[mask_high_cap, '[REDACTED_BY_SCRIPT]']

    # Gate B: Low Capacity (<10MW) in Proximity (<500m).
    # This allows the model to learn a negative/neutral coefficient (Shielding/Survivorship).
    mask_low_cap = (X['[REDACTED_BY_SCRIPT]'] < 10) & (X['sssi_dist_to_nearest_m'] < 500)
    X['[REDACTED_BY_SCRIPT]'] = 0.0
    X.loc[mask_low_cap, '[REDACTED_BY_SCRIPT]'] = X.loc[mask_low_cap, 'SIC_ECO_DECAY_BUFFER_SCORE']

    # 3. The "Impact Risk Zone" (IRZ) Proxy (Directive 3)
    # Footprint-to-Proximity Ratio.
    # Recalculate estimated area (hecta-scale) locally to ensure statelessness.
    # Assumption: ~1.6 hectares per MW.
    est_area_hectares = X['[REDACTED_BY_SCRIPT]'] * 1.6
    X['SIC_ECO_IRZ_PROXY_RATIO'] = est_area_hectares / (X['sssi_dist_to_nearest_m'] + epsilon)

    # 4. Ecological Uncertainty Score (Directive 4)
    # Flags "[REDACTED_BY_SCRIPT]" zones where variance explodes.
    # Mid-sized projects (10-25MW) near SSSIs (<1km) are the most volatile.
    X['[REDACTED_BY_SCRIPT]'] = (
        (X['[REDACTED_BY_SCRIPT]'] >= 10) & 
        (X['[REDACTED_BY_SCRIPT]'] <= 25) & 
        (X['sssi_dist_to_nearest_m'] < 1000)
    ).astype(int)

    # 5. The "Golden Ratio" of Planning (Directive 5)
    # Capacity Density vs Ecological Sensitivity.
    # A continuous pressure variable scaling linearly with size.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['SIC_ECO_DECAY_BUFFER_SCORE']

    # Cleanup
    X.drop(columns=['temp_sssi_dist_clipped'], inplace=True)
    
    return X


def engineer_stratum_ecological_drag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Ecological Drag' Engine.
    Generates stratum-specific environmental friction features mandated by 
    the '[REDACTED_BY_SCRIPT]' forensic analysis.
    """
    X = df.copy()
    # Robust extraction with safe defaults
    dist = X.get('sssi_dist_to_nearest_m', 5000).fillna(5000)
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # 1. Decay Buffer (Directive 1)
    # Inverse-distance weighting clipped at 2km. Focuses on the "Danger Zone".
    decay_score = (2000 - dist.clip(upper=2000)) / 2000.0
    
    # 2. Stratum Gates (Directive 2)
    # Gate A: High Capacity (>25MW) in Danger Zone (<2km) -> Penalty
    high_cap_mask = (cap > 25) & (dist < 2000)
    drag_high = pd.Series(0.0, index=X.index)
    drag_high[high_cap_mask] = decay_score[high_cap_mask] * cap[high_cap_mask]
    
    # Gate B: Low Capacity (<10MW) in Proximity (<500m) -> Shield (Neutral/Negative)
    low_cap_mask = (cap < 10) & (dist < 500)
    shield_low = pd.Series(0.0, index=X.index)
    shield_low[low_cap_mask] = decay_score[low_cap_mask]
    
    # 3. IRZ Proxy (Directive 3)
    # Footprint-to-Proximity Ratio. Assumption: ~1.6 hectares per MW.
    area_ha = cap * 1.6
    irz_proxy = area_ha / (dist + 1e-6)
    
    # 4. Uncertainty Flag (Directive 4)
    # Flags "[REDACTED_BY_SCRIPT]" (10-25MW, <1km) where variance explodes.
    uncertainty = ((cap >= 10) & (cap <= 25) & (dist < 1000)).astype(int)
    
    # 5. Golden Ratio (Directive 5)
    # Continuous pressure: Capacity Density vs Sensitivity.
    golden_ratio = cap * decay_score
    
    # Return DataFrame of ONLY the new features for clean appending
    return pd.DataFrame({
        'SIC_ECO_DECAY_SCORE': decay_score,
        'SIC_ECO_DRAG_HIGH_CAP': drag_high,
        'SIC_ECO_SHIELD_LOW_CAP': shield_low,
        'SIC_ECO_IRZ_PROXY': irz_proxy,
        '[REDACTED_BY_SCRIPT]': uncertainty,
        '[REDACTED_BY_SCRIPT]': golden_ratio
    }, index=X.index)


def engineer_bmv_gold_plate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Gold-Plated BMV' Engine.
    Generates stratum-specific features to model the 'BMV Paradox':
    1. Technical Complexity Proxy (Anti-BMV).
    2. Gold-Plated Probability Gate (Accelerator vs NSIP Shadow).
    3. Regional Land Scarcity (Sequential Test Context).
    """
    X = df.copy()
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_bmv = X.get('alc_is_bmv_at_site', 0).fillna(0)
    
    # 1. Technical Complexity Proxy (The Anti-BMV) (Directive 1)
    # Logic: Non-BMV (0) * Terrain Gradient. High Gradient on Non-BMV = Slow/Difficult.
    # We use .get() with a default of 0 to be robust if the gradient feature is missing.
    terrain_gradient = X.get('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = (1 - is_bmv) * terrain_gradient

    # 2. Gold-Plated Probability Gate (Directive 2)
    # Level 2 (Fast): 15-40MW on BMV. (Survivorship Bias - "The Winners")
    # Accelerator: High value = Short duration.
    X['SIC_BMV_ACCELERATOR'] = ((cap >= 15) & (cap <= 40) & (is_bmv == 1)).astype(int)
    
    # Level 3 (Slow): >40MW on BMV. (NSIP Shadow / Food Security Scrutiny)
    X['SIC_BMV_NSIP_SHADOW'] = ((cap > 40) & (is_bmv == 1)).astype(int)

    # 3. Regional Land Scarcity (Sequential Test Flag) (Directive 3)
    # Logic: High Penalty if Site is BMV (1) but Region has alternatives (Low BMV %).
    # If Region is 100% BMV, penalty is 0 (No choice).
    # If Region is 0% BMV, penalty is 1 (Why here?).
    region_bmv_pct = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = is_bmv * (1 - (region_bmv_pct / 100.0))

    return X[['[REDACTED_BY_SCRIPT]', 'SIC_BMV_ACCELERATOR', 'SIC_BMV_NSIP_SHADOW', '[REDACTED_BY_SCRIPT]']]

def engineer_stratified_specialist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidated entry point for all Stratum-Specific features.
    """
    # NOTE: '[REDACTED_BY_SCRIPT]' is removed to prevent duplication of 
    # '[REDACTED_BY_SCRIPT]' which is generated globally by Mandate 26.1.
    # df_eco = engineer_stratum_ecological_drag(df) <--- REMOVED
    
    df_bmv = engineer_bmv_gold_plate_features(df)
    df_lpa = engineer_lpa_infrastructure_competence(df)
    df_phys_pol = engineer_physics_and_politics_features(df)
    df_uncertainty = engineer_uncertainty_paradox_features(df)
    df_socio_a = engineer_sociological_friction_features(df)
    df_contract = engineer_contractual_inertia_features(df)
    
    # Directive: Call the new Bio-Temporal Ecology Engine
    df_taxonomy = engineer_constraint_taxonomy_features(df)
    
    df_socio_b = engineer_social_friction_engine(df)
    
    # Directive: Call the BESS Trojan Horse Detector
    df_bess_trojan = engineer_bess_trojan_horse_features(df)

    # Directive: Call the Grid Hierarchy Engine
    df_grid_hierarchy = engineer_grid_hierarchy_features(df)
    
    # Directive: Call the Rosetta Stone Engines (Correlation Analysis)
    df_thermal = engineer_thermal_physics_engine(df)
    df_named_entity = engineer_named_entity_gravity(df)
    df_lpa_profile = engineer_lpa_performance_profiling(df)
    df_logistics = engineer_logistical_friction_engine(df)
    
    # Concatenate all specialist feature blocks
    # Removed df_eco from the list
    return pd.concat([
        df_bmv, df_lpa, df_phys_pol, df_uncertainty, df_socio_a, 
        df_contract, df_taxonomy, df_socio_b, df_bess_trojan, df_grid_hierarchy,
        df_thermal, df_named_entity, df_lpa_profile, df_logistics
    ], axis=1)


def engineer_lpa_infrastructure_competence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The '[REDACTED_BY_SCRIPT]' Engine.
    Models the 'Competence Trap' where 'Fast' housing councils fail at infrastructure.
    
    Directives:
    1. Solar Maturity: Experience beats administrative speed.
    2. Overload Metric: Dynamic queue weight vs. capacity.
    3. Housing Competition: The '[REDACTED_BY_SCRIPT]' friction.
    4. Stratified Speed: Capacity-weighted decision timelines.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Robust getter helper
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val
    
    # Safe getters for critical columns
    exp = _get_safe('lpa_total_experience', 0)
    cap = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    queue_mw = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    avg_days = _get_safe('[REDACTED_BY_SCRIPT]', 180)
    density = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    
    # 1. Solar Maturity Score (Directive 1)
    # Log-scaled experience. High experience reduces the penalty of high capacity.
    # Feature: Experience * Capacity.
    maturity_log = np.log1p(exp)
    X['[REDACTED_BY_SCRIPT]'] = maturity_log * cap
    
    # 2. Resource Crunch / Overload Metric (Directive 2)
    # The impact of the queue depends on the LPA's "bandwidth" (experience).
    # A queue of 200MW kills a novice council (Exp=0), but is routine for an expert (Exp=50).
    X['[REDACTED_BY_SCRIPT]'] = queue_mw / (exp + 1)
    
    # 3. Housing Competition Flag (Directive 3)
    # "Fast" councils (Low days) in High Density areas (Housing pressure) struggling with Large Solar (Land competition).
    # Inverse decision days (Speed) * Density * Capacity.
    # High Speed + High Density + High Cap = High Friction (Competition).
    speed_proxy = 1 / (avg_days + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = speed_proxy * density * cap
    
    # 4. Stratified Speed Interaction (Directive 4 Proxy)
    # Enables the model to learn that "Fast" is good for Small, but bad/irrelevant for Large.
    # We provide the raw interaction surface.
    X['SIC_LPA_SPEED_VS_CAPACITY'] = avg_days * cap

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'SIC_LPA_SPEED_VS_CAPACITY']]



def engineer_physics_and_politics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Physics & Politics' Engine.
    Implements the Stratum-Specific Forensic Directives (2-5).
    
    Directives:
    2. Grid Complexity: Non-linear cable run friction.
    3. NIMBY Power: Property value * Density.
    4. Cumulative Switch: Precedent (Positive) vs Saturation (Negative).
    5. Legal Lockout: Agri-scheme + BMV constraints.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # --- Directive 2: The "Grid Complexity" Vector ---
    # The 'Cable Run' Killer. 
    # Logic: Distance is linear, but intersections are exponential friction points (Wayleaves).
    dist_km = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    # Prefer OH/DHV intersection counts if available, fall back to total
    intersections = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Interaction: dist * (1 + intersections^2)
    # We square intersections to model the exponential complexity of multiple wayleaves.
    X['SIC_GRID_CONNECTION_FRICTION'] = dist_km * (1 + (intersections ** 2))

    # --- Directive 3: The "NIMBY Power" Index ---
    # Money fights back.
    # Logic: High Property Value + High Density = Maximum Resistance.
    prop_val_idx = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    X['[REDACTED_BY_SCRIPT]'] = prop_val_idx * density

    # --- Directive 4: The "Cumulative Impact" Switch ---
    # Precedent vs. Saturation.
    # We use '[REDACTED_BY_SCRIPT]' as the proxy for nearby MW.
    nearby_mw = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # We create two features to allow the linear model (LightGBM split) to treat them differently.
    # Feature A: Precedent Benefit (Active for < 20MW neighbors). 
    # Represents the "Cluster" effect.
    X['[REDACTED_BY_SCRIPT]'] = nearby_mw.clip(upper=20)
    
    # Feature B: Saturation Penalty (Active for > 50MW neighbors).
    # Represents "Grid Full".
    X['[REDACTED_BY_SCRIPT]'] = (nearby_mw - 50).clip(lower=0)

    # --- Directive 5: The "Legal Lockout" Flag ---
    # The Binary Kill Switch for large sites.
    # Logic: Countryside Stewardship + BMV = Legal Nightmare.
    cs_bool = X.get('cs_on_site_bool', 0).fillna(0)
    bmv_bool = X.get('alc_is_bmv_at_site', 0).fillna(0)
    
    # Weighted sum. CS is a contract (Hard), BMV is policy (Soft/Hard).
    X['[REDACTED_BY_SCRIPT]'] = (cs_bool * 2) + bmv_bool
    
    # Interaction with Capacity (only matters for large sites)
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * cap

    return X[['SIC_GRID_CONNECTION_FRICTION', '[REDACTED_BY_SCRIPT]', 
              '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
              '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]

def engineer_uncertainty_paradox_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Uncertainty Paradox' Engine.
    Models the '[REDACTED_BY_SCRIPT]' where clean sites trigger fishing expeditions.
    
    Directives:
    1. Undefined Risk Penalty: Penalize clean sites > 10MW.
    2. Weighted Constraint Score: Heritage > Eco > Admin.
    3. Synergy Interaction: Heritage (Digging) vs Eco (Preservation) conflicts.
    """
    X = df.copy()
    
    # Safe getters
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    constraint_count = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # --- Directive 1: The "Undefined Risk" Penalty ---
    # The "Devil You Don't Know".
    # Logic: If Capacity > 10MW AND 0 Constraints -> High Risk of "Fishing Expedition".
    # We create a continuous penalty that activates above 10MW for clean sites.
    is_clean_site = (constraint_count == 0).astype(int)
    # Scale penalty by capacity above 10MW
    vacuum_intensity = is_clean_site * (cap - 10).clip(lower=0)
    X['[REDACTED_BY_SCRIPT]'] = vacuum_intensity

    # --- Directive 2: Constraint Severity Weighting ---
    # Not all constraints are equal.
    # We infer presence based on proximity (<500m) if boolean flags are missing.
    
    # Heritage (Visual/Digging) - Weight 5
    hp_dist = X.get('hp_dist_to_nearest_m', 5000).fillna(5000)
    nt_dist = X.get('nt_dist_to_nearest_m', 5000).fillna(5000)
    has_heritage = ((hp_dist < 500) | (nt_dist < 200)).astype(int)
    
    # Physical/Eco (Biology/Flooding) - Weight 3
    sssi_dist = X.get('sssi_dist_to_nearest_m', 5000).fillna(5000)
    aw_dist = X.get('aw_dist_to_nearest_m', 5000).fillna(5000)
    # Note: Flood zone data missing, using Eco proxies
    has_eco = ((sssi_dist < 500) | (aw_dist < 500)).astype(int)
    
    # Administrative/Policy - Weight 1
    is_bmv = X.get('alc_is_bmv_at_site', 0).fillna(0)
    
    X['[REDACTED_BY_SCRIPT]'] = (has_heritage * 5) + (has_eco * 3) + (is_bmv * 1)

    # --- Directive 3: The "Synergy" Interaction ---
    # The "Mitigation Deadlock".
    # Conflict: Archaeology/Heritage (Requires digging/visibility) vs Ecology (Requires preservation/screening).
    # If a site has BOTH, mitigation for one likely violates the other.
    X['SIC_CONFLICTING_MITIGATION_FLAG'] = has_heritage * has_eco

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'SIC_CONFLICTING_MITIGATION_FLAG']]


def engineer_sociological_friction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Sociological Friction' Engine.
    Models the '[REDACTED_BY_SCRIPT]' based on local demographics.
    
    Directives:
    1. NIMBY Potential: Time-Rich/Asset-Rich vs. Transient.
    2. Digital Mobilization: 'Digital Seniors' tail risk.
    3. Utilitarian Safe Harbor: The only viable host for >30MW.
    4. Parish Council Drag: 'Settled Offline' friction for small sites.
    """
    X = df.copy()
    
    # Safe getter helper to handle missing columns and NaN values robustly
    def _get_safe(col_name):
        if col_name in X.columns:
            return X[col_name].fillna(0)
        return 0.0

    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    
    # Safe getters for OAC groups using SANITIZED column names (underscores replaced spaces/hyphens)
    oac_digital_seniors = _get_safe('[REDACTED_BY_SCRIPT]')
    oac_settled_offline = _get_safe('[REDACTED_BY_SCRIPT]')
    oac_veterans = _get_safe('site_lsoa_oac_e_Veterans')
    oac_professionals = _get_safe('site_lsoa_oac_e_Professionals')
    oac_utilitarians = _get_safe('site_lsoa_oac_e_Rational_Utilitarians')
    
    # --- Directive 1: The "NIMBY Potential" Score ---
    # Resistance Energy = (Time Rich + Asset Rich) - (Transient/Apathetic)
    # Weighting based on "Free Time" and "Wealth".
    resistance_energy = (
        (oac_digital_seniors * 1.0) + 
        (oac_settled_offline * 0.8) + 
        (oac_veterans * 0.8) + 
        (oac_professionals * 0.6)
    )
    # Interaction: Energy * Capacity (Visual Impact Proxy)
    X['[REDACTED_BY_SCRIPT]'] = resistance_energy * cap

    # --- Directive 2: The "Digital Mobilization" Flag ---
    # The "Fat Tail" generator. High Internet Literacy + Home Ownership.
    # We use a threshold to flag domination by Digital Seniors.
    X['[REDACTED_BY_SCRIPT]'] = oac_digital_seniors

    # --- Directive 3: The "Utilitarian" Safe Harbor ---
    # The only viable host for >30MW.
    # Logic: If Cap > 25MW, the absence of Utilitarians is fatal.
    # We create a "Viability Score" that scales with capacity.
    # Score = Utilitarian % * (Capacity - 25). Active only for large sites.
    large_project_scaling = (cap - 25).clip(lower=0)
    X['[REDACTED_BY_SCRIPT]'] = oac_utilitarians * large_project_scaling

    # --- Directive 4: The "Parish Council" Drag Coefficient ---
    # "Settled Offline" friction for small sites (<10MW).
    # Bureaucratic drag (letters/meetings) rather than legal warfare.
    small_project_mask = (cap < 10).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = oac_settled_offline * small_project_mask

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
              '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]


def engineer_contractual_inertia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Contractual Inertia' Engine.
    Models the stratum-specific impact of Agri-Environment Schemes.
    
    Directives:
    1. Active Landowner Bonus: CS on small sites (<5MW) = Competence.
    2. Public Money Penalty: CS on large sites (>20MW) = Liability.
    3. Substitution Check: CS on Good Land vs Poor Land (BNG Potential).
    4. Entrenchment Proxy: Value as a proxy for contract depth.
    """
    X = df.copy()
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    cs_bool = X.get('cs_on_site_bool', 0).fillna(0)
    is_bmv = X.get('alc_is_bmv_at_site', 0).fillna(0)
    
    # --- Directive 1: The "Active Landowner" Bonus ---
    # Small scale (<5MW) + CS = Professional Farm Business -> Faster.
    # We use a negative sign convention for "Bonus" (reduction in friction), 
    # but feature engineering usually produces magnitude. The model learns the coefficient.
    # We produce an interaction term magnitude.
    small_scale_mask = (cap < 5).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = cs_bool * small_scale_mask

    # --- Directive 2: The "Public Money" Penalty ---
    # Large scale (>20MW) + CS = Breach of Contract -> Slower.
    # Scaled linearly by capacity to reflect the magnitude of the "clawback" negotiation.
    large_scale_mask = (cap > 20).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = cs_bool * large_scale_mask * cap

    # --- Directive 3: The "Substitution" Check (BNG Offset) ---
    # Interaction: CS * Land Quality.
    # Scenario A: CS on BMV (High Quality). Hard to argue "Solar is better". High Friction.
    X['SIC_CS_SUBSTITUTION_FRICTION_HIGH'] = cs_bool * is_bmv
    
    # Scenario B: CS on Non-BMV (Low Quality). Easier to argue "[REDACTED_BY_SCRIPT]". 
    # This acts as a dampener on the penalty.
    X['[REDACTED_BY_SCRIPT]'] = cs_bool * (1 - is_bmv)

    # --- Directive 4: The "Contract Duration" Proxy ---
    # We lack start dates, so we use Total Value as a proxy for "Scheme Entrenchment".
    # High Value = Deeper Contract = Harder to unwind.
    # Interaction with Large Scale only (Small scale value is negligible).
    cs_value = X.get('cs_on_site_total_value', 0).fillna(0)
    X['SIC_CS_SCHEME_ENTRENCHMENT'] = np.log1p(cs_value) * large_scale_mask

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
              'SIC_CS_SUBSTITUTION_FRICTION_HIGH', '[REDACTED_BY_SCRIPT]',
              'SIC_CS_SCHEME_ENTRENCHMENT']]





def engineer_social_friction_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Social Friction' Engine (SFP).
    Models the 'Silent Risk' of Action Groups and Judicial Reviews.
    
    Directives:
    1. Viewshed Proxy: Visual Exposure (Size * Eyes * Prominence).
    2. Affluence Multiplier: Wealth = Ability to object (Capacity * Wealth).
    3. Cumulative Fatigue: Exponential saturation penalty.
    4. Controversy Tier: Stratified risk injection.
    """
    X = df.copy()
    
    # Robust getter helper to prevent '[REDACTED_BY_SCRIPT]' crash
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val
    
    # Safe getters using robust helper
    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    density = _get_safe('[REDACTED_BY_SCRIPT]')
    gradient = _get_safe('[REDACTED_BY_SCRIPT]') # Proxy for prominence
    prop_val = _get_safe('[REDACTED_BY_SCRIPT]') # Proxy for Affluence/IMD
    legacy_count = _get_safe('[REDACTED_BY_SCRIPT]')
    
    # 1. Directive 1: The "Viewshed" Proxy (Geometric Friction)
    # Logic: High Visibility = High Capacity (Size) * High Density (Eyes) * High Gradient (Slope/Prominence).
    # A large farm on a hill near a town is the worst case.
    X['[REDACTED_BY_SCRIPT]'] = cap * density * (1 + gradient)

    # 2. Directive 2: The "Affluence" Multiplier (Demographic Friction)
    # Wealth is the fuel for legal obstruction.
    # Logic: Capacity * Property Value Index (Wealth Proxy).
    # Large projects in wealthy areas face "Professional" opposition.
    X['[REDACTED_BY_SCRIPT]'] = cap * prop_val

    # 3. Directive 3: The "Cumulative Fatigue" Saturation
    # "Enough is Enough" - Exponential decay of tolerance.
    # Logic: Existing Sites ^ 2.
    X['[REDACTED_BY_SCRIPT]'] = legacy_count ** 2

    # 4. Directive 4: Stratified Uncertainty Injection (Controversy Tier)
    # Categorical/Ordinal Tier based on Capacity and Exposure.
    # Tier 0: Small (<5MW).
    # Tier 1: Mid-sized, Low Exposure.
    # Tier 2: Large (>20MW) OR High Exposure.
    # Tier 3: Large (>20MW) AND High Exposure (The "War Zone").
    
    controversy_tier = pd.Series(0, index=X.index)
    
    # Define High Exposure Threshold (logical thresholds based on feature ranges)
    # Density > 0.5 (some people) and Gradient > 2 (some slope).
    high_exposure = (density > 0.5) & (gradient > 2)
    
    mid_cap = (cap >= 5) & (cap < 20)
    large_cap = (cap >= 20)
    
    controversy_tier[mid_cap] = 1
    controversy_tier[large_cap] = 2
    controversy_tier[large_cap & high_exposure] = 3
    
    X['SIC_SOCIO_CONTROVERSY_TIER'] = controversy_tier

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
              '[REDACTED_BY_SCRIPT]', 'SIC_SOCIO_CONTROVERSY_TIER']]


def engineer_national_congestion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: National Grid Queue Proxy.
    Models the 'National Backlog' by calculating the rolling volume of 
    concurrent applications in the system.
    """
    print("[REDACTED_BY_SCRIPT]")
    X = df.copy()
    
    # Ensure sorted by time for correct rolling calculation
    if 'submission_date' not in X.columns:
         raise ValueError("Critical Error: 'submission_date'[REDACTED_BY_SCRIPT]")
    
    X = X.sort_values('submission_date')
    
    # Robust check for capacity column
    if '[REDACTED_BY_SCRIPT]' not in X.columns:
         X['[REDACTED_BY_SCRIPT]'] = 0.0

    # Proxy: Rolling sum of capacity submitted in the last 365 days (National Momentum)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=365) 
    
    X['SIC_NAT_QUEUE_MOMENTUM_MW'] = X.rolling(
        window='365D', on='submission_date', closed='left'
    )['[REDACTED_BY_SCRIPT]'].sum().fillna(0)
    
    # Interaction: Project Scale * National Queue
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['SIC_NAT_QUEUE_MOMENTUM_MW']
    
    return X


def calculate_maturity_weights(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Architectural Mandate: Right-Censoring Correction (The 2024 Fix).
    Calculates sample weights to downweight recent data.
    """
    print("[REDACTED_BY_SCRIPT]")
    
    if 'submission_date' not in df.columns or target_col not in df.columns:
        print("[REDACTED_BY_SCRIPT]")
        return pd.Series(1.0, index=df.index)

    # 1. Reconstruct Decision Date
    decision_dates = df['submission_date'] + pd.to_timedelta(df[target_col], unit='D')
    
    # 2. Define Reference Date
    reference_date = decision_dates.max()
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 3. Calculate Age in Days
    age_days = (reference_date - decision_dates).dt.days
    
    # 4. Calculate Weight
    weights = (age_days / 365.0).clip(lower=0.1, upper=1.0)
    
    print(f"[REDACTED_BY_SCRIPT]")
    return weights


def engineer_constraint_taxonomy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Constraint Taxonomy' Engine.
    Models the distinction between 'Hard Kill' vs 'Soft Trap' constraints,
    replacing generic seasonality with species-specific bio-temporal logic.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Robust getter helper
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    # Safe getters using robust helper
    sub_month = _get_safe('submission_month', 6).astype(int)
    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    
    # --- Proximity & Habitat Flags (The Biological Reality) ---
    # Thresholds: 100m for Habitat relevance, 500m for SPA/Birds
    
    # 1. Amphibians (Newts) - Window: Mar(3) - Jun(6)
    wetland_near = (_get_safe('ph_dist_to_nearest_m', 5000) < 100) & \
                   ((_get_safe('[REDACTED_BY_SCRIPT]') == 1) | \
                    (_get_safe('[REDACTED_BY_SCRIPT]') == 1))
    
    # 2. Bats - Window: May(5) - Aug(8)
    woodland_near = (_get_safe('ph_dist_to_nearest_m', 5000) < 100) & \
                    (_get_safe('[REDACTED_BY_SCRIPT]') == 1)
    aw_near = (_get_safe('aw_dist_to_nearest_m', 5000) < 100)
    has_bats = (woodland_near | aw_near)
    
    # 3. Winter Birds - Window: Nov(11) - Feb(2)
    spa_near = (_get_safe('spa_dist_to_nearest_m', 5000) < 500)
    coastal_near = (_get_safe('ph_dist_to_nearest_m', 5000) < 500) & \
                   (_get_safe('[REDACTED_BY_SCRIPT]') == 1)
    has_winter_birds = (spa_near | coastal_near)
    
    # 4. Botany - Window: Apr(4) - Jul(7)
    sssi_near = (_get_safe('sssi_dist_to_nearest_m', 5000) < 100)
    botany_hab = (_get_safe('[REDACTED_BY_SCRIPT]') == 1) | \
                 (_get_safe('[REDACTED_BY_SCRIPT]') == 1)
    has_botany = sssi_near & botany_hab # SSSI usually implies the botany is protected

    # --- Directive 1: The "Species-Window" Matrix ---
    # Calculate months to wait for next valid survey window
    def calculate_wait(current_m, start_m, end_m):
        in_window = (current_m >= start_m) & (current_m <= end_m)
        if start_m > end_m: # Window spans year end (e.g. Nov-Feb)
             in_window = (current_m >= start_m) | (current_m <= end_m)
        wait = pd.Series(0, index=current_m.index)
        mask_before = (~in_window) & (current_m < start_m)
        wait[mask_before] = start_m - current_m[mask_before]
        mask_after = (~in_window) & (current_m > start_m) 
        wait[mask_after] = 12 - current_m[mask_after] + start_m
        return wait

    # Calculate wait for each constraint type
    wait_amphib = calculate_wait(sub_month, 3, 6) * wetland_near
    wait_bats = calculate_wait(sub_month, 5, 8) * has_bats
    wait_birds = calculate_wait(sub_month, 11, 2) * has_winter_birds
    wait_botany = calculate_wait(sub_month, 4, 7) * has_botany
    
    X['[REDACTED_BY_SCRIPT]'] = pd.concat([wait_amphib, wait_bats, wait_birds, wait_botany], axis=1).max(axis=1)

    # --- Directive 2: The "Competence" Split ---
    log_cap = np.log1p(cap) + epsilon
    X['SIC_ECO_COMPETENCE_INTERACTION'] = X['[REDACTED_BY_SCRIPT]'] * (1 / log_cap)

    # --- Directive 3: The "Scrutiny Multiplier" ---
    is_visible_season = sub_month.isin([5, 6, 7, 8])
    is_large_project = (cap > 30)
    X['[REDACTED_BY_SCRIPT]'] = (is_visible_season & is_large_project).astype(int)

    # --- Directive 4: The "[REDACTED_BY_SCRIPT]" Index ---
    has_summer_req = (wetland_near | has_bats | has_botany)
    has_winter_req = has_winter_birds
    X['[REDACTED_BY_SCRIPT]'] = (has_summer_req & has_winter_req).astype(int)

    # --- Legacy & Subjectivity Directives (Retained) ---
    np_near = (_get_safe('np_dist_to_nearest_m', 5000) < 100).astype(int)
    sac_near = (_get_safe('sac_dist_to_nearest_m', 5000) < 100).astype(int)
    aonb_near = (_get_safe('aonb_dist_to_nearest_m', 5000) < 100).astype(int)
    hp_near = (_get_safe('hp_dist_to_nearest_m', 5000) < 100).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = (
        (aonb_near * 10) + (hp_near * 10) + 
        (aw_near.astype(int) * 5) + (wetland_near.astype(int) * 5)
    )

    # Encirclement Proxy
    aw_count = _get_safe('aw_count_in_2km')
    aw_dist = _get_safe('aw_dist_to_nearest_m', 5000)
    X['[REDACTED_BY_SCRIPT]'] = aw_count / (aw_dist + 1e-6)

    # Lethal Constraint Flag (>10MW + Hard Constraint)
    lethal_set = (sssi_near | sac_near | np_near)
    X['[REDACTED_BY_SCRIPT]'] = ((cap > 10) & lethal_set).astype(int)

    # --- Final Directive: The <5MW Exemption ---
    small_proj_mask = (cap < 5)
    cols_to_zero = [
        '[REDACTED_BY_SCRIPT]', 
        'SIC_ECO_COMPETENCE_INTERACTION', 
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]'
    ]
    X.loc[small_proj_mask, cols_to_zero] = 0

    return X[['[REDACTED_BY_SCRIPT]', 'SIC_ECO_COMPETENCE_INTERACTION',
              '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
              '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
              '[REDACTED_BY_SCRIPT]']]


def engineer_bess_trojan_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'BESS Trojan Horse' Engine.
    Detects hidden 'Solar + Storage' configurations.
    """
    X = df.copy()
    epsilon = 1e-6
    
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    density = _get_safe('[REDACTED_BY_SCRIPT]')
    loading = _get_safe('[REDACTED_BY_SCRIPT]')
    
    # --- Directive 1: Lexical & Inferred BESS Extraction ---
    has_bess = pd.Series(0, index=X.index)
    
    if '[REDACTED_BY_SCRIPT]' in X.columns:
        keywords = r'[REDACTED_BY_SCRIPT]'
        has_bess_nlp = X['[REDACTED_BY_SCRIPT]'].str.contains(keywords, case=False, na=False).astype(int)
        has_bess = has_bess | has_bess_nlp
    
    storage_density = _get_safe('[REDACTED_BY_SCRIPT]') + _get_safe('[REDACTED_BY_SCRIPT]')
    has_bess_inferred = ((storage_density > 0.5) & (cap > 5)).astype(int)
    
    X['SIC_INF_BESS_FLAG'] = (has_bess | has_bess_inferred).astype(int)

    # --- Directive 2: The "Fire Safety" Proximity ---
    X['SIC_INF_BESS_FIRE_SAFETY_RISK'] = X['SIC_INF_BESS_FLAG'] * cap * density

    # --- Directive 3: The "Grid Service" Classification ---
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_INF_BESS_FLAG'] * loading

    # --- Directive 4: The "Containerization" Visual Proxy ---
    is_protected = (
        (_get_safe('aonb_dist_to_nearest_m', 5000) < 2000) |
        (_get_safe('np_dist_to_nearest_m', 5000) < 2000) |
        (_get_safe('hp_dist_to_nearest_m', 5000) < 1000)
    ).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_INF_BESS_FLAG'] * is_protected

    return X[['SIC_INF_BESS_FLAG', 'SIC_INF_BESS_FIRE_SAFETY_RISK', 
              '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]


def engineer_grid_hierarchy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Grid Hierarchy' Engine.
    Models the bureaucratic and physical friction of grid connections.
    """
    X = df.copy()
    epsilon = 1e-6
    
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    volts = _get_safe('[REDACTED_BY_SCRIPT]', 11)
    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    
    # --- Directive 1: The "Bureaucratic Queue" Flag ---
    X['[REDACTED_BY_SCRIPT]'] = (volts >= 33).astype(int)

    # --- Directive 2: The "Impedance Mismatch" (Current Density) ---
    amps = (cap * 1000) / ((volts * 1.732) + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = amps

    # "Cable Melter" Score
    is_lv = (volts < 33)
    X['[REDACTED_BY_SCRIPT]'] = (amps / 400.0) * is_lv

    # --- Directive 3: The "Forced Upgrade" Trap ---
    X['[REDACTED_BY_SCRIPT]'] = ((cap < 5) & (volts >= 33)).astype(int)

    # --- Directive 4: The "Private Wire" Exception Filter ---
    X['[REDACTED_BY_SCRIPT]'] = ((cap > 20) & (volts < 33)).astype(int)

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
              '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
              '[REDACTED_BY_SCRIPT]']]


def engineer_thermal_physics_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Thermal Physics' Engine.
    Models the '[REDACTED_BY_SCRIPT]' insight.
    """
    X = df.copy()
    epsilon = 1e-6
    
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    winter_load = _get_safe('[REDACTED_BY_SCRIPT]')
    summer_load = _get_safe('[REDACTED_BY_SCRIPT]')
    flex_ratio = _get_safe('[REDACTED_BY_SCRIPT]')
    
    # 1. Thermal Headroom Criticality
    max_load = pd.concat([winter_load, summer_load], axis=1).max(axis=1)
    available_headroom_pct = (100 - max_load).clip(lower=0)
    X['[REDACTED_BY_SCRIPT]'] = cap / (available_headroom_pct + epsilon)

    # 2. The "DNOA" Signal (Active Network Management)
    X['[REDACTED_BY_SCRIPT]'] = (flex_ratio > 0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * cap

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]


def engineer_named_entity_gravity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Named Entity' Gravity Engine.
    Captures the PR/Political risk of specific high-profile "Plan Killers".
    """
    X = df.copy()
    
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    celebrity_keywords = [
        'Minsmere', 'New Forest', 'Thames Path', 'Lee Valley', 
        'South Downs', 'Peak District', 'Broads', 'Cotswold'
    ]
    
    celebrity_risk = pd.Series(0, index=X.index)
    
    for col in X.columns:
        if any(prefix in col for prefix in ['spa_', 'sssi_', 'np_', 'nt_', 'aonb_']):
            if any(keyword.lower() in col.lower() for keyword in celebrity_keywords):
                if 'dist' in col:
                    pass 
                elif X[col].nunique() <= 2: 
                    celebrity_risk = celebrity_risk | X[col]

    X['SIC_ENV_CELEBRITY_SITE_RISK'] = celebrity_risk.astype(int)
    
    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_ENV_CELEBRITY_SITE_RISK'] * cap

    return X[['SIC_ENV_CELEBRITY_SITE_RISK', '[REDACTED_BY_SCRIPT]']]


def engineer_lpa_performance_profiling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The 'Judge & Jury' Engine.
    Operationalizes the 'LPA Profiling' directive.
    """
    X = df.copy()
    
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    avg_days = _get_safe('[REDACTED_BY_SCRIPT]', 180)
    approval_rate = _get_safe('lpa_major_commercial_approval_rate', 0.8)
    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    
    # 1. Local Authority Efficiency Index (The "Velocity" Signal)
    X['SIC_LPA_EFFICIENCY_INDEX'] = avg_days
    
    # 2. Efficiency Impact (The "Gridlock" Penalty)
    X['[REDACTED_BY_SCRIPT]'] = avg_days * cap
    
    # 3. Political Resistance Proxy (The "Rejection" Signal)
    X['SIC_LPA_POLITICAL_RESISTANCE'] = (1 - approval_rate) * cap

    return X[['SIC_LPA_EFFICIENCY_INDEX', '[REDACTED_BY_SCRIPT]', 'SIC_LPA_POLITICAL_RESISTANCE']]


def engineer_logistical_friction_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate: The '[REDACTED_BY_SCRIPT]' Engine.
    Operationalizes the 'Logistics' directive for large sites.
    """
    X = df.copy()
    epsilon = 1e-6
    
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    road_len = _get_safe('[REDACTED_BY_SCRIPT]')
    
    # 1. Logistical Viability Score (Friction)
    X['SIC_LOGISTICAL_FRICTION'] = cap / (road_len + epsilon)
    
    # 2. Construction Scale Threshold
    is_heavy_plant = (cap > 10).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_LOGISTICAL_FRICTION'] * is_heavy_plant

    return X[['SIC_LOGISTICAL_FRICTION', '[REDACTED_BY_SCRIPT]']]


def engineer_gm_lpa_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 1: Deconstruct the 'LPA Speed' Feature.
    Replaces the blunt 'Speed Tier' with specific instability and overload metrics.
    """
    X = df.copy()
    epsilon = 1e-6

    # 1. LPA Variance Index (The "Instability" Signal)
    # Diagnosis: A "Fast" council with high variance is a "Risky" council.
    # We use the explicit variance metric calculated during metadata aggregation.
    X['[REDACTED_BY_SCRIPT]'] = X.get('[REDACTED_BY_SCRIPT]', 0)

    # 2. LPA Overload Factor (The "Collapse" Risk)
    # Diagnosis: Fast councils collapse under weight.
    # Metric: Queue Depth (Open Apps) / Clearance Rate (Speed).
    # We use the engineered queue count (SIC_LPA_QUEUE_COUNT) if available, 
    # falling back to workload trend proxies if not.
    queue_depth = X.get('SIC_LPA_QUEUE_COUNT', X.get('[REDACTED_BY_SCRIPT]', 0) * 0.5)
    
    # Clearance Rate Proxy: 365 / Avg Decision Days (Apps per year "slot")
    avg_days = X.get('[REDACTED_BY_SCRIPT]', 180).replace(0, 180)
    clearance_capacity = 365.0 / avg_days
    
    X['[REDACTED_BY_SCRIPT]'] = queue_depth / (clearance_capacity + epsilon)

    # 3. The "False Efficiency" Trap
    # Interaction: Fast Avg Speed * High Variance. 
    # This specifically flags the "Speed Trap" councils.
    is_fast_statistically = (avg_days < 112).astype(int) # < 16 weeks
    X['[REDACTED_BY_SCRIPT]'] = is_fast_statistically * X['[REDACTED_BY_SCRIPT]']

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]


def engineer_gm_bmv_survivorship_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2: The 'Soil Relevance' Gate.
    Explicitly models the 'Survivorship Bias' where small BMV projects are fast (Permitted Development behavior)
    but large BMV projects face a '[REDACTED_BY_SCRIPT]'.
    """
    X = df.copy()
    
    # Safe getters
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_bmv = X.get('alc_is_bmv_at_site', 0).fillna(0)

    # 1. The "Small Scale Shield" (Survivorship Bias)
    # Logic: Is BMV AND Cap < 1MW. 
    # These are farmers powering barns. They pass quickly. We allow the model to see this "Low Risk" gate.
    X['SIC_GM_BMV_SMALL_SCALE_SHIELD'] = (is_bmv * (cap < 1.0)).astype(int)

    # 2. The "[REDACTED_BY_SCRIPT]" (The Kill Switch)
    # Logic: Is BMV AND Cap > 5MW.
    # These effectively remove food production land. High scrutiny, high friction.
    # We scale this by capacity to indicate the magnitude of the loss.
    X['[REDACTED_BY_SCRIPT]'] = (is_bmv * (cap > 5.0)).astype(int) * cap

    # 3. The "Messy Land" Baseline
    # Explicitly flag Non-BMV land for large projects as the "Standard Path".
    X['[REDACTED_BY_SCRIPT]'] = ((1 - is_bmv) * (cap > 5.0)).astype(int)

    return X[['SIC_GM_BMV_SMALL_SCALE_SHIELD', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]


def engineer_gm_stratified_zoning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 1: Stratified Urban Character (10-15MW Update).
    1. Small (<5MW): Linear Residential Conflict.
    2. Med (5-10MW): Urban Cliff Edge (Risk).
    3. Large (10-15MW): Brownfield Loophole (Bonus).
       High Urban Fabric > 50% = Industrial Retrofit = Fast.
    """
    X = df.copy()
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    urban_pct = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Stratum Gates
    is_small = (cap < 5.0).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)

    # --- Stratum A: Linear Conflict (<5MW) ---
    industrial_pct = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    residential_density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    conflict_ratio = residential_density / (industrial_pct + 1.0)
    X['[REDACTED_BY_SCRIPT]'] = is_small * conflict_ratio * cap

    # --- Stratum B: Urban Cliff Edge (5-10MW) ---
    # Urban > 15% is a "Kill Switch" zone for ground-mount in this range.
    is_high_urban_med = (urban_pct > 15).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_med * is_high_urban_med

    # --- Stratum C: Brownfield Loophole (10-15MW) ---
    # Directive 1: High Urban (>50%) is "Industrial Retrofit".
    # Logic: If Urban > 50% AND Cap > 10MW -> Bonus (Negative Friction).
    is_industrial_retrofit = (urban_pct > 50).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large * is_industrial_retrofit * -1.0

    # Universal Synergy
    X['[REDACTED_BY_SCRIPT]'] = industrial_pct

    return X[[
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]



def engineer_gm_wayleave_friction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2: The '[REDACTED_BY_SCRIPT]' Engine.
    Replaces linear distance with '[REDACTED_BY_SCRIPT]'.
    Models the probability of obstruction based on Landowner Fragmentation.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Safe getters
    dist_km = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    intersections = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # 1. Landowner Fragmentation Proxy
    # Diagnosis: 1km in a city crosses 20 backyards (High Legal Risk). 1km in a field crosses 1 farm (Low Legal Risk).
    # Logic: Base Fragmentation (1.0) + Density-driven Fragmentation.
    # We assume 'Settlement Density' correlates with '[REDACTED_BY_SCRIPT]'.
    fragmentation_factor = 1.0 + (density * 5.0) # Scale density to have impact
    
    # Feature: Estimated Title Counts (The "People Count" instead of Meter Count)
    X['[REDACTED_BY_SCRIPT]'] = dist_km * fragmentation_factor

    # 2. The "Wayleave Trap" Probability
    # Diagnosis: Crossing infrastructure (Roads/Rail) implies a sophisticated 3rd party (Network Rail/Highways Agency).
    # These are harder than private landowners.
    # Logic: Title Count + (Infrastructure Intersections * Hard Multiplier)
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] + (intersections * 3.0)

    # 3. The "Ransom Strip" Interaction
    # Diagnosis: High Risk + High Capacity = Ransom Scenario.
    # Large projects have deep pockets, encouraging landowners to hold out.
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['SIC_GM_RANSOM_STRIP_RISK'] = X['[REDACTED_BY_SCRIPT]'] * np.log1p(cap)

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'SIC_GM_RANSOM_STRIP_RISK']]



def engineer_gm_hard_soft_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 3: Stratified Constraint Negotiation.
    1. Small (1-5MW): Linear Risk.
    2. Med (5-10MW): Visual vs Technical Split.
    3. Large (10-15MW): BNG Hazard (Negotiable vs Binary).
    """
    X = df.copy()
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)

    def _is_near(prefix, thresh=100):
        return (X.get(f"[REDACTED_BY_SCRIPT]", 5000).fillna(5000) < thresh).astype(int)

    # --- 1. Small (1-5MW): Linear Risk (Retained) ---
    w_sum = (_is_near('sssi_') + _is_near('aw_') + _is_near('sac_') + 
             _is_near('aonb_') + _is_near('hp_')) * 5.0
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * w_sum

    # --- 2. Med (5-10MW): Visual vs Tech (Retained) ---
    has_greenbelt = (X.get('[REDACTED_BY_SCRIPT]', 0) > 0).astype(int)
    is_visual = (_is_near('aonb_') | _is_near('np_') | _is_near('hp_') | has_greenbelt)
    is_tech = (_is_near('sssi_') | _is_near('ph_') | _is_near('sac_'))
    
    X['[REDACTED_BY_SCRIPT]'] = is_med * is_visual * 10.0
    X['[REDACTED_BY_SCRIPT]'] = is_med * is_tech * 2.0

    # --- 3. Large (10-15MW): BNG Hazard (Directive 3) ---
    # A. Negotiable Habitats (Soft): Priority Habitat, Peat/Wetland.
    # Risk = High (S106 Negotiation Delay).
    is_negotiable = (_is_near('ph_')) # PH is the main BNG driver
    X['[REDACTED_BY_SCRIPT]'] = is_large * is_negotiable * 5.0
    
    # B. Binary Blockers (Hard): Ancient Woodland, SSSI.
    # Risk = Low (Standard 15m Buffer, Box Ticking).
    # We create a "Buffer Shield" feature.
    is_binary = (_is_near('aw_') | _is_near('sssi_'))
    X['[REDACTED_BY_SCRIPT]'] = is_large * is_binary * -2.0 # Negative friction (Fast)

    return X[[
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]



def engineer_gm_categorical_sanitation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 4: Sanitation of Categorical Data.
    Implements 'Rare Label Encoding' and 'Zombie Excision' logic.
    """
    X = df.copy()
    
    # 1. Technology Type Cleanup (The "Ghost Categories")
    # Diagnosis: Types 0 and 3 are statistical noise.
    # Logic: Map anything not Solar (21) or Wind (assume 10 for example) to 'Other_Legacy'.
    # Since we are in the GM Specialist which assumes Tech 21, we flag anomalies.
    if 'technology_type' in X.columns:
        # Create a binary "Valid Solar" flag rather than remapping the int column directly,
        # to preserve type consistency for downstream models.
        X['SIC_GM_DATA_VALIDITY_FLAG'] = X['technology_type'].apply(lambda x: 1 if x == 21 else 0)
    else:
        X['SIC_GM_DATA_VALIDITY_FLAG'] = 1 # Assume valid if column missing in specialist context

    return X[['SIC_GM_DATA_VALIDITY_FLAG']]


def engineer_gm_constraint_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 1: Stratified Constraint Severity & Rigor.
    1. Small (1-5MW): Linear Risk.
    2. Med (5-10MW): Prep Shield.
    3. Large (10-15MW): Application Rigor Proxy.
       0 Constraints = High Risk (Amateur). 2+ Constraints = Low Risk (Professional).
    """
    X = df.copy()
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med_utility = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large_ind = (cap >= 10.0).astype(int)

    # --- 1. Proximity & Weights ---
    def _is_proximate(col_prefix):
        dist_col = f"[REDACTED_BY_SCRIPT]"
        is_within_col = f"[REDACTED_BY_SCRIPT]"
        flag = X[is_within_col].fillna(0) if is_within_col in X.columns else 0
        if dist_col in X.columns:
            dist_flag = (X[dist_col] < 50).astype(int)
            return (flag | dist_flag).astype(int)
        return int(flag)

    w_sssi = _is_proximate('sssi_') * 10.0
    w_aw = _is_proximate('aw_') * 10.0
    w_sac = _is_proximate('sac_') * 10.0
    w_spa = _is_proximate('spa_') * 10.0
    w_np = _is_proximate('np_') * 8.0
    w_aonb = _is_proximate('aonb_') * 8.0
    w_hp = _is_proximate('hp_') * 5.0
    w_nt = _is_proximate('nt_') * 2.0
    w_crow = _is_proximate('crow_') * 2.0
    
    raw_score = (w_sssi + w_aw + w_sac + w_spa + w_np + w_aonb + w_hp + w_nt + w_crow)
    constraint_count = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)

    # --- 2. 1-5MW: Linear Risk + Ambiguity ---
    pct_agri = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_agri = (pct_agri > 50).astype(int)
    is_ambiguous_trap = ((raw_score == 0) & (is_agri == 1)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * (raw_score + (is_ambiguous_trap * 10.0))

    # --- 3. 5-10MW: Prep Shield ---
    is_high_prep_med = (constraint_count >= 2).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * is_high_prep_med * 1.0
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * (1 - is_high_prep_med)

    # --- 4. 5-15MW: Application Rigor (The Speculator Filter) ---
    # EXPANDED (v5.1): Now targets the 5-10MW "Med" stratum as well.
    # Logic: 0 Constraints on a >5MW site implies a "Desktop Study" only = High RFI Risk.
    
    # Define the "Speculator Band" (5-15MW)
    is_speculator_band = ((cap >= 5.0) & (cap < 15.0)).astype(int)
    
    is_zero_constraints = (constraint_count == 0).astype(int)
    is_heavy_constraints = (constraint_count >= 2).astype(int)
    
    # Feature: "Phantom Project" Risk (Penalty)
    # Penalizes >5MW sites that claim to have zero environmental impact (unlikely/lazy).
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * is_zero_constraints * 1.0
    
    # Feature: Professional Rigor Shield (Bonus)
    # Acknowledges that detailed constraint mapping implies a mature developer.
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * is_heavy_constraints * 1.0
    
    # Retain Non-BMV Complexity from previous directive (still valid context)
    is_bmv = X.get('alc_is_bmv_at_site', 0).fillna(0)
    is_non_bmv = (1 - is_bmv).astype(int)
    messy_score = _is_proximate('ph_') + _is_proximate('sssi_') + _is_proximate('aw_')
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_non_bmv * (messy_score + 1.0) * 10.0

    return X[[
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]



def engineer_gm_specialist_consolidated_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidated wrapper for the v4.0 GM Specialist features.
    Includes the previous 'Contextual Gate' directives AND the new 'Forensic' directives.
    """
    # 1. Previous Directives (Speed Trap, Soil Relevance, Urban Separation)
    df_variance = engineer_gm_lpa_variance_features(df)
    df_bmv = engineer_gm_bmv_survivorship_features(df)
    df_zoning = engineer_gm_stratified_zoning(df)
    
    # 2. New Forensic Directives (Wayleave Probability, Constraint Severity, Sanitation)
    df_wayleave = engineer_gm_wayleave_friction(df)
    df_severity = engineer_gm_constraint_severity(df)
    df_sanitation = engineer_gm_categorical_sanitation(df)
    
    # Concatenate all GM intelligence features
    return pd.concat([
        df_variance, df_bmv, df_zoning, 
        df_wayleave, df_severity, df_sanitation
    ], axis=1)

def refine_feature_selection_by_cohort(df: pd.DataFrame, cohort_type: str = 'GM') -> pd.DataFrame:
    """
    Architectural Directive 1: The 'Feature Slaughter'.
    Enforces the 'Parallel Universe' split by selectively dropping noise features.
    """
    X = df.copy()
    
    if cohort_type == 'Non-GM':
        # DIRECTIVE 1: HARD DROP for Non-GM (Rooftop)
        # These features confuse the model for rooftop projects.
        slaughter_list = [
            '[REDACTED_BY_SCRIPT]',
            'alc_grade_at_site', 'alc_is_bmv_at_site',
            'sssi_dist_to_nearest_m', 'aw_dist_to_nearest_m' # Ecological distance irrelevant for roof
            # Add others as identified in diagnosis
        ]
        # Drop only if they exist
        cols_to_drop = [c for c in slaughter_list if c in X.columns]
        X.drop(columns=cols_to_drop, inplace=True)
        
    elif cohort_type == 'GM':
        # For GM, we RETAIN the physical features but might drop 
        # features that are purely "building" related if any exist.
        pass
        
    return X

def engineer_gm_social_saturation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2: Stratified Social & Speculative Saturation.
    EXPANDED (v5.1): Adds 'Speculative Burst' detection for 5-15MW.
    """
    X = df.copy()
    
    neighbor_count = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    is_micro = (cap < 1.0).astype(int)
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med_utility = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large_ind = (cap >= 10.0).astype(int)
    
    # New: Define Speculator Band (5-15MW)
    is_speculator_band = ((cap >= 5.0) & (cap < 15.0)).astype(int)

    # --- 1. Micro (Tiers) ---
    conditions_micro = [(neighbor_count <= 1), (neighbor_count > 1) & (neighbor_count <= 3), (neighbor_count > 3)]
    X['[REDACTED_BY_SCRIPT]'] = is_micro * np.select(conditions_micro, [0, 1, 2], default=2)

    # --- 2. Small (Exponential) ---
    density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    exponential_score = (neighbor_count ** 1.5) * (density + 0.1)
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * exponential_score * cap

    # --- 3. Med (Zones) -> Renamed to Speculative Burst ---
    # High density of projects in the 5-15MW range typically means "Land Rush" behavior.
    # This leads to Grid Queue congestion (DNO delays) rather than just Social saturation.
    
    # If >5 neighbors in 5km, it's a "Hotspot".
    is_hotspot = (neighbor_count > 5).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * is_hotspot * neighbor_count

    # --- 4. Large (Cluster Shield) ---
    # Logic: Frontier (0-3) = High Risk. Clustered (4-10) = Shield. Saturated (>10) = Risk.
    is_frontier = (neighbor_count <= 3).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_frontier * 1.0
    
    is_clustered = ((neighbor_count > 3) & (neighbor_count <= 10)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_clustered * 1.0 
    
    is_saturated = (neighbor_count > 10).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_saturated * 1.0

    return X[[
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', # REPLACED: SIC_GM_SATURATION_ZONE_INTERACTION_MED
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_gm_grid_traffic_light(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2: The 'Grid Traffic Light' System.
    Discretizes continuous headroom years into 'Critical', 'Urgent', and 'Open' states.
    Handles the non-linear behavior of 'Negative Years' (Retroactive Constraint).
    """
    X = df.copy()
    
    # Safe getter for Years Until Constraint
    # '[REDACTED_BY_SCRIPT]' is the best proxy for "[REDACTED_BY_SCRIPT]"
    years_to_constraint = X.get('[REDACTED_BY_SCRIPT]', 5).fillna(5)
    
    # Logic: Status Codes
    # Critical (< 0): Deficit. "Statement of Works" territory.
    # Urgent (0-2): Imminent. Race condition.
    # Comfortable (2-5): Standard planning.
    # Open (> 5): No friction.
    
    conditions = [
        (years_to_constraint < 0),
        (years_to_constraint >= 0) & (years_to_constraint <= 2),
        (years_to_constraint > 2) & (years_to_constraint <= 5),
        (years_to_constraint > 5)
    ]
    # Codes: 3=Critical, 2=Urgent, 1=Comfortable, 0=Open
    choices = [3, 2, 1, 0]
    
    X['[REDACTED_BY_SCRIPT]'] = np.select(conditions, choices, default=1)
    
    # Interaction: Grid Status * Capacity
    # Trying to squeeze a 50MW project into an "Urgent" grid is harder than a 5MW one.
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * cap

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]


def engineer_gm_dno_friction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 3: The 'DNO Friction' Scalar.
    Normalizes duration based on the administrative efficiency of the DNO provider.
    Models the 'Infrastructure Tax' of difficult regions (e.g., NGED Wales vs UKPN).
    """
    X = df.copy()
    
    # 1. DNO Administrative Burden Scalar
    # Based on forensic analysis: NGED (Midlands/Wales) > UKPN (South East)
    # We construct the scalar from the One-Hot Encoded flags or source column.
    
    # Default burden
    burden = pd.Series(1.0, index=X.index)
    
    # Apply specific scalars if flags exist
    # NGED: High friction (Wales terrain, rural legacy) -> 1.2
    if 'dno_source_nged' in X.columns:
        burden[X['dno_source_nged'] == 1] = 1.2
        
    # UKPN: Low friction (Digitized, flexible) -> 0.8
    if 'dno_source_ukpn' in X.columns:
        burden[X['dno_source_ukpn'] == 1] = 0.8
        
    X['[REDACTED_BY_SCRIPT]'] = burden
    
    # 2. DNO x Grid Status Interaction
    # Hypothesis: A "Critical" grid status is handled differently by different DNOs.
    # High Burden * High Grid Status Code = Maximum Delay.
    grid_status = X.get('[REDACTED_BY_SCRIPT]', 0) # Assumes previous function run or default 0
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * grid_status

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]


def refine_feature_selection_non_gm_simplification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 4: The 'Non-GM' Simplification (Rooftop Immunity).
    Aggressively slaughters irrelevant geospatial features for Rooftop Solar.
    Retains only LPA Capacity, Building Context, and Heritage Constraints.
    """
    X = df.copy()
    
    # The "Keep" List (Concepts, not exact columns - we filter for matches)
    # 1. LPA Capacity: 'lpa_' prefix
    # 2. Building Type: 'technology_type' (already filtered mostly), 'mounting_type'
    # 3. Visual/Heritage Constraints: 'aonb', 'np' (National Park), 'hp' (Historic Park), 'conservation_area'
    
    keep_patterns = [
        'lpa_', 
        'technology_type', 'mounting_type', 
        'aonb_', 'np_', 'hp_', 'conservation_', 'listed_'
    ]
    
    # The "Kill" List (Concepts to purge)
    # Grid, Soil, Neighbors, Roads, Flood, Archaeology
    kill_patterns = [
        'dist_to_nearest_substation', 'headroom', 'voltage', 'fault_level', 'dno_', # Grid
        'alc_', 'soil_', # Soil
        'der_', 'legacy_', 'neighbor_', 'knn_', # Neighbors
        'road_', 'railway_', 'access_', # Logistics/Roads
        'flood_', 'archaeology_', 'sssi_', 'sac_', 'spa_' # Irrelevant Eco/Phys
    ]
    
    # Identify columns to drop
    cols_to_drop = []
    for col in X.columns:
        # If it matches a Kill pattern...
        if any(p in col for p in kill_patterns):
            # ...AND does not match a Keep pattern (safety check)
            if not any(k in col for k in keep_patterns):
                cols_to_drop.append(col)
                
    # Execution
    X_purged = X.drop(columns=cols_to_drop)
    
    # Hardening: Ensure we haven't stripped it bare.
    # We must retain '[REDACTED_BY_SCRIPT]' as a scalar even if it wasn't in keep_patterns explicitly
    if '[REDACTED_BY_SCRIPT]' not in X_purged.columns and '[REDACTED_BY_SCRIPT]' in X.columns:
        X_purged['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]']
        
    return X_purged


def engineer_gm_specialist_consolidated_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidated wrapper for the v4.1 GM Specialist features (Final Bake-Off).
    Integrates 'Social Saturation' and 'Grid Politics' into the pipeline.
    """
    # 1. Previous Layers (Variance, BMV, Zoning, Wayleaves, Severity)
    # Assuming previous functions are available in scope or imported
    df_prev = engineer_gm_specialist_consolidated_v2(df) 
    
    # 2. New Grid Politics & Saturation Layers
    df_saturation = engineer_gm_social_saturation(df)
    
    # Dependency: Grid Traffic Light needs to run before DNO Friction (for interaction)
    df_grid = engineer_gm_grid_traffic_light(df)
    
    # Merge grid features back into a temp df to allow DNO interaction to find them
    df_temp = pd.concat([df, df_grid], axis=1)
    df_dno = engineer_gm_dno_friction(df_temp)
    
    # 3. Concatenate all
    return pd.concat([
        df_prev, 
        df_saturation, 
        df_grid, 
        df_dno
    ], axis=1)


def engineer_gm_constraint_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2: The GM '[REDACTED_BY_SCRIPT]'.
    Moves from binary flags to a 'Deadliness Score' based on forensic weightings.
    Replaces Target Encoding with Domain-Hardened Weights to prevent leakage.
    """
    X = df.copy()
    
    # Robust Proximity Checker
    def _is_proximate(col_prefix):
        dist_col = f"[REDACTED_BY_SCRIPT]"
        is_within_col = f"[REDACTED_BY_SCRIPT]"
        if is_within_col in X.columns:
            flag = X[is_within_col].fillna(0)
        else:
            flag = 0
        if dist_col in X.columns:
            dist_flag = (X[dist_col] < 50).astype(int)
            return (flag | dist_flag).astype(int)
        return flag.astype(int)

    # 1. Apply Forensic Weights (The "Hierarchy of Denial")
    # Historic Parkland: The "Veto" (Weight 1.8)
    w_hp = _is_proximate('hp_') * 1.8
    
    # AONB: The "Major Headache" (Weight 1.4)
    w_aonb = _is_proximate('aonb_') * 1.4
    
    # Priority Habitat: "Mitigable" (Weight 1.1)
    w_ph = _is_proximate('ph_') * 1.1
    
    # SSSI: "Statutory Block" (Weight 1.6 - Inferred between AONB and Parkland)
    w_sssi = _is_proximate('sssi_') * 1.6
    
    # Nitrate/Generic Eco Zone (Weight 0.9)
    # Using 'sac_' (Special Area of Conservation) as proxy for high-level eco zones
    w_sac = _is_proximate('sac_') * 0.9

    # 2. Calculate Hierarchy Score
    # We take the MAXIMUM weight, not the sum. 
    # Logic: A project is killed by its deadliest constraint, not the sum of small ones.
    # (Unlike the "Severity Score" in v4.0 which summed them, this directive implies a hierarchy/ranking).
    
    # Construct a dataframe of weights to find the max
    weights_df = pd.DataFrame({
        'hp': w_hp, 'aonb': w_aonb, 'ph': w_ph, 'sssi': w_sssi, 'sac': w_sac
    })
    
    X['[REDACTED_BY_SCRIPT]'] = weights_df.max(axis=1)

    return X[['[REDACTED_BY_SCRIPT]']]


def engineer_gm_nimby_mobilization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2: Stratified NIMBY Mobilization.
    1. Micro: Sentiment.
    2. Small/Med: Community Resistance (Urban Interaction).
    3. Large (10-15MW): Retiree Resistance Weighting.
       Digital Seniors (3.0) vs Utilitarians (0.5).
       Interact with Visual Impact Index (Terrain).
    """
    X = df.copy()
    
    # Robust getter helper to prevent '[REDACTED_BY_SCRIPT]' crash
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val
    
    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    is_micro = (cap < 1.0).astype(int)
    is_small_med = ((cap >= 1.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)
    
    # Common Getters
    seniors = _get_safe('[REDACTED_BY_SCRIPT]')
    utilitarians = _get_safe('site_lsoa_oac_e_Rational_Utilitarians')
    veterans = _get_safe('site_lsoa_oac_e_Veterans')
    
    # --- 1. Micro: Sentiment ---
    wealth_idx = _get_safe('[REDACTED_BY_SCRIPT]')
    density = _get_safe('[REDACTED_BY_SCRIPT]')
    X['SIC_GM_NIMBY_SENTIMENT_MICRO'] = is_micro * (wealth_idx / (density + 0.1))

    # --- 2. Small/Med: Community Resistance (Base) ---
    # Weight: Seniors (3.0), Utilitarians (1.5), Veterans (1.0)
    mob_score_base = (seniors * 3.0) + (utilitarians * 1.5) + (veterans * 1.0)
    urban_fabric = _get_safe('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = is_small_med * mob_score_base * (urban_fabric + 1.0)

    # --- 3. Large: Retiree Resistance (Directive 2) ---
    # Weight Update: Utilitarians are now Heroic (0.5), Seniors are Villainous (3.0).
    # Logic: A flat, utilitarian area (0.5) speeds up the project.
    retiree_friction_score = (seniors * 3.0) + (utilitarians * 0.5) + (veterans * 1.0)
    
    # Interaction: Visual Impact Index (Gradient)
    # Seniors who can't see the farm don't object.
    gradient = _get_safe('[REDACTED_BY_SCRIPT]')
    visual_impact = (gradient + 1.0)
    
    X['SIC_GM_RETIREE_RESISTANCE_LARGE'] = is_large * retiree_friction_score * visual_impact

    return X[[
        'SIC_GM_NIMBY_SENTIMENT_MICRO', 
        '[REDACTED_BY_SCRIPT]',
        'SIC_GM_RETIREE_RESISTANCE_LARGE'
    ]]



def engineer_grid_queue_contention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2: The 'Queue Congestion' Engine (1-5MW).
    Graph 58 Analysis: Headroom is static, but the Queue is dynamic.
    Models the 'Race Condition' where High Headroom + High Activity = High Risk.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Safe getters
    headroom = X.get('[REDACTED_BY_SCRIPT]', 10).fillna(10)
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Activity Proxy:
    # In lieu of a live "Substation Queue" API, we use the engineered '[REDACTED_BY_SCRIPT]'
    # as the best proxy for regional development activity.
    active_queue_mw = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # 1. The "Race Condition" Ratio
    # Logic: Total Active MW / Available Headroom.
    # If Headroom is 20MW, but Queue is 40MW, the effective headroom is -20MW.
    X['SIC_GRID_CONTENTION_RATIO'] = (active_queue_mw + cap) / (headroom + epsilon)
    
    # 2. The "Headroom Illusion" Trap
    # Logic: If Headroom looks "Safe" (>10MW) BUT Contention Ratio > 1.0.
    # This flags cases where the model would naively predict "Fast" based on headroom alone.
    is_visually_safe = (headroom > 10).astype(int)
    is_actually_full = (X['SIC_GRID_CONTENTION_RATIO'] > 1.0).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = is_visually_safe * is_actually_full

    return X[['SIC_GRID_CONTENTION_RATIO', '[REDACTED_BY_SCRIPT]']]



def engineer_gm_stratified_grid_contention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 1: Stratified Grid Logic (10-15MW Update).
    1. Small (1-5MW): Queue Contention.
    2. Med (5-10MW): Green Lanes.
    3. Large (10-15MW): Headroom Paradox (ANM Shortcut).
       Negative Headroom = Fast. Middle Headroom = Slow.
    """
    X = df.copy()
    
    # Robust getter helper
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    headroom = _get_safe('[REDACTED_BY_SCRIPT]', 10)
    cap = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    years = _get_safe('[REDACTED_BY_SCRIPT]', 5.0)
    
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med_utility = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large_ind = (cap >= 10.0).astype(int)
    
    # 1. Global Traffic Light (Retained)
    conditions = [(headroom < 0), (headroom >= 0) & (headroom <= 10), (headroom > 10)]
    X['SIC_GM_GRID_TRAFFIC_LIGHT'] = np.select(conditions, [2, 1, 0], default=0)
    
    # 2. Small Ind: Queue Contention
    queue_count = _get_safe('SIC_LPA_QUEUE_COUNT', 0)
    is_race = ((headroom > 10) & (queue_count > 2)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * is_race * queue_count

    # 3. Med Utility: Deadline Pressure
    pressure_score = np.select([(years >= 0) & (years < 2.0), (years > 5.0)], [2.0, 0.5], default=1.0)
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * pressure_score
    
    dno_propensity = _get_safe('[REDACTED_BY_SCRIPT]', 0.5) 
    is_green_lane = (headroom > 50).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * is_green_lane * dno_propensity

    # 4. Large Ind: The "Headroom Paradox" (Directive 1)
    # A. ANM Curtailment (Fast): Headroom < 0
    is_anm = (headroom < 0).astype(int)
    
    # B. Unconstrained (Fast): Headroom > 50
    is_unconstrained = (headroom > 50).astype(int)
    
    # C. Study Risk (Slow): Headroom 0-20
    is_study_risk = ((headroom >= 0) & (headroom <= 20)).astype(int)
    
    # Combined "Contract Type" Score for Large Ind
    # 1.0 = Fast (ANM/Unconstrained). -1.0 = Slow (Study Risk). 0 = Neutral.
    contract_score = (is_anm * 1.0) + (is_unconstrained * 1.0) + (is_study_risk * -1.0)
    
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * contract_score

    # Asset Banking (Retained)
    is_banker = (years > 4.0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_banker * 1.0

    return X[[
        'SIC_GM_GRID_TRAFFIC_LIGHT', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]



def engineer_gm_regional_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 4: Regional Rejection Propensity.
    1. Global: Base Propensity Score.
    2. Small/Med (<10MW): High Impact (Regional Personality matters).
    3. Large (>=10MW): Deprioritized (National Frameworks override Region).
    """
    X = df.copy()
    
    # Base Propensity
    rejection_score = pd.Series(0.5, index=X.index)
    if 'dno_source_ukpn' in X.columns:
        rejection_score[X['dno_source_ukpn'] == 1] = 0.8
    if 'dno_source_nged' in X.columns:
        rejection_score[X['dno_source_nged'] == 1] = 0.3
        
    X['[REDACTED_BY_SCRIPT]'] = rejection_score
    
    # Stratified Impact
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_regional_scale = (cap < 10.0).astype(int)
    
    # Only calculate the interaction for <10MW. For >=10MW, this remains 0.
    X['[REDACTED_BY_SCRIPT]'] = is_regional_scale * X['[REDACTED_BY_SCRIPT]'] * cap

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]


def apply_nongm_stratified_protocol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 4: Stratified Non-GM Protocol (Visual Creep Update).
    1. Grid: Retained (Physics).
    2. Ecology: Conditional (Carports).
    3. Visual: Conditional (National Parks/AONB < 2km).
    """
    X = df.copy()
    
    # 1. Define "Keep" Concepts (Updated)
    # Added 'np_' and 'aonb_' to retention list for Visual Creep check
    retain_patterns = [
        'lpa_', 'technology_', 'mounting_', 'installed_capacity', 'submission_',
        'listed_', 'conservation_', 'SIC_LPA_', 'SIC_GM_', 'SIC_NONGM_',
        'dist_to_nearest_substation', 'headroom', 'voltage', 'dno_', # Grid
        'sssi_', 'sac_', 'spa_', 'ph_', 'aw_', # Ecology (Carport check)
        'np_', 'aonb_', 'resistance_ohm' # Visual (Creep check) & Physics
    ]
    
    # 2. Define "Kill" Concepts
    # Removed Soil/Logistics/Neighbors. Kept Archaeology/Flood.
    kill_patterns = [
        'alc_', 'soil_', 
        'der_', 'legacy_', 'neighbor_', 'knn_', 
        'road_', 'railway_', 'access_', 
        'flood_', 'archaeology_'
    ]
    
    cols_to_drop = []
    for col in X.columns:
        if any(p in col for p in kill_patterns):
            if not any(k in col for k in retain_patterns):
                cols_to_drop.append(col)
    
    X_purged = X.drop(columns=cols_to_drop)
    
    # 3. Apply "[REDACTED_BY_SCRIPT]" (Directive 4)
    # Logic: If Distance to NP/AONB > 2000m, Zero it out. 
    # Only close proximity matters for rooftops (Glint & Glare).
    
    visual_cols = [c for c in X_purged.columns if 'np_' in c or 'aonb_' in c]
    if visual_cols:
        # Check distances. If either is < 2000, keep. Else zero.
        dist_np = X_purged.get('np_dist_to_nearest_m', 5000).fillna(5000)
        dist_aonb = X_purged.get('aonb_dist_to_nearest_m', 5000).fillna(5000)
        
        is_far = (dist_np > 2000) & (dist_aonb > 2000)
        X_purged.loc[is_far, visual_cols] = 0

    # 4. Apply "Carport" Logic (Ecology)
    eco_prefixes = ['sssi_', 'sac_', 'spa_', 'ph_', 'aw_']
    eco_cols = [c for c in X_purged.columns if any(p in c for p in eco_prefixes)]
    
    if eco_cols and 'ph_dist_to_nearest_m' in X_purged.columns:
        is_pure_rooftop = (X_purged['ph_dist_to_nearest_m'].fillna(1000) > 50)
        X_purged.loc[is_pure_rooftop, eco_cols] = 0

    # Hardening
    if '[REDACTED_BY_SCRIPT]' not in X_purged.columns and '[REDACTED_BY_SCRIPT]' in X.columns:
        X_purged['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]']

    return X_purged


def engineer_gm_specialist_consolidated_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidated wrapper for the Final End-State GM Specialist.
    Combines v4.0 (Wayleaves) + v4.1 (Saturation) + v4.2 (Hierarchy/NIMBY/Traffic Light).
    """
    # 1. Load previous layers (Wayleaves, Saturation, DNO Friction)
    # Assuming previous function exists in scope
    df_prev = engineer_gm_specialist_consolidated_v3(df)
    
     # 2. New End-State Layers
    df_hierarchy = engineer_gm_constraint_hierarchy(df)
    df_nimby = engineer_gm_nimby_mobilization(df)
    
    # Note: Traffic Light v2 replaces v1 logic if overlapping, or adds to it.
   # We use v2 as the definitive "Headroom" metric, v1 (Status Code) was based on "Years".
    # Both are valid signals (Time vs Capacity). We keep both.
    df_grid_v2 = engineer_gm_stratified_grid_contention(df)
    
    # 3. Concatenate
    return pd.concat([
        df_prev,
        df_hierarchy,
        df_nimby,
        df_grid_v2
    ], axis=1)


def engineer_gm_ecological_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 1: Stratified Ecological Calendar.
    1. Micro (<1MW): Soft Seasonality.
    2. Med (5-10MW): Hard Lockout (Penalty).
    3. Large (10-15MW): Retrospective Bonus (Preparedness Paradox).
    """
    X = df.copy()
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    month = X.get('submission_month', 6).fillna(6).astype(int)
    
    is_micro = (cap < 1.0).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)
    
    # --- Constraints ---
    def _is_near(prefix):
        return (X.get(f"[REDACTED_BY_SCRIPT]", 5000).fillna(5000) < 100).astype(int)

    constraint_count = (_is_near('sssi_') + _is_near('sac_') + _is_near('spa_') + _is_near('aw_') + _is_near('ph_'))
    is_broad_winter = month.isin([9, 10, 11, 12, 1, 2]).astype(int)

    # 1. Micro: Soft Seasonality
    X['SIC_GM_MISSED_WINDOW_MICRO'] = is_micro * (constraint_count > 0) * is_broad_winter

    # 2. Med (5-10MW): Hard Lockout (Penalty)
    # Winter submission with constraints = Delay.
    X['[REDACTED_BY_SCRIPT]'] = is_med * constraint_count * is_broad_winter

    # 3. Large (10-15MW): Retrospective Bonus (Directive 1)
    # Winter submission with constraints = Competence/Pre-work.
    # Logic: High Constraint + Winter = Speed Bonus (Negative value).
    # We invert the logic of the Med stratum.
    X['[REDACTED_BY_SCRIPT]'] = is_large * constraint_count * is_broad_winter * -1.0

    return X[[
        'SIC_GM_MISSED_WINDOW_MICRO', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]



def engineer_gm_stratified_political_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 4: Political Epochs & Systemic Load.
    1. Epochs: Global era definition.
    2. Systemic Load: Post-2020 Grid Backlog (Permanent Shift).
    3. Elections: Proximity (Micro) vs Crossing (Ind).
    """
    X = df.copy()
    year = X.get('submission_year', 2020).fillna(2020).astype(int)
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # --- 1. Regulatory Epochs ---
    conditions = [
        (year <= 2015), (year == 2016), 
        (year >= 2017) & (year <= 2019),
        (year >= 2020) # Post-COVID / Net Zero Rush
    ]
    X['[REDACTED_BY_SCRIPT]'] = np.select(conditions, [0, 1, 2, 3], default=2)

    # --- 2. Systemic Load Factor (Directive 4) ---
    # Post-2020, the system is fundamentally slower due to grid backlog.
    # Active for all Industrial projects (>1MW).
    is_industrial = (cap >= 1.0).astype(int)
    is_post_2020 = (year >= 2020).astype(int)
    
    # This feature pushes the baseline duration up for recent projects.
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * is_post_2020 * 1.0
    
    # Election Paralysis (2024 specific)
    X['[REDACTED_BY_SCRIPT]'] = (cap >= 5.0).astype(int) * (year >= 2024).astype(int)

    # --- 3. Election Logic (Retained) ---
    if 'submission_date' not in X.columns:
        yr = X.get('submission_year', 2020)
        mo = X.get('submission_month', 6)
        temp_date = pd.to_datetime(dict(year=yr, month=mo, day=15))
    else:
        temp_date = pd.to_datetime(X['submission_date'])

    election_dates = [
        pd.Timestamp('2010-05-06'), pd.Timestamp('2015-05-07'),
        pd.Timestamp('2017-06-08'), pd.Timestamp('2019-12-12'),
        pd.Timestamp('2024-07-04')
    ]
    
    is_micro = (cap < 1.0).astype(int)
    
    # Micro: Proximity
    def get_days_to_next(date_val):
        for ed in election_dates:
            if ed > date_val: return (ed - date_val).days
        return 1000
    days_to_next = temp_date.apply(get_days_to_next)
    X['[REDACTED_BY_SCRIPT]'] = is_micro * (1.0 / (days_to_next + 30.0))

    # Industrial: Crossing
    est_duration = pd.Timedelta(days=250)
    end_date = temp_date + est_duration
    
    def check_crossing(row):
        return 1 if any(row['start'] < ed < row['end'] for ed in election_dates) else 0
        
    crossing_df = pd.DataFrame({'start': temp_date, 'end': end_date})
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * crossing_df.apply(check_crossing, axis=1)

    return X[[
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]'
    ]]



def refine_feature_selection_non_gm_sterilization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 3: The Non-GM 'Sterilization'.
    Removes all Time-Cyclic features from the Non-GM pipeline.
    Non-GM planning is an administrative factory line; seasonality is noise.
    """
    X = df.copy()
    
    # Identify Cyclic/Temporal Features to Drop
    # Note: We might keep 'submission_year' as a linear trend proxy (standards change),
    # but Directive says "[REDACTED_BY_SCRIPT]".
    # Specific targets: Month, Seasonality, Day, Election data.
    
    cyclic_patterns = [
        'submission_month', 'month_', 'season', 
        'submission_day', 'day_',
        'election_', 'purdah', 'political_clock'
    ]
    
    cols_to_drop = []
    for col in X.columns:
        if any(p in col.lower() for p in cyclic_patterns):
            cols_to_drop.append(col)
            
    X_purged = X.drop(columns=cols_to_drop)
    
    return X_purged


def engineer_gm_specialist_consolidated_v5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidated wrapper for the FINAL (v5.0) GM Specialist features.
    Integrates v4.0 (Wayleaves) + v4.1 (Saturation/Politics) + v4.2 (Hierarchy) + v5.0 (Temporal Physics).
    This is the definitive feature set for the GM Stratified Residual Model.
    """
    # 1. Load v4.2 Layers (Hierarchy, NIMBY, Traffic Light, Wayleaves, Saturation)
    # Assuming previous function exists in scope
    df_v4 = engineer_gm_specialist_consolidated_v4(df)
    
     # 2. New Temporal Physics Layers (Bio-Political Time)
    df_eco_cal = engineer_gm_ecological_calendar(df)
    df_pol_haz = engineer_gm_stratified_political_epochs(df)
    
    # 3. Concatenate
    return pd.concat([
        df_v4, 
        df_eco_cal, 
        df_pol_haz
    ], axis=1)



def engineer_gm_developer_competence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2 & 4: Stratified Competence & Configuration.
    1. Amperage Stress (All Ind).
    2. Med (5-10MW): Storage = Quality (Bonus).
    3. Large (10-15MW): Storage = Safety Drag (Penalty).
    4. Large: CHP = Industrial Bonus.
    """
    X = df.copy()
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    voltage = X.get('[REDACTED_BY_SCRIPT]', 11).fillna(11)
    
    is_industrial = (cap >= 1.0).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)

    # 1. Amperage Stress (Retained)
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * (cap / (voltage + 1e-6))

    # --- Storage Logic ---
    has_storage = X.get('SIC_INF_BESS_FLAG', 0)
    if 'SIC_INF_BESS_FLAG' not in X.columns:
        # Fallback detection
        s1 = X.get('[REDACTED_BY_SCRIPT]', 0)
        s2 = X.get('[REDACTED_BY_SCRIPT]', 0)
        stor_dens = s1 + s2
        
        if np.isscalar(stor_dens):
            has_storage = int(stor_dens > 0)
        else:
            has_storage = (stor_dens > 0).astype(int)

    # 2. Med (5-10MW): Institutional Quality (Bonus)
    X['[REDACTED_BY_SCRIPT]'] = is_med * has_storage

    # 3. Large (10-15MW): Fire Safety Drag (Penalty) (Directive 2)
    # "The Safety Tax".
    X['[REDACTED_BY_SCRIPT]'] = is_large * has_storage

    # 4. Large (10-15MW): CHP Bonus (Directive 4)
    # CHP implies Industrial Host (Greenhouse/Factory) = Fast.
    has_chp = X.get('chp_enabled', 0).fillna(0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_large * has_chp

    return X[[
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]



def engineer_stratified_grid_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 3: Stratified Grid Physics.
    1. All Ind: Stiffness Risk.
    2. Med (5-10MW): Physical Viability (Fault Levels).
    3. Large (10-15MW): Infrastructure Hug (Proximity Reward).
    """
    X = df.copy()
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_industrial = (cap >= 1.0).astype(int)
    is_med = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large = (cap >= 10.0).astype(int)

    # 1. Stiffness Risk (Retained)
    impedance = X.get('resistance_ohm', 1.0).fillna(1.0)
    if 'resistance_ohm' not in X.columns and '[REDACTED_BY_SCRIPT]' in X.columns:
        fl = X['[REDACTED_BY_SCRIPT]'].replace(0, 100)
        impedance = 1000.0 / fl
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * cap * impedance

    # 2. Med: Physical Viability (Retained)
    fault_level_ka = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    line_density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0) + \
                   X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = is_med * fault_level_ka * (line_density + 0.1)

    # 3. Large: Infrastructure Hug (Directive 3)
    # Logic: Close to Substation (<1km) OR Close to Constrained TX (<1km - ANM target).
    # Distance is key.
    dist_sub = X.get('[REDACTED_BY_SCRIPT]', 5.0).fillna(5.0)
    # If dnoa_dist exists (in meters), convert to km.
    dist_ctx = X.get('[REDACTED_BY_SCRIPT]', 5000.0).fillna(5000.0) / 1000.0
    
    # Minimum effective distance to "The Grid"
    eff_dist = np.minimum(dist_sub, dist_ctx)
    
    # Score:
    # <1km: Hug Bonus (High).
    # >3km: Wayleave Drag (Penalty).
    # We construct a score where 0 distance = Max Bonus.
    # Inverse distance weighting, clipped.
    hug_score = 1.0 / (eff_dist + 0.1) # Max ~10
    
    # Apply only if distance < 3km. If >3km, score becomes negligible or we apply explicit penalty.
    # For now, the inverse metric naturally penalizes distance.
    X['[REDACTED_BY_SCRIPT]'] = is_large * hug_score

    return X[[
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_gm_grid_black_holes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Stratified Grid Friction Engine.
    1. Global: 'Black Hole' GSP detection (applies to all).
    2. Industrial (1-5MW): 'Grid Gap' Impedance Risk (Specific Physics).
    """
    X = df.copy()
    
    # 1. Global Black Hole Detection (GSPs that are administratively broken)
    black_hole_risk = pd.Series(0, index=X.index)
    for col in X.columns:
        if '[REDACTED_BY_SCRIPT]' in col:
            black_hole_risk = black_hole_risk | X[col]
    
    X['[REDACTED_BY_SCRIPT]'] = black_hole_risk.astype(int)
    
    # 2. Global Interaction (Bad GSP + Bad Headroom = Death Zone)
    grid_status = X.get('SIC_GM_GRID_TRAFFIC_LIGHT', 0)
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * grid_status

    # 3. Stratified Grid Gap (1-5MW Specific)
    # Physics: >1MW on <33kV line = Voltage Rise Risk.
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    volts = X.get('[REDACTED_BY_SCRIPT]', 11).fillna(11)
    dist = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)

    # Gate: Must be Industrial Scale AND LV Network
    is_industrial = (cap >= 1.0).astype(int)
    is_lv = (volts < 33).astype(int)
    
    # Feature: Impedance Risk.
    # Scaled by Distance (Impedance) and Capacity (Current).
    # Only active for Industrial stratum. Micro stratum is immune (0).
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * is_lv * dist * cap

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]

def enforce_bess_stratum_gate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: BESS Stratification Gate.
    Ensures Battery signals are treated according to scale physics.
    """
    X = df.copy()
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Stratum Gates
    is_micro = (cap < 1.0)
    is_industrial = (cap >= 1.0) # 1-5MW+
    
    # Target Features
    bess_features = [
        'SIC_INF_BESS_FLAG', 
        'SIC_INF_BESS_FIRE_SAFETY_RISK', 
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        'SIC_BESS_CO_LOCATION_INTENSITY'
    ]
    
    existing_bess_cols = [c for c in bess_features if c in X.columns]
    
    if existing_bess_cols:
        # 1. Micro Stratum: Noise Suppression.
        # <1MW BESS is often internal/behind-the-meter and doesn't trigger major planning friction.
        X.loc[is_micro, existing_bess_cols] = 0
        
        # 2. Industrial Stratum: Complexity Multiplier.
        # >1MW BESS triggers "Hazardous Substance" consultations and specific Grid constraints.
        # If we have detected BESS Grid Complexity, we Amplify it for Industrial scale.
        if '[REDACTED_BY_SCRIPT]' in X.columns:
             X.loc[is_industrial, '[REDACTED_BY_SCRIPT]'] *= 1.5

    return X


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
    
    # UPDATED to V2: 12 High-fidelity features with Robust Getters
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
    # The logs showed 11 NaNs in SIC_GM_SATURATION_EXP_SMALL. 
    # We apply a global fillna(0) to safety-seal the matrix.
    df_final_sanitized.fillna(0, inplace=True)
    
    return df_final_sanitized

def engineer_gm_stratified_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive 2: Stratified Regime Engine (10-15MW Update).
    1. LPA Speed:
       - <5MW: Bonus (Efficiency).
       - 5-10MW: Trap (Collapse).
       - 10-15MW: Bonus (Legal Bully Effect).
    """
    X = df.copy()
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    avg_days = X.get('[REDACTED_BY_SCRIPT]', 180).fillna(180)
    
    # Stratum Gates
    is_micro = (cap < 1.0).astype(int)
    is_small_ind = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med_utility = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_large_ind = (cap >= 10.0).astype(int)

    # --- 1. LPA Competence (Tri-State Logic) ---
    is_fast_council = (avg_days < 112).astype(int) # < 16 Weeks
    speed_score = (112 - avg_days).clip(lower=0)
    variance = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)

    # State A: Efficiency Bonus (<5MW)
    X['[REDACTED_BY_SCRIPT]'] = (is_micro | is_small_ind) * is_fast_council * speed_score

    # State B: Complexity Collapse (5-10MW)
    # Fast councils break under 5-10MW complexity.
    X['[REDACTED_BY_SCRIPT]'] = is_med_utility * is_fast_council * speed_score * variance

    # State C: Legal Bully Resurrection (10-15MW)
    # Large developers use their own teams to utilize fast councils.
    # We remove the variance penalty and restore the speed bonus.
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_fast_council * speed_score

    # --- 2. Agricultural Justification (Directive 3: Site Complexity Proxy) ---
    is_bmv = X.get('alc_is_bmv_at_site', 0).fillna(0)
    
    # Micro: Survivor Shield
    X['[REDACTED_BY_SCRIPT]'] = is_micro * is_bmv
    
    # Ind/Med: Burden (>2MW, <10MW)
    # Standard policy friction.
    is_burdened_scale = ((cap > 2.0) & (cap < 10.0)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_burdened_scale * is_bmv * cap
    
    # Large (10-15MW): The "Policy Firewall" (Graph 2)
    # For Large projects, BMV is actually FAST (Pre-approved Exceptions).
    # Non-BMV is SLOW (Complex/Messy land).
    # We model this as a "Complexity Shield" for BMV at >10MW.
    X['[REDACTED_BY_SCRIPT]'] = is_large_ind * is_bmv * 10.0

    # --- 3. Grid Gap (1-5MW Specific) ---
    voltage = X.get('[REDACTED_BY_SCRIPT]', 11).fillna(11)
    is_lv = (voltage < 33).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_small_ind * is_lv * cap

    # --- 4. LPA Attention Deficit (5-15MW Specific) - NEW ---
    # Stratum Gates: 5-15MW
    is_mid_stratum = ((cap >= 5.0) & (cap < 15.0)).astype(int)
    
    # Logic: Increasing Workload Trend = Deprioritization of "Middle Child" projects.
    workload_trend = X.get('lpa_workload_trend', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = is_mid_stratum * workload_trend

    return X[[
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]



def engineer_nongm_specialist_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Stratified Non-GM Specialist Features (Physics Update).
    1. Grid Activation (Headroom).
    2. DNO Binary (UKPN).
    3. Grid Stiffness (Impedance) - Directive 3.
    """
    X = df.copy()
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    headroom = X.get('[REDACTED_BY_SCRIPT]', 10).fillna(10)
    is_industrial = (cap >= 1.0).astype(int)
    
    # 1. Grid Activation
    is_constrained = (cap > headroom).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * is_constrained
    X['SIC_NONGM_GRID_DEFICIT_INTENSITY_IND'] = is_industrial * is_constrained * (cap - headroom).clip(lower=0)

    # 2. DNO Binary Encoding
    is_ukpn = pd.Series(0, index=X.index)
    if 'dno_source_ukpn' in X.columns:
        is_ukpn = X['dno_source_ukpn']
    X['SIC_NONGM_IS_UKPN_IND'] = is_industrial * is_ukpn
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_NONGM_IS_UKPN_IND'] * is_constrained

    # 3. Grid Stiffness (Directive 3)
    # Logic: High Resistance = Slow Connection.
    impedance = X.get('resistance_ohm', 1.0).fillna(1.0)
    if 'resistance_ohm' not in X.columns and '[REDACTED_BY_SCRIPT]' in X.columns:
        fault_level = X['[REDACTED_BY_SCRIPT]'].replace(0, 100)
        impedance = 1000.0 / fault_level

    # Feature: Impedance Drag
    X['SIC_NONGM_GRID_STIFFNESS_FRICTION_IND'] = is_industrial * cap * impedance

    return X[['[REDACTED_BY_SCRIPT]', 'SIC_NONGM_GRID_DEFICIT_INTENSITY_IND', 
              'SIC_NONGM_IS_UKPN_IND', '[REDACTED_BY_SCRIPT]',
              'SIC_NONGM_GRID_STIFFNESS_FRICTION_IND']]




def engineer_gm_micro_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Micro-Stratum Context Engine.
    Models the 'Pony Paddock' friction for <5MW projects.
    Logic: Small Scale * Wealthy Neighbors * Urban Proximity = High Variance Friction.
    """
    X = df.copy()
    
    # Safe getters
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    wealth_idx = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    urban_density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Gate: Only active for Small/Micro (<5MW)
    is_small = (cap < 5.0).astype(int)
    
    # 1. The "Pony Paddock" Friction
    # High Wealth + High Urban Density = "[REDACTED_BY_SCRIPT]"
    # We dampen it by capacity because smaller = easier to hide, but the friction exists per unit MW.
    X['[REDACTED_BY_SCRIPT]'] = is_small * wealth_idx * urban_density
    
    # 2. The "Farmyard" Shield
    # Conversely, if it'[REDACTED_BY_SCRIPT]'s just a barn array.
    agri_pct = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = is_small * agri_pct * (1 - urban_density)

    return X[['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']]



def engineer_gm_small_scale_friction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Small Scale (1-5MW) Friction Engine.
    Targeting the 'Valley of Death' stratum (Quartile 1).
    1. Diversification Shield: Working farms vs Speculation.
    2. Parish Friction: Local density drag.
    3. Voltage Visibility: Low voltage clutter risk.
    """
    X = df.copy()
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Gate: Strictly 1-5MW (Quartile 1 Focus)
    is_target_stratum = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    
    # 1. Diversification Shield
    # Logic: High Agri (>50%) + Some Industrial (>1%) = Working Farm Diversification.
    # These are viewed as "Business Upgrades" rather than "Industrial Invasion".
    agri_pct = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    ind_pct = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    has_diversification = ((agri_pct > 50) & (ind_pct > 1.0)).astype(int)
    # Negative friction (Speed Bonus)
    X['[REDACTED_BY_SCRIPT]'] = is_target_stratum * has_diversification * -1.0 
    
    # 2. Parish Friction (Nosy Neighbors)
    # 1-5MW is the scale where Parish Councils are most lethal.
    # Settlement Density * Capacity.
    density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = is_target_stratum * density * cap
    
    # 3. Voltage Visibility (Clutter)
    # 1-5MW connects to distribution (11/33kV). High local density of these lines + New Project = "Visual Clutter".
    ohl_density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = is_target_stratum * ohl_density * cap

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_gm_grid_edge_viability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Grid Edge Viability Engine (1-15MW).
    EXPANDED (v5.1): Now covers the 'Speculator Valley' (5-15MW).
    1. Voltage Rise Risk (1-5MW): Physics constraint.
    2. Hierarchy Mismatch (5-15MW): The 11kV vs 33kV Trap.
    3. Capex Friction: Distance relative to revenue.
    """
    X = df.copy()
    epsilon = 1e-6
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    volts = X.get('[REDACTED_BY_SCRIPT]', 11).fillna(11)
    
    # Stratum Gates
    is_small = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_med_speculator = ((cap >= 5.0) & (cap < 15.0)).astype(int) # The Problem Zone
    
    # 1. Voltage Rise Risk (1-5MW)
    fault_level = X.get('[REDACTED_BY_SCRIPT]', 100).fillna(100)
    X['[REDACTED_BY_SCRIPT]'] = is_small * (cap / (fault_level + epsilon))
    
    # 2. Hierarchy Mismatch (5-15MW) - NEW
    # Connecting >5MW to <33kV is a massive delay risk (Statement of Works hell).
    # Connecting >5MW to >=33kV is "Professional" (Fast).
    is_lv_network = (volts < 33).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_med_speculator * is_lv_network * cap
    
    # 3. Capex Friction (The "Economic" Trap) - Applies to both
    # A 5km cable for 2MW is dead. A 10km cable for 10MW is marginal.
    dist = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = (is_small | is_med_speculator) * (dist / (cap + epsilon))

    # 4. Complex Connection Drag (5-15MW Specific) - NEW
    # Distance * Constraints. A clean 5km line is okay. A 5km line through an SSSI is a "Zombie" project.
    constraint_count = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = is_med_speculator * dist * (1 + constraint_count)

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_gm_speculator_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: The '[REDACTED_BY_SCRIPT]' Engine.
    Targets the high-variance 5-15MW stratum by identifying signatures of 
    'Land Banking', 'Priority Deficit', and 'Legal Traps'.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Safe getters
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Define the "Speculator Valley" (5-15MW)
    # We broaden slightly to 4-16MW to catch edge cases
    is_speculator_band = ((cap >= 4.0) & (cap < 16.0)).astype(int)
    
    # --- 1. The "EIA Threshold Trap" ---
    # Logic: 5-15MW projects are "Schedule 2". If they are near ANY sensitive area, 
    # they risk a full EIA (add 12 months).
    # Sensitive Areas: SSSI, AONB, NP, Scheduled Monuments (HP).
    def _is_near(prefix):
        return (X.get(f"[REDACTED_BY_SCRIPT]", 5000).fillna(5000) < 1000).astype(int) # 1km Impact Zone
    
    sensitive_zone = (_is_near('sssi_') | _is_near('aonb_') | _is_near('np_') | _is_near('hp_'))
    
    # Feature: The Trap. (Speculator Band * Sensitive Zone).
    # Professional projects mitigate this. Speculators get stuck in Screening/Scoping loops.
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * sensitive_zone * 10.0

    # --- 2. The "Cluster Runt" Index (Relative Priority) ---
    # Logic: Am I the smallest fish in the pond?
    # We use density metrics to infer neighbor scale.
    neighbor_mw_density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    neighbor_count = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Estimate average neighbor size (avoid div by zero)
    avg_neighbor_size = neighbor_mw_density / (neighbor_count + epsilon)
    
    # Ratio: My Size / Avg Neighbor Size.
    # If I am 10MW and avg is 40MW -> Ratio 0.25 (Runt).
    # If I am 10MW and avg is 5MW -> Ratio 2.0 (King).
    # We only care if we are the Runt (< 1.0).
    runt_ratio = (cap / (avg_neighbor_size + epsilon))
    is_runt = (runt_ratio < 0.5).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * is_runt * (1.0 - runt_ratio)

    # --- 3. The "Land Banking" Signature ---
    # Logic: Good Land (BMV) + Bad Grid (Far) = Speculation.
    # Real developers avoid BMV (Policy risk) and hug the grid (Cost).
    # Land Bankers want land that retains value if the solar fails (i.e., BMV).
    is_bmv = X.get('alc_is_bmv_at_site', 0).fillna(0)
    dist_grid = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Signature: BMV * Distance.
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * is_bmv * dist_grid

    # --- 4. LPA Speculator Fatigue ---
    # Logic: High Withdrawal Rate = LPA is actively killing speculative apps.
    withdrawal_rate = X.get('lpa_withdrawal_rate', 0).fillna(0)
    
    # Interaction: Withdrawal Rate * Speculator Band.
    # A 10MW app in a "High Withdrawal" LPA is almost certainly doomed to delay/death.
    X['[REDACTED_BY_SCRIPT]'] = is_speculator_band * withdrawal_rate

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_nsip_jurisdiction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: The 'NSIP Jurisdiction' Engine.
    Corrects the '[REDACTED_BY_SCRIPT]' for >50MW projects.
    Replaces LPA noise with Planning Inspectorate (PINS) statutory signals.
    FIX 1.1: Resolved AttributeError in _is_present helper.
    """
    X = df.copy()
    
    # Safe getters
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # The NSIP Gate: Strictly > 50MW
    is_nsip = (cap > 50.0).astype(int)
    
    # --- 1. The Statutory Baseline (The "Anchor") ---
    # NSIPs have a statutory max timeline: 
    # Acceptance (28d) + Pre-exam (~90d) + Exam (6m) + Report (3m) + Decision (3m)
    # Total ~ 15-18 months (~500 days).
    # This feature tells the model: "[REDACTED_BY_SCRIPT]".
    X['[REDACTED_BY_SCRIPT]'] = is_nsip * 500.0
    
    # --- 2. The "LPA Irrelevance" Filter ---
    # For NSIPs, the LPA is just a consultee. Their "Decision Speed" doesn't matter.
    # We create a feature to NEGATE LPA metrics for these rows.
    # Logic: High value = NSIP. Model can use this to coefficient-mask LPA features.
    X['SIC_NSIP_LPA_BYPASS_FLAG'] = is_nsip
    
    # --- 3. The "Examination Intensity" Proxy ---
    # NSIP delays happen during "Examination" (6 months).
    # Intensity is driven by the number of statutory consultees (Environment Agency, Natural England, Heritage).
    # We sum the constraints to proxy "Examination Friction".
    def _is_present(col):
        if col in X.columns:
            return (X[col].fillna(0) > 0).astype(int)
        return 0.0 # Safe scalar return if column missing
        
    consultee_friction = (
        _is_present('sssi_is_within') * 2.0 + # Natural England (Hard)
        _is_present('aonb_is_within') * 1.5 + # Natural England (Landscape)
        _is_present('hp_is_within') * 1.5 +   # Historic England
        _is_present('flood_zone_3') * 2.0     # Environment Agency (Hypothetical, usually critical)
    )
    
    # Multiplier: 30 days delay per friction unit
    X['[REDACTED_BY_SCRIPT]'] = is_nsip * consultee_friction * 30.0

    return X[[
        '[REDACTED_BY_SCRIPT]',
        'SIC_NSIP_LPA_BYPASS_FLAG', 
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_voltage_cliff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: The 'Voltage Cliff' Engine.
    Targets the 30-40MW stratum where thermal limits create high variance.
    """
    X = df.copy()
    epsilon = 1e-6
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    volts = X.get('[REDACTED_BY_SCRIPT]', 33).fillna(33) # Assume 33kV if unknown for this scale
    
    # --- 1. The "[REDACTED_BY_SCRIPT]" Band ---
    # Explicitly target the problem zone.
    is_cliff_band = ((cap >= 30.0) & (cap <= 40.0)).astype(int)
    
    # --- 2. The "Circuit Saturation" Probability ---
    # Logic: 35MW is the typical limit.
    # Feature: (Capacity - 30) / 10. 
    # At 30MW = 0.0 (Safe). At 40MW = 1.0 (Critical).
    saturation_curve = ((cap - 30.0) / 10.0).clip(0, 1)
    
    X['[REDACTED_BY_SCRIPT]'] = is_cliff_band * saturation_curve
    
    # --- 3. The "Upgrade Trigger" Interaction ---
    # If Grid is ALREADY constrained (Headroom < 10), putting a 35MW project on it
    # guarantees a "[REDACTED_BY_SCRIPT]" delay.
    headroom = X.get('[REDACTED_BY_SCRIPT]', 100).fillna(100)
    is_constrained = (headroom < 10.0).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = is_cliff_band * is_constrained * 100.0 #(Days penalty)

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_gm_threshold_evasion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: The 'Threshold Evasion' Engine.
    Targets the 40-50MW stratum to detect 'System Gaming'.
    1. Cap Hugging: 49.9MW flag.
    2. Salami Slicing: Cumulative neighbors > 50MW.
    3. Shadow NSIP: Evasion * Complexity.
    """
    X = df.copy()
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # --- 1. The "Cap Hugger" Flag ---
    # Explicitly identifies projects sitting just under the 50MW NSIP limit.
    # We define the "Danger Zone" as 45MW to 49.99MW.
    is_cap_hugger = ((cap >= 45.0) & (cap < 50.0)).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = is_cap_hugger

    # --- 2. The "Salami Slicing" Detector ---
    # "Salami Slicing" is splitting a 100MW project into two 49.9MW applications 
    # to bypass NSIP. This is legally fraught and triggers Judicial Review risk.
    # Logic: If My_Cap + Neighbor_Cap > 60MW (buffer), risk is high.
    neighbor_mw = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    cumulative_mw = cap + neighbor_mw
    
    is_slicing_risk = ((cap > 30.0) & (cumulative_mw > 60.0)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_slicing_risk * cumulative_mw

    # --- 3. The "Shadow NSIP" Complexity ---
    # If you evade the NSIP limit (Cap Hugger) AND you are in a complex area, 
    # the LPA will treat you as an NSIP anyway (Shadow Regime).
    # Complexity Proxy: SSSI + Ancient Woodland + AONB.
    def _is_present(col):
        if col in X.columns:
            return (X[col].fillna(0) > 0).astype(int)
        return 0
        
    complexity = _is_present('sssi_is_within') + _is_present('aw_is_within') + _is_present('aonb_is_within')
    X['[REDACTED_BY_SCRIPT]'] = is_cap_hugger * complexity * 10.0

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_gm_voltage_step_up(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: The 'Voltage Step-Up' Engine.
    Models the physical necessity of 132kV connections for >30MW projects.
    """
    X = df.copy()
    epsilon = 1e-6
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    voltage = X.get('[REDACTED_BY_SCRIPT]', 33).fillna(33)
    
    # Gate: High Capacity (>30MW)
    is_high_cap = (cap >= 30.0).astype(int)
    
    # --- 1. The "Physics Denial" Penalty ---
    # Trying to put >30MW into <33kV is a physics violation (Thermal Overload).
    # Trying to put >30MW into 33kV is a "Maybe" (High Variance).
    # Trying to put >30MW into 132kV is "Correct" (Low Variance).
    
    # Feature: Voltage Headroom Ratio (Voltage * X / Capacity)
    # 132kV / 30MW = 4.4 (Safe)
    # 33kV / 30MW = 1.1 (Tight)
    # 11kV / 30MW = 0.3 (Dead)
    # We simplify to a categorical Mismatch Score.
    
    mismatch_score = pd.Series(0.0, index=X.index)
    
    # Severe Mismatch: >30MW on <33kV
    mask_severe = (cap >= 30.0) & (voltage < 33)
    mismatch_score[mask_severe] = 10.0
    
    # Moderate Mismatch (The Variance Zone): >30MW on 33kV
    mask_mod = (cap >= 30.0) & (voltage == 33)
    mismatch_score[mask_mod] = 5.0
    
    # Safe: >30MW on >=132kV (Bonus)
    mask_safe = (cap >= 30.0) & (voltage >= 132)
    mismatch_score[mask_safe] = -2.0
    
    X['[REDACTED_BY_SCRIPT]'] = mismatch_score

    # --- 2. The "Upgrade Cost" Proxy ---
    # If Mismatch is High (5.0 or 10.0), distance matters exponentially (Cable Cost).
    # A 10km run at 33kV to fix a mismatch is financially fatal.
    dist = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    X['[REDACTED_BY_SCRIPT]'] = is_high_cap * mismatch_score * dist

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_global_administrative_rhythm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Global Administrative Rhythm.
    Captures the 'Pulse' of the bureaucracy and LPA domain expertise.
    1. Holiday Drag: December/August slowdowns.
    2. Solar Specialization: Does this LPA know Solar?
    3. Winter Fatigue: Submission late in the year.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Safe getters
    month = X.get('submission_month', 6).fillna(6).astype(int)
    legacy_count = X.get('nearby_legacy_count', 0).fillna(0)
    lpa_vol = X.get('[REDACTED_BY_SCRIPT]', 100).fillna(100)
    
    # --- 1. The "Holiday Drag" Coefficient ---
    # August (Summer Hols) and December (Xmas) are dead zones.
    # Logic: If month is 8 or 12, penalty.
    is_dead_zone = month.isin([8, 12]).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_dead_zone * 15.0 # Days penalty baseline
    
    # --- 2. The "[REDACTED_BY_SCRIPT]" Ratio ---
    # Does this LPA do a lot of energy projects relative to its size?
    # Proxy: Nearby Legacy Count (Solar history) / Current Workload.
    # High Ratio = Specialist (Efficient). Low Ratio = Novice (Inefficient/Fearful).
    # We invert it to make it a "Novice Penalty".
    specialization_ratio = legacy_count / (lpa_vol + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = 1.0 / (specialization_ratio + 0.1)

    # --- 3. The "Winter Fatigue" Accumulator ---
    # Applications submitted Q4 often drift into the next year's budget/cycle.
    is_q4 = month.isin([10, 11, 12]).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_q4 * 20.0 # Days penalty

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_micro_visual_agitation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Micro-Stratum Visual Agitation.
    Targets Stratum 0 (<1MW) and 1 (1-5MW).
    1. Hilltop Exposure: Gradient * Density.
    2. Village Clutter: Settlement * Poles.
    3. Garden Grabbing: Urban Fabric * Small Scale.
    """
    X = df.copy()
    
    # Robust getter helper to prevent '[REDACTED_BY_SCRIPT]' crash
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    cap = _get_safe('[REDACTED_BY_SCRIPT]')
    is_target = (cap < 5.0).astype(int)
    
    # --- 1. Hilltop Exposure (The "Eye Sore") ---
    # High Gradient (Slope) + High Settlement Density (Eyes) = Agitation.
    gradient = _get_safe('[REDACTED_BY_SCRIPT]')
    density = _get_safe('[REDACTED_BY_SCRIPT]')
    
    X['[REDACTED_BY_SCRIPT]'] = is_target * gradient * density * cap

    # --- 2. Village Clutter (The "Wires" Effect) ---
    # Small projects connect to LV/11kV poles. 
    # High Pole Density + High Settlement Density = "Too much clutter".
    poles = _get_safe('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = is_target * poles * density

    # --- 3. Garden Grabbing (Urban Infill Risk) ---
    # Small projects in high Urban Fabric areas look like "Garden Grabbing".
    urban = _get_safe('[REDACTED_BY_SCRIPT]')
    X['[REDACTED_BY_SCRIPT]'] = is_target * urban * cap

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_regional_hostility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Regional Hostility Engine.
    Models the 'Blue Wall' effect using legacy refusal data.
    1. Refusal Gravity: Proximity to nearest failure.
    2. Success Isolation: Distance to success vs failure.
    3. Hostile Territory Flag: High Refusal Proximity.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Safe getters
    dist_refusal = X.get('[REDACTED_BY_SCRIPT]', 50.0).fillna(50.0)
    dist_success = X.get('[REDACTED_BY_SCRIPT]', 50.0).fillna(50.0)
    
    # --- 1. Refusal Gravity (The "Cursed Earth" Metric) ---
    # Inverse distance to nearest refusal. 
    # Logic: High value = Standing on the grave of a failed project.
    X['[REDACTED_BY_SCRIPT]'] = 1.0 / (dist_refusal + 0.1)
    
    # --- 2. The "Hostility Ratio" ---
    # Logic: Is failure closer than success?
    # Ratio > 1.0 means Success is further away than Refusal (Bad).
    X['[REDACTED_BY_SCRIPT]'] = (dist_success + 0.1) / (dist_refusal + 0.1)
    
    # --- 3. The "No-Go Zone" Flag ---
    # If refusal is < 1km away, this is a highly contentious site.
    X['[REDACTED_BY_SCRIPT]'] = (dist_refusal < 1.0).astype(int)

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_grid_topology_depth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Grid Topology Depth.
    Models the disconnect between Physical and Electrical proximity.
    """
    X = df.copy()
    epsilon = 1e-6
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    dist_nearest = X.get('[REDACTED_BY_SCRIPT]', 5.0).fillna(5.0)
    dist_governing = X.get('[REDACTED_BY_SCRIPT]', 5.0).fillna(5.0)
    
    # --- 1. The "Grid Shadow" Delta ---
    # Logic: If Governing >>> Nearest, the local grid is weak/incompatible.
    # This implies a long cable run despite "apparent" grid proximity.
    shadow_delta = (dist_governing - dist_nearest).clip(lower=0)
    X['[REDACTED_BY_SCRIPT]'] = shadow_delta
    
    # --- 2. The "Connection Efficiency" Score ---
    # Ideal world: Governing == Nearest (Score 1.0).
    # Real world: Governing is 10x further (Score 0.1).
    X['SIC_GRID_CONNECTION_EFFICIENCY'] = (dist_nearest + 0.1) / (dist_governing + 0.1)
    
    # --- 3. Capacity-Weighted Isolation ---
    # High Capacity in a Deep Shadow = Financial Zombie.
    X['[REDACTED_BY_SCRIPT]'] = cap * shadow_delta

    return X[[
        '[REDACTED_BY_SCRIPT]',
        'SIC_GRID_CONNECTION_EFFICIENCY',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_logistics_access_failure_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Logistics Access Failure (V2).
    Models the 'Highways Objection' risk for large sites.
    """
    X = df.copy()
    epsilon = 1e-6
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    road_len = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Gate: Only matters for Industrial Scale (>5MW)
    is_industrial = (cap > 5.0).astype(int)
    
    # --- 1. The "Narrow Lane" Ratio ---
    # Capacity per km of Major Road.
    # High Cap / Low Road = High Traffic Friction.
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * (cap / (road_len + 0.1))
    
    # --- 2. The "Landlocked" Penalty ---
    # If Major Road Length is effectively zero (<100m) for a large project.
    is_landlocked = (road_len < 100).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_industrial * is_landlocked * cap
    
    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]
    
def engineer_social_fairness_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Social Fairness Ratio.
    Models the 'Proportionality' of the development relative to the host community.
    1. Occupation Ratio: Capacity / Settlement Density.
    2. Urban Service: Capacity * Urban Fabric (The "Good Citizen" bonus).
    """
    X = df.copy()
    epsilon = 1e-6
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    density = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0) # 0-1 score
    
    # --- 1. The "Occupation" Ratio ---
    # Logic: 50MW / 0.01 Density = 5000 (Massive Disproportion).
    # Logic: 5MW / 0.5 Density = 10 (Proportional).
    # High Score = "[REDACTED_BY_SCRIPT]".
    X['SIC_SOCIAL_OCCUPATION_RATIO'] = cap / (density + 0.01)
    
    # --- 2. The "Urban Service" Shield ---
    # If the site is actually IN an urban area (Urban Fabric > 20%), 
    # it is viewed as "Serving the City" rather than "[REDACTED_BY_SCRIPT]".
    urban_fabric = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_urban_context = (urban_fabric > 20.0).astype(int)
    
    # Bonus (Negative Friction) for matching scale to demand.
    X['[REDACTED_BY_SCRIPT]'] = is_urban_context * cap * -1.0

    return X[[
        'SIC_SOCIAL_OCCUPATION_RATIO',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_cable_route_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Cable Route Physics.
    Models the engineering friction of the connection path.
    1. Terrain Drag: Distance * Gradient.
    2. Water Crossing Risk: Distance * Wetland Proximity (Proxy for water tables).
    """
    X = df.copy()
    
    # Robust getter helper to prevent '[REDACTED_BY_SCRIPT]' crash
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return default_val

    dist = _get_safe('[REDACTED_BY_SCRIPT]', 5.0)
    gradient = _get_safe('[REDACTED_BY_SCRIPT]', 0)

    # --- 1. Terrain Drag ---
    # Logic: Distance * Gradient.
    # High value = Expensive, slow engineering (Wayleave disputes over route changes).
    X['[REDACTED_BY_SCRIPT]'] = dist * gradient

    # --- 2. Hydro-Engineering Risk ---
    # If the site is near a wetland AND has a long connection, 
    # the cable likely crosses high water tables or drainage ditches.
    wetland_col = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    
    # FIX: Handle scalar return from _get_safe when column is missing
    if hasattr(wetland_col, 'astype'):
        is_wetland_near = (wetland_col == 1).astype(int)
    else:
        is_wetland_near = int(wetland_col == 1)
        
    X['[REDACTED_BY_SCRIPT]'] = dist * is_wetland_near

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_agri_scarcity_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Agri-Scarcity Pressure.
    Models the 'Last Green Field' syndrome.
    Ratio of Site Agriculture to Regional Agriculture.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Site Context (Is the site a farm?)
    site_agri = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    # Regional Context (Is the region farming country?)
    region_agri = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # --- 1. The Scarcity Ratio ---
    # Logic: Site (100%) / Region (10%) = 10.0 (High Scarcity - "The Last Farm").
    # Logic: Site (100%) / Region (90%) = 1.1 (Abundance - "Just another field").
    X['SIC_AGRI_SCARCITY_RATIO'] = site_agri / (region_agri + epsilon)
    
    # --- 2. The "Food Security" Weaponization ---
    # High Scarcity * High Capacity * BMV.
    # The ultimate planning blocker.
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    is_bmv = X.get('alc_is_bmv_at_site', 0).fillna(0)
    
    X['[REDACTED_BY_SCRIPT]'] = X['SIC_AGRI_SCARCITY_RATIO'] * cap * is_bmv

    return X[[
        'SIC_AGRI_SCARCITY_RATIO',
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_seasonal_access_friction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Seasonal Access Friction.
    Models the 'Mud on Road' risk for rural construction.
    """
    X = df.copy()
    
    # Robust getter helper to prevent '[REDACTED_BY_SCRIPT]' crash
    # and ensure vector operations (like .isin) work on the result.
    def _get_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        # Return a Series of defaults matching the index to support vector ops
        return pd.Series(default_val, index=X.index)
    
    # Inputs
    month = _get_safe('submission_month', 6).astype(int)
    road_len = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    gradient = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    cap = _get_safe('[REDACTED_BY_SCRIPT]', 0)
    
    # --- 1. The "Winter Access" Trap ---
    # Submission in Autumn (9-11) means Decision in Winter/Spring (Mud Season).
    # Construction planning condition discharge becomes harder.
    is_autumn_sub = month.isin([9, 10, 11]).astype(int)
    
    # Low Road Density (Rural Lanes) + High Gradient (Runoff) + Autumn Sub = Nightmare.
    # Inverse Road Length (High = 0 friction).
    lane_factor = 1.0 / (road_len + 0.1)
    
    X['[REDACTED_BY_SCRIPT]'] = is_autumn_sub * lane_factor * gradient * cap

    return X[[
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_mid_stratum_rescue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Mid-Stratum Rescue (5-10MW).
    Targeted features to resolve the RMSE 97.8 failure.
    1. Voltage Stress: MW per kV.
    2. Line Tap Trap: Dist(Sub) vs Dist(OHL).
    3. Novice Fear: LPA Experience vs Scale.
    4. Swarm Density: Cumulative neighbors.
    """
    X = df.copy()
    epsilon = 1e-6
    
    cap = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    
    # Gate: Strictly 5-10MW
    is_target_stratum = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    
    # --- 1. Voltage Stress Ratio ---
    # 9MW on 11kV = 0.81 (Critical). 9MW on 33kV = 0.27 (Safe).
    volts = X.get('[REDACTED_BY_SCRIPT]', 11).fillna(11)
    X['[REDACTED_BY_SCRIPT]'] = is_target_stratum * (cap / (volts + epsilon))
    
    # --- 2. The "Line Tap" Trap ---
    # If Substation is far (>3km) but 33kV Line is close (<500m), they are likely 'Tapping'.
    # Tapping requires complex protection relays and outage windows = DELAY.
    dist_sub = X.get('[REDACTED_BY_SCRIPT]', 5.0).fillna(5.0)
    dist_ohl = X.get('[REDACTED_BY_SCRIPT]', 5000.0).fillna(5000.0) / 1000.0 # to km
    
    is_tapping_candidate = ((dist_sub > 3.0) & (dist_ohl < 0.5)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_target_stratum * is_tapping_candidate * 10.0 # Penalty
    
    # --- 3. Novice LPA Fear Factor ---
    # A novice LPA panics at 9MW. An experienced one processes it.
    exp = X.get('lpa_total_experience', 1).fillna(1)
    # High Cap + Low Exp = Fear.
    X['[REDACTED_BY_SCRIPT]'] = is_target_stratum * (cap / (exp + 1.0))
    
    # --- 4. The "Swarm" Cumulative Penalty ---
    # 5-10MW projects cluster. Neighbors amplify rejection risk.
    neighbors = X.get('[REDACTED_BY_SCRIPT]', 0).fillna(0)
    X['[REDACTED_BY_SCRIPT]'] = is_target_stratum * neighbors * cap

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_mid_stratum_rescue_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Mid-Stratum Rescue V2 (5-10MW).
    Resolves RMSE 97.8 failure with high-density feature injection.
    
    Target Stratum: 5-10MW (The 'Valley of Death').
    """
    X = df.copy()
    epsilon = 1e-6
    
    # 1. ROBUST GETTER FIX: Always return a Series
    def _get_safe_series(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return pd.Series(default_val, index=X.index)
    
    cap = _get_safe_series('[REDACTED_BY_SCRIPT]')
    
    # 2. STRATUM GATE: Strictly 5-10MW
    # All features outside this band MUST be 0.0 to prevent noise in other strata models
    is_mid = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    
    # --- A. GRID PHYSICS (The "Voltage Trap") ---
    # 9MW on 11kV is a nightmare. 9MW on 33kV is expensive.
    volts = _get_safe_series('[REDACTED_BY_SCRIPT]', 11)
    dist = _get_safe_series('[REDACTED_BY_SCRIPT]', 5.0)
    
    # Feature 1: Voltage Mismatch Ratio (Capacity / Voltage)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * (cap / (volts + epsilon))
    
    # Feature 2: Capex Drag (Distance / Capacity). 
    # Mid-scale projects cannot afford long cable runs.
    X['[REDACTED_BY_SCRIPT]'] = is_mid * (dist / (cap + epsilon))
    
    # Feature 3: The "T-Off" Probability.
    # Low distance to 33kV line + High distance to Substation = T-off Risk.
    ohl_dist = _get_safe_series('[REDACTED_BY_SCRIPT]', 5000) / 1000.0
    is_tapping = ((dist > 3.0) & (ohl_dist < 0.5)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * is_tapping * 10.0

    # --- B. LPA POLITICS (The "Committee Zone") ---
    # 5-10MW is too big for delegated powers, goes to Committee.
    lpa_exp = _get_safe_series('lpa_total_experience', 0)
    lpa_days = _get_safe_series('[REDACTED_BY_SCRIPT]', 180)
    
    # Feature 4: Novice Committee Risk. 
    # Inexperienced LPAs panic at >5MW.
    X['SIC_MID_LPA_NOVICE_PANIC'] = is_mid * (cap / (lpa_exp + 1.0))
    
    # Feature 5: The "Decision Drift" Amplifier.
    # Mid-scale projects drift disproportionately in slow councils.
    X['[REDACTED_BY_SCRIPT]'] = is_mid * lpa_days * (1.0 / (cap + 0.1))

    # --- C. LAND USE & SPECULATION (The "Land Banking" Check) ---
    is_bmv = _get_safe_series('alc_is_bmv_at_site', 0)
    neighbors = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    
    # Feature 6: The "Gold-Plate" Trap.
    # 5-10MW on BMV land is hard to justify (unlike <1MW barns or >50MW NSIP).
    X['[REDACTED_BY_SCRIPT]'] = is_mid * is_bmv * 5.0
    
    # Feature 7: Speculator Swarm.
    # High density of 5-10MW neighbors = Grid Queue Lockout.
    X['[REDACTED_BY_SCRIPT]'] = is_mid * neighbors
    
    # Feature 8: "Land Banking" Profile.
    # Good Land (BMV) + Bad Grid (Far) = Speculator.
    X['[REDACTED_BY_SCRIPT]'] = is_mid * is_bmv * dist

    # --- D. TECHNICAL & VISUAL (The "Clutter" Effect) ---
    density = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    poles = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    
    # Feature 9: Visual Agitation.
    # Mid-scale is visible but doesn't get "National Infrastructure" deference.
    X['[REDACTED_BY_SCRIPT]'] = is_mid * density * cap
    
    # Feature 10: Wire Clutter.
    # Connecting to 11/33kV adds more poles to an area already full of them.
    X['[REDACTED_BY_SCRIPT]'] = is_mid * poles * density

    # --- E. LOGISTICS (The "Lane Width" Check) ---
    road_len = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    
    # Feature 11: Construction Access Constriction.
    # 5-10MW requires cranes/HGVs but often sites on narrow lanes.
    X['SIC_MID_ROAD_ACCESS_CONSTRICTION'] = is_mid * (1.0 / (road_len + 0.1))

    # Feature 12: DNO Queue Proxy (New).
    # Using the engineered queue momentum if available.
    queue_mw = _get_safe_series('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_mid * (queue_mw / (cap + epsilon))

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'SIC_MID_LPA_NOVICE_PANIC',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'SIC_MID_ROAD_ACCESS_CONSTRICTION',
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_distribution_grit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Distribution Grit (V12.0).
    Targets the 0-10MW strata (Quartiles 0, 1, 2).
    Models the 'Messy Reality' of distribution-connected generation.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Robust Series Getter
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return pd.Series(default_val, index=X.index)
    
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    
    # Stratum Gates
    is_micro = (cap < 1.0).astype(int)
    is_small = ((cap >= 1.0) & (cap < 5.0)).astype(int)
    is_mid = ((cap >= 5.0) & (cap < 10.0)).astype(int)
    is_dist_connected = (cap < 15.0).astype(int) # Broad gate for all Dist connected
    
    # --- A. WAYLEAVE INTENSITY (The "Cable Tax") ---
    # Crossing roads/rails is expensive. Small projects can't amortize this cost.
    crossings = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 5.0)
    
    # Feature 1: Wayleave Density (Crossings per km).
    # High density = High legal friction per meter of cable.
    wayleave_density = crossings / (dist + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = is_dist_connected * wayleave_density
    
    # Feature 2: The "Ransom Strip" Multiplier (Small/Mid).
    # If Cap < 10MW AND Crossings > 5, the project is likely held to ransom.
    is_high_cross = (crossings > 5).astype(int)
    X['SIC_GRIT_RANSOM_STRIP_RISK'] = (is_small | is_mid) * is_high_cross * 10.0

    # --- B. COMMITTEE THRESHOLDS (The "Political Step") ---
    # Feature 3: The Delegated Powers Cliff.
    # Projects > 5MW (or 5 Hectares) usually go to Planning Committee (Slow/Risky).
    # Projects < 5MW often stay Delegated (Fast).
    # We flag the specific "Danger Zone" just above 5MW (5-7MW).
    is_cliff_edge = ((cap >= 5.0) & (cap < 7.0)).astype(int)
    lpa_vol = _get_series_safe('[REDACTED_BY_SCRIPT]', 100)
    X['SIC_GRIT_COMMITTEE_CLIFF_EDGE'] = is_cliff_edge * (1.0 / (lpa_vol + epsilon))

    # --- C. ANM & CURTAILMENT (The "Grid Veto") ---
    # Feature 4: Curtailment Exposure.
    # Small projects (1-10MW) are often forced into Active Network Management (ANM) zones first.
    # If Headroom < Cap, ANM is guaranteed.
    headroom = _get_series_safe('[REDACTED_BY_SCRIPT]', 10)
    is_constrained = (cap > headroom).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = (is_small | is_mid) * is_constrained * (cap - headroom)

    # --- D. MICRO-PHYSICS (<1MW) ---
    # Feature 5: Pole-Mount Viability.
    # If <500kW (0.5MW), can mount on pole. If >0.5MW, need ground pad (Expensive/Slow).
    # Logic: 0.5MW - 1.0MW is the "Awkward Micro" zone.
    is_awkward_micro = ((cap >= 0.5) & (cap < 1.0)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_awkward_micro * dist

    # --- E. URBAN FRINGE FRICTION ---
    # Feature 6: The "Dog Walker" Index.
    # Small/Mid projects on the edge of town (1-2km) face intense recreational opposition.
    settlement_dist = _get_series_safe('dist_to_nearest_settlement_m', 1000) # Assuming exists or proxied
    density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    
    # High Density + Mid Scale = High Friction
    X['[REDACTED_BY_SCRIPT]'] = is_mid * density * cap

    # --- F. AGRICULTURAL DISPLACEMENT ---
    # Feature 7: The "Best Field" Sacrifice.
    # For a 5MW farm, losing the "Best Field" (Grade 1/2) is a massive emotive issue for the farmer/community.
    # Unlike 50MW where it's a corporate transaction.
    is_grade_1_2 = (_get_series_safe('alc_grade_at_site', 3) <= 2).astype(int)
    X['SIC_GRIT_BEST_FIELD_SACRIFICE'] = (is_small | is_mid) * is_grade_1_2 * 5.0

    # --- G. DNO SPECIFICITY ---
    # Feature 8: DNO Engineering Fee Scale.
    # UKPN/WPD have specific breakpoints for engineering fees at 1MW and 5MW.
    # We model the "Fee Jump" friction.
    X['[REDACTED_BY_SCRIPT]'] = (is_cliff_edge | is_awkward_micro) * 5.0

    # --- H. VISUAL CONTAINMENT ---
    # Feature 9: The "Hidden Valley" Bonus.
    # Small projects can be completely hidden by topography. Large ones cannot.
    # High Gradient for Small/Mid = Bonus (Negative Friction).
    gradient = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = (is_small | is_mid) * gradient * -1.0

    # Feature 10: Cumulative Parish Fatigue.
    # 3 small projects in one parish = Revolt.
    neighbor_count = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small * neighbor_count

    return X[[
        '[REDACTED_BY_SCRIPT]',
        'SIC_GRIT_RANSOM_STRIP_RISK',
        'SIC_GRIT_COMMITTEE_CLIFF_EDGE',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        'SIC_GRIT_BEST_FIELD_SACRIFICE',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]

def engineer_grid_edge_inertia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Grid Edge & Social Inertia (V13.0).
    Targets 0-10MW.
    Models the physics of Secondary Substations and the sociology of 'Quiet Areas'.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Robust Series Getter (Standard Protocol)
    def _get_series_safe(col_name, default_val=0):
        if col_name in df.columns:
            return df[col_name].fillna(default_val)
        return pd.Series(default_val, index=df.index)
    
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    
    # Stratum Gates
    is_micro = (cap < 1.0).astype(int)
    is_small_mid = ((cap >= 1.0) & (cap < 10.0)).astype(int)
    is_dist_connected = (cap < 15.0).astype(int)
    
    # --- A. SECONDARY SUBSTATION PHYSICS (The "Flicker" Trap) ---
    # Projects <5MW often affect the local secondary network stability.
    
    # Feature 1: Utilisation Spike Risk.
    # Adding generation to a highly utilised secondary sub triggers reinforcement.
    sec_util = _get_series_safe('[REDACTED_BY_SCRIPT]', 50.0)
    X['[REDACTED_BY_SCRIPT]'] = (is_micro | is_small_mid) * (sec_util / 100.0) * cap
    
    # Feature 2: Voltage Complaint Density.
    # High customer density + New Generation = Voltage Flicker Complaints.
    # DNOs reject these to avoid regulatory fines (IIS).
    cust_density = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = (is_micro | is_small_mid) * cust_density * cap

    # --- B. THE IDNO LEGAL TRAP ---
    # Independent DNOs require tripartite legal agreements.
    # Fixed time/cost penalty that hurts small projects disproportionately.
    
    # Feature 3: IDNO Fixed Friction.
    is_idno = (
        _get_series_safe('idno_is_within', 0).astype(bool) | 
        _get_series_safe('[REDACTED_BY_SCRIPT]', 0).astype(bool)
    ).astype(int)
    # Inverse capacity weighting: The smaller the project, the bigger the pain.
    X['[REDACTED_BY_SCRIPT]'] = is_dist_connected * is_idno * (1.0 / (cap + 0.5))

    # --- C. TRANQUILITY RUPTURE (Social Inertia) ---
    # Projects in "Quiet" areas (Low Retail/Vice density) face "Change of Character" objections.
    # High Retail/Vice density = noisy/busy area = Less friction.
    
    # Feature 4: The "Quiet Village" Rupture.
    vice_idx = _get_series_safe('[REDACTED_BY_SCRIPT]', 1.0)
    # Inverse of noise index * Capacity.
    X['[REDACTED_BY_SCRIPT]'] = is_small_mid * (1.0 / (vice_idx + 0.1)) * cap

    # --- D. LTDS BLIGHT (Future Grid Shadow) ---
    # Long Term Development Statement (LTDS) shows future works.
    # Feature 5: The "Wait for Upgrade" Freeze.
    # If upgrade is close (<1km) and imminent (<2 years), DNO says "Wait".
    ltds_dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 5000.0)
    ltds_years = _get_series_safe('[REDACTED_BY_SCRIPT]', 5.0)
    
    is_blighted = ((ltds_dist < 1000.0) & (ltds_years > 0) & (ltds_years < 3)).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_dist_connected * is_blighted * ltds_years

    # --- E. PRIMARY SUBSTATION LOCKOUT ---
    # Feature 6: LCT Saturation Lockout.
    # If the Primary is already exporting net energy (Generation > Demand),
    # adding small solar is "[REDACTED_BY_SCRIPT]'s back".
    # Large projects pay for upgrades; small ones get refused.
    gen_dem_ratio = _get_series_safe('[REDACTED_BY_SCRIPT]', 0.5)
    is_saturated = (gen_dem_ratio > 1.0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = (is_micro | is_small_mid) * is_saturated * gen_dem_ratio

    # --- F. RESIDENTIAL RAT RUNS ---
    # Feature 7: The "School Run" Conflict.
    # Construction traffic on residential streets (High Customer Count / Low Road Class).
    # Assuming secondary sub customer count proxies for residential density on local loops.
    sec_customers = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_mid * sec_customers * cap

    # --- G. RURAL POVERTY LEVERAGE ---
    # Feature 8: Deprivation Opportunity (The "Renegade Farmer").
    # In deprived rural areas, landowners are more desperate/aggressive, pushing harder against LPA.
    # High Deprivation (Rank) = High Friction? No, often High Speed (Economic necessity).
    # We hypothesize High Deprivation + Small Scale = Faster (Less resistance, higher motivation).
    imd_rank = _get_series_safe('[REDACTED_BY_SCRIPT]', 15000) # Lower rank = More deprived
    # Normalize: 1 is most deprived.
    deprivation_score = 1.0 / (imd_rank + 1.0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_mid * deprivation_score * -1.0 # Bonus

    return X[[
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]


def engineer_hazards_margins_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Directive: Hazards & Margins (V14.0).
    Targets 0-10MW (Quartiles 0, 1).
    Models 'Technical Killers' (Glare, Flood) and 'Economic Zombies'.
    """
    X = df.copy()
    epsilon = 1e-6
    
    # Robust Series Getter
    def _get_series_safe(col_name, default_val=0.0):
        if col_name in X.columns:
            return X[col_name].fillna(default_val)
        return pd.Series(default_val, index=X.index)
    
    cap = _get_series_safe('[REDACTED_BY_SCRIPT]')
    
    # Stratum Gates
    is_micro = (cap < 1.0).astype(int)
    is_small_mid = ((cap >= 1.0) & (cap < 10.0)).astype(int)
    is_target_band = (cap < 10.0).astype(int)
    
    # --- A. THE GLINT & GLARE TRAP (Railways) ---
    # Network Rail objections are slow and technical.
    # Logic: High Rail Density near site = High Probability of Glare Assessment.
    rail_len = _get_series_safe('railway_length_1km', 0)
    # Scaled by capacity (larger surface area = more glare).
    X['SIC_HAZARD_RAIL_GLARE_RISK'] = is_target_band * rail_len * cap

    # --- B. HYDRO-TOPOGRAPHY (The "Soggy Bottom" Proxy) ---
    # Flat terrain + Wetland proximity = Flood Zone 2/3 (Sequential Test Risk).
    # Small developers often pick these "cheap" fields.
    gradient = _get_series_safe('[REDACTED_BY_SCRIPT]', 10)
    wetland_dist = _get_series_safe('ph_dist_to_nearest_m', 5000) # Check wetlands specifically if possible, else PH
    
    # "Flat" is low gradient. "Wet" is close proximity.
    is_flat = (gradient < 2.0).astype(int)
    is_wet = (wetland_dist < 200).astype(int)
    
    # The Trap: Flat & Wet.
    X['[REDACTED_BY_SCRIPT]'] = is_target_band * is_flat * is_wet * 5.0

    # --- C. ECONOMIC ZOMBIE RATIO (Viability) ---
    # A 1MW project cannot afford a 5km cable. It becomes a "Zombie" (stalled).
    # Logic: Distance / Capacity. 
    # High Dist / Low Cap = High Cost / Low Revenue = Zombie.
    dist = _get_series_safe('[REDACTED_BY_SCRIPT]', 0.1)
    # Ratio: km per MW.
    # 5km / 1MW = 5.0 (High Risk). 1km / 5MW = 0.2 (Low Risk).
    zombie_ratio = dist / (cap + 0.1)
    
    # Gate: Extreme ratios only (>2.0).
    is_zombie_candidate = (zombie_ratio > 2.0).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_target_band * is_zombie_candidate * zombie_ratio

    # --- D. THE "BEST LAND" TRAP (Policy) ---
    # Small farmers trying to put solar on Grade 1/2 land face "Sequential Test" policy blocks.
    # It's an amateur mistake. Professional developers avoid Grade 1/2.
    grade = _get_series_safe('alc_grade_at_site', 3)
    is_grade_1_2 = (grade <= 2).astype(int)
    
    X['[REDACTED_BY_SCRIPT]'] = is_target_band * is_grade_1_2 * 10.0

    # --- E. THE G99 PROTECTION CLIFF (<1MW) ---
    # <50kW (0.05MW) is G98/Simple G99. >50kW requires complex protection relays.
    # This feature flags the "Complexity Step" for Micro projects.
    is_above_g99 = (cap > 0.05).astype(int)
    X['[REDACTED_BY_SCRIPT]'] = is_micro * is_above_g99

    # --- F. VILLAGE ENCIRCLEMENT (Fragmentation) ---
    # Small projects often sit *between* villages. 
    # Logic: Settlement Density * Rail Density (Infrastructure + People).
    # Represents "Messy" peri-urban environments vs "Clean" open fields.
    settlement = _get_series_safe('[REDACTED_BY_SCRIPT]', 0)
    X['[REDACTED_BY_SCRIPT]'] = is_small_mid * settlement * (rail_len + 1.0)

    return X[[
        'SIC_HAZARD_RAIL_GLARE_RISK',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]'
    ]]































































if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR_V35, exist_ok=True)
    dossier_content = ["[REDACTED_BY_SCRIPT]"]
    
    print("[REDACTED_BY_SCRIPT]")

    # --- Setup: Load data and architectural definitions (v3.4 Protocol) ---
    X_general, y_general = phase_1_load_and_sanitize_data()
    
     # --- Isolate Ground Mount Solar Cohort for Specialization ---
    master_df = X_general.join(y_general)
    # Filter to only the desired technology types
    allowed_types = [21, 3, 26, 2, 6, 8]
    gm_solar_df = master_df[master_df['technology_type'].isin(allowed_types)]
    
    # REMEDIATION: Filter for farms submitted after 2021
    gm_solar_df = gm_solar_df[gm_solar_df['submission_year'] > 2021]
    print(f"[REDACTED_BY_SCRIPT]")
    #gm_solar_df = master_df

    # --- Mandate 19.0: Pre-Split Temporal Context Engineering ---
    # We must engineer temporal features (Political Clock) and Congestion Metrics (Queue Depth)
    # BEFORE splitting, while we have the full historical context.
    gm_solar_df, _, _, _ = engineer_temporal_context_features(gm_solar_df)
    gm_solar_df = engineer_lpa_congestion_metrics(gm_solar_df, target_col=y_general.name)
    
    # --- Mandate 20.0: BESS Variance Engineering ---
    # Injecting the Battery Co-Location features to capture the high-variance safety/grid regimes.
    gm_solar_df = engineer_bess_variance_features(gm_solar_df)
    
    # --- Mandate 21.0: Tower Paradox Engineering ---
    # Injecting features to decode the counter-intuitive "More Towers = More Delay" relationship.
    gm_solar_df = engineer_tower_paradox_features(gm_solar_df)
    
    print("[REDACTED_BY_SCRIPT]")

    # --- Mandate 22.0: Infrastructure Crossing Engineering ---
    # Injecting features to model the legal (wayleave) and physical friction of crossing roads/rail.
    gm_solar_df = engineer_infrastructure_crossing_features(gm_solar_df)
    
    print("[REDACTED_BY_SCRIPT]")

    # --- Mandate 23.0: Ecological Variance Engineering ---
    # Injecting features to capture high-variance delays from Priority Habitats, SACs, and Ancient Woodland.
    gm_solar_df = engineer_ecological_variance_features(gm_solar_df)
    
    print("[REDACTED_BY_SCRIPT]")

    # --- Mandate 24.0: Demographic Resistance Engineering ---
    # Injecting features to model NIMBY friction from specific OAC demographic groups ("e-Veterans", "Offline Communities").
    gm_solar_df = engineer_demographic_resistance_features(gm_solar_df)
    
    print("[REDACTED_BY_SCRIPT]")

    # --- Mandate 25.0: Physical Footprint Engineering ---
    # Injecting features to model the impact of Log-Scale Site Area and Power Density (Land Use Efficiency).
    gm_solar_df = engineer_physical_footprint_features(gm_solar_df)
    
    print("[REDACTED_BY_SCRIPT]")

    # --- Mandate 26.0: Brownfield Shield Engineering ---
    # Injecting features to model the variance-reducing effect of proximity to incumbents and industrial land.
    gm_solar_df = engineer_brownfield_shield_features(gm_solar_df)
    
    # --- Mandate 26.1: Stratified Ecological Drag Engineering (Forensic Remediation) ---
    # Injecting features to explicitly model the non-linear relationship between capacity and SSSI proximity.
    gm_solar_df = engineer_ecological_drag_features(gm_solar_df)
    
    print("[REDACTED_BY_SCRIPT]")

    # --- Mandate 27.0: Temporal Velocity Engineering ---
    gm_solar_df = engineer_temporal_velocity_features(gm_solar_df, target_col=y_general.name)

    # --- Mandate 28.0: National Grid Queue Proxy ---
    # Injecting the National Queue Momentum to explain systemic grid delays without using 'Year'.
    gm_solar_df = engineer_national_congestion_metrics(gm_solar_df)
    
    print("[REDACTED_BY_SCRIPT]")

    # --- ARCHITECTURAL INTERVENTION: Pre-emptive Excision of Unlearnable Samples ---
    # The previous post-hoc cleaning was flawed. This new protocol identifies and removes
    # unlearnable samples from the entire dataset *before* any splits or training,
    # ensuring a clean data foundation for the entire pipeline.
    
    # Smart Group Selection for Phase 0b
    if 'amaryllis_id' in gm_solar_df.columns and gm_solar_df['amaryllis_id'].nunique() > 1:
        groups_0b = gm_solar_df['amaryllis_id']
    elif 'lpa_name' in gm_solar_df.columns and gm_solar_df['lpa_name'].nunique() > 1:
        groups_0b = gm_solar_df['lpa_name']
    else:
        # Force fallback to KFold(shuffle=True) inside phase_0b by providing a single group
        groups_0b = pd.Series(0, index=gm_solar_df.index)

    unlearnable_indices = phase_0b_identify_and_excise_unlearnable_samples(
        gm_solar_df.drop(columns=[y_general.name]),
        gm_solar_df[y_general.name],
        groups=groups_0b,
        outlier_threshold=900
    )
    if not unlearnable_indices.empty:
        # Surgically remove from the canonical target file for future runs.
        y_canonical = pd.read_csv(Y_REG_PATH, index_col=0).squeeze("columns")
        original_count = y_canonical.notna().sum()
        y_canonical.loc[unlearnable_indices] = np.nan
        y_canonical.to_csv(Y_REG_PATH, header=True)
        final_count = y_canonical.notna().sum()
        print(f"[REDACTED_BY_SCRIPT]")

        # Surgically remove from the in-memory dataframe for the current run.
        gm_solar_df.drop(index=unlearnable_indices, inplace=True)
        print(f"[REDACTED_BY_SCRIPT]")

    # --- Execute 3-Way Split (Train/Validation/Test) to Prevent Meta-Level Leakage (Mandate 41.4) ---
    # First, split off the final, unimpeachable holdout test set (15%).
    #HOLDOUT
    gm_train_val_df, gm_test_df = train_test_split(
        gm_solar_df, test_size=0.1, random_state=67,
        stratify=pd.qcut(gm_solar_df['[REDACTED_BY_SCRIPT]'], q=4, labels=False, duplicates='drop')
    )
    

    # Next, split the remaining data into training (70%) and validation (15%).
    # The validation set is for tuning the final calibrator, not the models themselves.

    #TEST
    train_val_stratify_col = pd.qcut(gm_train_val_df['[REDACTED_BY_SCRIPT]'], q=4, labels=False, duplicates='drop')
    gm_train_df, gm_val_df = train_test_split(
        gm_train_val_df, test_size=0.05, 
        random_state=67, stratify=train_val_stratify_col
    )

    gm_val_df = gm_val_df[gm_val_df['[REDACTED_BY_SCRIPT]'] == 1]
    gm_val_df.reset_index(drop=True, inplace=True)

    
    X_train_base, y_train_gm = gm_train_df.drop(columns=y_general.name), gm_train_df[y_general.name]
    X_val_base, y_val_gm = gm_val_df.drop(columns=y_general.name), gm_val_df[y_general.name]
    X_test_base, y_test_gm = gm_test_df.drop(columns=y_general.name), gm_test_df[y_general.name]
    
    print(f"[REDACTED_BY_SCRIPT]")

    # --- 1. EXECUTE BAYESIAN SMOOTHING (Priority Injection) ---
    X_train_base, X_val_base, X_test_base = engineer_lpa_risk_features(
        X_train_base, y_train_gm, X_val_base, X_test_base, m_factor=10
    )

    # --- 2. EXECUTE FRICTION MATRIX ---
    X_train_base, X_val_base, X_test_base = engineer_friction_matrix_features(
        X_train_base, X_val_base, X_test_base
    )
    
    # --- 3. TEMPORAL & STRATEGIC ENGINEERING ---
    # Step A: Temporal
    X_train_temporal, baseline_year, tfi_imputer, tfi_scaler = engineer_temporal_context_features(X_train_base)
    X_val_temporal, _, _, _ = engineer_temporal_context_features(
        X_val_base, baseline_year=baseline_year, imputer=tfi_imputer, scaler=tfi_scaler
    )
    X_test_temporal, _, _, _ = engineer_temporal_context_features(
        X_test_base, baseline_year=baseline_year, imputer=tfi_imputer, scaler=tfi_scaler
    )
    
    # Step B: SICs
    X_train_sics = engineer_strategic_interaction_features(X_train_temporal)
    X_val_sics = engineer_strategic_interaction_features(X_val_temporal)
    X_test_sics = engineer_strategic_interaction_features(X_test_temporal)

    # --- 4. SAFE PURGE ---
    # We define specific columns to KEEP to prevent accidental deletion
    protected_cols = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'LPA_Chaos_Index',
        'LPA_Stability_Index', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'FI_Paranoia_Factor_SSSI', 
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'SIC_LPA_CHAOS_APPROVAL',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    
    # Verify existence before purge
    missing_protected = [c for c in protected_cols if c not in X_train_sics.columns]
    if missing_protected:
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print(f"[REDACTED_BY_SCRIPT]")

    # --- PRESERVE GROUPS FOR CV ---
    # We need lpa_name (and potentially amaryllis_id) for GroupKFold to prevent leakage.
    # We extract them before they are purged as 'toxic_cols'.
    print("[REDACTED_BY_SCRIPT]")
    
    def _extract_groups(df):
        groups = pd.DataFrame(index=df.index)
        if 'lpa_name' in df.columns:
            groups['lpa_name'] = df['lpa_name']
        if 'amaryllis_id' in df.columns:
            groups['amaryllis_id'] = df['amaryllis_id']
        # Fallback if neither exists (should not happen given previous logic)
        if groups.empty:
             groups['group_id'] = df.index 
        return groups

    X_train_groups = _extract_groups(X_train_sics)
    X_val_groups = _extract_groups(X_val_sics)
    X_test_groups = _extract_groups(X_test_sics)

    # Remove operational columns but keep numeric types
    toxic_cols = ['lpa_name', 'submission_date']
    
    X_train_sics = X_train_sics.drop(columns=[c for c in toxic_cols if c in X_train_sics.columns]).select_dtypes(include=[np.number])
    X_val_sics = X_val_sics.drop(columns=[c for c in toxic_cols if c in X_val_sics.columns]).select_dtypes(include=[np.number])
    X_test_sics = X_test_sics.drop(columns=[c for c in toxic_cols if c in X_test_sics.columns]).select_dtypes(include=[np.number])

    # Final Verification
    if '[REDACTED_BY_SCRIPT]' in X_train_sics.columns:
        print("[REDACTED_BY_SCRIPT]")
    else:
        raise ValueError("[REDACTED_BY_SCRIPT]")

    # --- Continue to Phase 0 (Oracle) ---

    # --- Tune & Train Oracle & Generate Context Vectors (POST-SPLIT to prevent leakage) ---
    # AD-AM-33: Oracle MUST NOT see the k-NN features. It is trained on the pre-kNN feature set.
    # AD-AM-45: Now uses Hybrid Compression (Raw Top-N + PCA Tail).
    
    # Define artifact path for compression metadata
    ORACLE_COMPRESSION_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
    
    oracle_model, compression_artifacts = phase_0_tune_and_train_oracle_compressed(
        X_train_sics, y_train_gm, n_trials=TRIALS_GLOBAL
    )
    
    joblib.dump(oracle_model, ORACLE_MODEL_PATH)
    joblib.dump(compression_artifacts, ORACLE_COMPRESSION_PATH)
    print(f"[REDACTED_BY_SCRIPT]")

    # --- CRITICAL: Augment Feature Space with Oracle Compression ---
    # We must append the PCA components to the dataframes so downstream models can access the latent signals.
    
    def _augment_with_compression(df, artifacts):
        # Extract tail features and transform
        tail_cols = artifacts['tail_features']
        pipeline = artifacts['tail_pipeline']
        
        # Robust check for missing columns
        available_tail = [c for c in tail_cols if c in df.columns]
        if len(available_tail) != len(tail_cols):
            # Fill missing with 0 to allow transform (Safe failover)
            missing = set(tail_cols) - set(available_tail)
            for c in missing:
                df[c] = 0
        
        # Transform to get PCA components
        try:
            pca_data = pipeline.transform(df[tail_cols])
            # Create DataFrame with Index Alignment
            n_components = pca_data.shape[1]
            pca_cols = [f'ORACLE_PCA_{i}' for i in range(n_components)]
            df_pca = pd.DataFrame(pca_data, index=df.index, columns=pca_cols)
            
            # Concatenate
            return pd.concat([df, df_pca], axis=1), pca_cols
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            return df, []

    print("[REDACTED_BY_SCRIPT]")
    X_train_sics, pca_features = _augment_with_compression(X_train_sics, compression_artifacts)
    X_val_sics, _ = _augment_with_compression(X_val_sics, compression_artifacts)
    X_test_sics, _ = _augment_with_compression(X_test_sics, compression_artifacts)

    # Define the Global Context Vector (GCV) for k-NN
    # GCV = Top Raw Features + Latent PCA Features
    top_feats = compression_artifacts['top_features']
    
    n_top = 45 # max(1, int(len(top_feats) * 0.25))
    top_feats_25 = top_feats[:n_top]
    gcv = top_feats_25 + pca_features
    print(f"[REDACTED_BY_SCRIPT]'top_features'[REDACTED_BY_SCRIPT]")

    # --- CRITICAL FIX: Synchronize Feature Space for Specialist Injection ---
    # Now we inject the V16 Specialist Features into the augmented dataframes.
    print("[REDACTED_BY_SCRIPT]")
    
    def _inject_specialist(df):
        spec = engineer_gm_specialist_consolidated_final(df)
        df_aug = pd.concat([df, spec], axis=1)
        return df_aug.loc[:, ~df_aug.columns.duplicated()]

    X_train_sics = _inject_specialist(X_train_sics)
    X_val_sics = _inject_specialist(X_val_sics)
    X_test_sics = _inject_specialist(X_test_sics)

    # --- AD-AM-33: Execute k-NN Anomaly Detection Sub-System (Post 3-Way Split) ---
    # k-NN now utilizes the Compressed GCV for higher fidelity neighbor finding.
    X_train_knn, X_val_knn, X_test_knn, knn_scaler, knn_engine, knn_imputer = phase_1b_engineer_knn_anomaly_features(
        X_train_sics, y_train_gm, X_val_sics, X_test_sics, gcv
    )
    joblib.dump(knn_imputer, KNN_IMPUTER_V35_PATH)
    joblib.dump(knn_scaler, KNN_SCALER_V35_PATH)
    joblib.dump(knn_engine, KNN_ENGINE_V35_PATH)
    print(f"[REDACTED_BY_SCRIPT]")

    # --- AD-AM-43: Execute Post-kNN Feature Engineering ---
    print("[REDACTED_BY_SCRIPT]")
    X_train_knn = engineer_post_knn_sics(X_train_knn)
    X_val_knn = engineer_post_knn_sics(X_val_knn)
    X_test_knn = engineer_post_knn_sics(X_test_knn)
    print("[REDACTED_BY_SCRIPT]")

    # --- AD-AM-37: Execute Project Regime Identification (Post 3-Way Split) ---
    REGIME_SCALER_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
    REGIME_MODEL_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
    X_train_gm, X_val_gm, X_test_gm, regime_scaler, regime_model = phase_D_forge_project_regimes(X_train_knn, X_val_knn, X_test_knn)
    joblib.dump(regime_scaler, REGIME_SCALER_PATH)
    joblib.dump(regime_model, REGIME_MODEL_PATH)
    print(f"[REDACTED_BY_SCRIPT]")

    # --- Define Cohorts using the Fully Forged Feature Space ---
    # CRITICAL FIX: We use the fully engineered 'X_train_gm' to define cohorts.
    # This ensures that Bayesian, Friction, Specialist, AND now PCA features are correctly registered.
    print("[REDACTED_BY_SCRIPT]")
    cohorts = define_all_cohorts(X_train_gm)
    
    print("[REDACTED_BY_SCRIPT]")
    
    # Update for Compression: The Oracle expects the Fused (Raw+PCA) array, not the dataframe.
    # We use the '[REDACTED_BY_SCRIPT]' helper to prepare the input matrix dynamically.
    
    print("[REDACTED_BY_SCRIPT]")
    # For training, we used TimeSeriesSplit in tuning, but for OOF generation on the full train set,
    # we need to transform the data using the fitted pipeline first.
    # Note: apply_oracle_compression returns a numpy array.
    X_train_oracle_input = apply_oracle_compression(X_train_sics, compression_artifacts)

    # Use the preserved groups
    if 'amaryllis_id' in X_train_groups.columns and X_train_groups['amaryllis_id'].nunique() > 1:
        groups = X_train_groups['amaryllis_id']
    elif 'lpa_name' in X_train_groups.columns and X_train_groups['lpa_name'].nunique() > 1:
        groups = X_train_groups['lpa_name']
    else:
        groups = pd.Series(0, index=X_train_groups.index)

    # Robust CV Selection
    n_groups = groups.nunique()
    if n_groups <= 1:
        print(f"[REDACTED_BY_SCRIPT]")
        cv_oracle = KFold(n_splits=SPLITS_GLOBAL, shuffle=True, random_state=RANDOM_STATE)
        # cross_val_predict doesn't take 'groups' if cv doesn't need it, but passing it as kwarg might be ignored or error depending on sklearn version.
        # Safest is to not pass groups if using KFold.
        oracle_preds_train_array = cross_val_predict(
            oracle_model, 
            X_train_oracle_input, 
            y_train_gm, 
            cv=cv_oracle,
            n_jobs=-1
        )
    else:
        n_splits_oracle = min(SPLITS_GLOBAL, n_groups)
        if n_splits_oracle < SPLITS_GLOBAL:
             print(f"[REDACTED_BY_SCRIPT]")
        
        cv_oracle = GroupKFold(n_splits=n_splits_oracle)
        oracle_preds_train_array = cross_val_predict(
            oracle_model, 
            X_train_oracle_input, 
            y_train_gm, 
            cv=cv_oracle,
            groups=groups,
            n_jobs=-1
        )
    oracle_preds_train = pd.Series(oracle_preds_train_array, index=X_train_sics.index, name="oracle_prediction")
    
    print("[REDACTED_BY_SCRIPT]")
    X_val_oracle_input = apply_oracle_compression(X_val_sics, compression_artifacts)
    oracle_preds_val = pd.Series(oracle_model.predict(X_val_oracle_input), index=X_val_sics.index, name="oracle_prediction")
    
    X_test_oracle_input = apply_oracle_compression(X_test_sics, compression_artifacts)
    oracle_preds_test = pd.Series(oracle_model.predict(X_test_oracle_input), index=X_test_sics.index, name="oracle_prediction")

    # --- AD-AM-30: Forge Specialist Heads for Parallel "Conclave of Experts" Architecture ---
    # The heads are now trained on the true target variable, not the Oracle's residual.
    # The residual calculation and signal gating logic have been excised to prevent signal sterilization.
    print("[REDACTED_BY_SCRIPT]")
    # --- AD-AM-32 (Revised): Forge "Anti-Oracle" Regressors ---
    # 1. Define the specialist training target: the Oracle's raw residual.
    oracle_error_train = y_train_gm - oracle_preds_train
    oracle_error_val = y_val_gm - oracle_preds_val 
    
    # 2. Weighted Error Focusing (Regression Adaptation)
    # We still use bins to determine sample weights (forcing models to focus on tails),
    # but the TARGET variable passed to the models will be continuous.
    error_bins = [-np.inf, -100, -25, 25, 100, np.inf]
    # We use temporary bins solely for weight calculation
    temp_error_bins = pd.cut(oracle_error_train, bins=error_bins, labels=[0, 1, 2, 3, 4]).astype(int)
    
    class_counts = temp_error_bins.value_counts().to_dict()
    # Inverse frequency weighting to prioritize high-error samples
    sample_weights = temp_error_bins.map(lambda x: len(temp_error_bins) / class_counts[x])
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    # 3. Tune and train the specialists to predict the Oracle's RESIDUAL (Continuous).
    #    Passing 'oracle_error_train' (Continuous) instead of 'y_head_target_train' (Categorical).
    
    # Define groups for Phase A
    groups_train = X_train_groups['amaryllis_id'] if 'amaryllis_id' in X_train_groups.columns else X_train_groups['lpa_name']
    
    best_head_params_dict = phase_A_tune_individual_heads(
        X_train_gm, oracle_error_train, sample_weights, cohorts, groups_train, n_trials=TRIALS_GLOBAL
    )
    top_head_features = phase_E_train_persist_and_analyze_heads(
        X_train_gm, oracle_error_train, sample_weights, cohorts, best_head_params_dict, HEADS_V35_DIR
    )

    # 5. Generate head predictions (Residuals) for all data splits.
    print("[REDACTED_BY_SCRIPT]")
    head_resid_preds_train, head_resid_preds_val, head_resid_preds_test = {}, {}, {}
    data_splits = {
        'train': (X_train_gm, head_resid_preds_train),
        'val': (X_val_gm, head_resid_preds_val),
        'test': (X_test_gm, head_resid_preds_test)
    }
    
    for cohort_name, features in cohorts.items():
        model_path = os.path.join(HEADS_V35_DIR, f"[REDACTED_BY_SCRIPT]")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            for split_name, (X_data, pred_dict) in data_splits.items():
                model_features = [f for f in features if f in X_data.columns]
                if model_features:
                    if split_name == 'train':
                         # Cross-Validation Predictions for Training Data (Prevent Leakage)
                        n_groups_head = groups_train.nunique()
                        if n_groups_head <= 1:
                            # Fallback to KFold
                            cv_head = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                            preds = cross_val_predict(
                                model, 
                                X_train_gm[model_features], 
                                oracle_error_train, 
                                cv=cv_head,
                                n_jobs=-1
                            )
                        else:
                            n_splits_head = min(5, n_groups_head)
                            cv_head = GroupKFold(n_splits=n_splits_head)
                            preds = cross_val_predict(
                                model, 
                                X_train_gm[model_features], 
                                oracle_error_train, 
                                cv=cv_head,
                                groups=groups_train,
                                n_jobs=-1
                            )
                    else:
                        # Standard Inference for Val/Test
                        preds = model.predict(X_data[model_features])
                    
                    pred_dict[cohort_name] = pd.Series(preds, index=X_data.index)

    # 6. Forge the Arbiter's feature matrices (Regression Mode)
    X_arbiter_train, arbiter_scaler = phase_F_generate_ridge_arbiter_features(
        X_train_gm, oracle_preds_train, head_resid_preds_train, top_head_features, stratified_residual_preds=None
    )
    X_arbiter_val, _ = phase_F_generate_ridge_arbiter_features(
        X_val_gm, oracle_preds_val, head_resid_preds_val, top_head_features, stratified_residual_preds=None, scaler=arbiter_scaler
    )
    X_arbiter_test, _ = phase_F_generate_ridge_arbiter_features(
        X_test_gm, oracle_preds_test, head_resid_preds_test, top_head_features, stratified_residual_preds=None, scaler=arbiter_scaler
    )
    
    print(f"[REDACTED_BY_SCRIPT]")
    # 7. Tune and train the final Ridge Arbiter.
    oracle_error_train = y_train_gm - oracle_preds_train
    oracle_error_val = y_val_gm - oracle_preds_val
    
    groups_val = X_val_groups['amaryllis_id'] if 'amaryllis_id' in X_val_groups.columns else X_val_groups['lpa_name']
    
    final_arbiter = phase_G_tune_and_train_lgbm_arbiter(
        X_arbiter_train, oracle_error_train, X_arbiter_val, oracle_error_val,
        groups_train, groups_val
    )
    
    joblib.dump(final_arbiter, MODEL_V35_PATH)
    joblib.dump(arbiter_scaler, os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]"))
    
    # --- NEW PHASE: THE SOLAR SPECIALIST CASCADE (AD-AM-45) ---
    print("\n========================================================")
    print("[REDACTED_BY_SCRIPT]")
    print("========================================================")

    # A. Isolate Solar Data (Tech Type 21)
    # Logic: We need indices to slice the arrays.
    # SURGICAL REMOVAL: Do not re-define gm_solar_df here. It breaks index alignment with X_train_gm 
    # which may contain non-solar data if the initial filter was loose.
    # gm_solar_df remains the "Training Universe" superset for safe .loc[] lookups.
    # gm_solar_df = master_df[(master_df['technology_type'] == 21)]
    
    # B. Get Arbiter Predictions on Training Data (to calculate residuals)
    arbiter_correction_train = final_arbiter.predict(X_arbiter_train)
    current_pred_train = oracle_preds_train + arbiter_correction_train
    
    # C. Calculate Residuals for the Solar Specialist
    # Target = True - (Oracle + Arbiter)
    solar_resid_target_train = y_train_gm - current_pred_train
    
    # D. Phase H: Forge Solar Bridge (Compression)
    # We must execute this on the Solar subset.
    X_train_solar_comp, X_val_solar_comp, X_test_solar_comp, solar_features, solar_pca, solar_scaler = phase_H_forge_solar_bridge(
        X_train_gm, y_train_gm, X_val_gm, X_test_gm, n_top_features=40, n_pca_components=20
    )
    if 'technology_type' in X_train_solar_comp.columns:
        X_train_solar_comp = X_train_solar_comp[(X_train_solar_comp['technology_type'] == 21)]
    else:
        print("  WARNING: 'technology_type'[REDACTED_BY_SCRIPT]" + "="*80)

    
    # --- FORK: Preserve Base Compressed Data for GM Specialist (Clean Base) ---
    # We create a copy of the compressed features BEFORE injecting the Solar-Specific FIs.
    # This ensures the GM model uses the shared base (PCA+TopN) but its own distinct interaction logic.
    X_train_gm_base_comp = X_train_solar_comp.copy()
    X_val_gm_base_comp = X_val_solar_comp.copy()
    X_test_gm_base_comp = X_test_solar_comp.copy()

    # --- INJECTION: STRATUM-SPECIFIC SPECIALIST FEATURES (Solar Bridge) ---
    print("[REDACTED_BY_SCRIPT]")
    # Generate features using source data (to ensure raw cols exist) and append to compressed
    # NOTE: This wrapper ALREADY includes '[REDACTED_BY_SCRIPT]', so no separate injection is needed.
    # The Fix:
    X_train_solar_comp = pd.concat([X_train_solar_comp, engineer_stratified_specialist_features(X_train_gm.loc[X_train_solar_comp.index])], axis=1)
    X_val_solar_comp = pd.concat([X_val_solar_comp, engineer_stratified_specialist_features(X_val_gm)], axis=1)
    X_test_solar_comp = pd.concat([X_test_solar_comp, engineer_stratified_specialist_features(X_test_gm)], axis=1)
    # --- END INJECTION ---

    # --- AD-AM-45: Persist Solar Bridge Artifacts ---
    joblib.dump(solar_pca, SOLAR_BRIDGE_PCA_PATH)
    joblib.dump(solar_scaler, SOLAR_BRIDGE_SCALER_PATH)
    joblib.dump(solar_features, SOLAR_BRIDGE_TOP_FEATURES_PATH)
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    solar_resid_target_train = solar_resid_target_train.loc[X_train_solar_comp.index]
    # E. Phase I: Train Solar Residual Specialist (R2 Predictor)
    solar_residual_model = phase_I_train_solar_residual_specialist(X_train_solar_comp, solar_resid_target_train)
    
    # F. Get Solar Resid Predictions 
    solar_resid_pred_train = solar_residual_model.predict(X_train_solar_comp)
    solar_resid_pred_test = solar_residual_model.predict(X_test_solar_comp)
    
    # G. STRICT CASCADE: Calculate Residual for Stratified Model (R3 Target)
    # Target = Previous_Residual - Previous_Prediction
    stratified_target_train = solar_resid_target_train - solar_resid_pred_train
    
    # H. Phase J: Train Stratified Capacity Residuals (Solar)
    # Note: X_train_solar_comp NOW GUARANTEED to have '[REDACTED_BY_SCRIPT]' (Phase H change)
    
    groups_train_solar = groups_train.loc[X_train_solar_comp.index]
    
    solar_stratified_models, sol_bins, sol_labels = phase_J_train_stratified_capacity_residuals(
        X_train_solar_comp, stratified_target_train, groups_train_solar
    )

    # Persist Solar Stratified Models AND Bin Definitions
    joblib.dump({'models': solar_stratified_models, 'bins': sol_bins, 'labels': sol_labels}, SOLAR_STRATIFIED_PATH)
    print(f"[REDACTED_BY_SCRIPT]")


    # I. Generate Predictions for Solar Stratified (on Train & Test)
    # We need Train preds to calculate the residual for the NEXT layer (Ground Mount)
    stratified_resid_pred_train = generate_stratified_predictions(X_train_solar_comp, solar_stratified_models, sol_bins, sol_labels)
    
    X_test_comp_with_cap = X_test_solar_comp.copy()
    if '[REDACTED_BY_SCRIPT]' not in X_test_comp_with_cap.columns:
        X_test_comp_with_cap['[REDACTED_BY_SCRIPT]'] = X_test_gm['[REDACTED_BY_SCRIPT]']
    stratified_resid_pred_test = generate_stratified_predictions(X_test_comp_with_cap, solar_stratified_models, sol_bins, sol_labels)

    # --- NEW PHASE: THE GROUND MOUNT SPECIALIST CASCADE (AD-AM-47) ---
    print("\n==========================================================")
    print("[REDACTED_BY_SCRIPT]")
    print("==========================================================")

    # 1. Isolate Ground Mount Data (Subset of the Solar Subset)
    # We filter the indices from the Solar Train/Test sets
    # Note: '[REDACTED_BY_SCRIPT]' == 1
    
    # Recover original metadata to filter
    # We use the index to align everything.
    
    def _isolate_gm(X_original, y_original, *pred_series_list):
        """[REDACTED_BY_SCRIPT]"""
        gm_mask = (X_original['[REDACTED_BY_SCRIPT]'] == 1)
        
        X_gm = X_original[gm_mask]
        y_gm = y_original[gm_mask]
        
        filtered_preds = []
        for preds in pred_series_list:
            # Handle both Series (index-aware) and Arrays (index-blind)
            if isinstance(preds, pd.Series):
                filtered_preds.append(preds[gm_mask])
            else:
                # If array, assume aligned with X_original
                filtered_preds.append(preds[gm_mask])
                
        return X_gm, y_gm, filtered_preds, gm_mask

    # Filter Training Data
    # We need the previous layer predictions to calculate the cumulative residual
    # Previous Layers: Oracle, Arbiter, SolarSpec, SolarStrat
    
    # Re-gather predictions aligned to X_train_gm index
    pred_oracle_tr = oracle_preds_train
    pred_arbiter_tr = pd.Series(arbiter_correction_train, index=X_train_gm.index)
    
    # Fix: Reindex solar predictions to match full training set
    pred_solar_tr = pd.Series(solar_resid_pred_train, index=X_train_solar_comp.index).reindex(X_train_gm.index)
    pred_sol_strat_tr = stratified_resid_pred_train.reindex(X_train_gm.index)

    X_train_gm_only, y_train_gm_only, preds_train_list, train_mask = _isolate_gm(
        X_train_gm, y_train_gm, pred_oracle_tr, pred_arbiter_tr, pred_solar_tr, pred_sol_strat_tr
    )
    p_oracle_tr, p_arb_tr, p_sol_tr, p_sol_strat_tr = preds_train_list

    # --- INJECTION: V16 SPECIALIST FEATURES (GM Only) ---
    print("[REDACTED_BY_SCRIPT]")
    
    # Generate V16 Features (Epochs, Green Lanes, Bio-Physics, etc.)
    # We generate these ONCE and reuse them for both Scale Corrector (Phase L) and Specialist (Phase K).
    # This ensures perfect consistency.
    train_gm_fis_v16 = engineer_gm_specialist_consolidated_final(X_train_gm_only)
    
    # Inject into Raw Training Set (for Phase L Scale Corrector)
    X_train_gm_only = pd.concat([X_train_gm_only, train_gm_fis_v16], axis=1)
    # --- END INJECTION ---
    
    # Filter Test Data (for final eval)
    pred_oracle_te = oracle_preds_test
    pred_arbiter_te = pd.Series(final_arbiter.predict(X_arbiter_test), index=X_test_gm.index)
    pred_solar_te = pd.Series(solar_resid_pred_test, index=X_test_gm.index)
    pred_sol_strat_te = pd.Series(stratified_resid_pred_test.values, index=X_test_gm.index)

    X_test_gm_only, y_test_gm_only, preds_test_list, test_mask = _isolate_gm(
        X_test_gm, y_test_gm, pred_oracle_te, pred_arbiter_te, pred_solar_te, pred_sol_strat_te
    )
    X_test_gm_only = X_test_gm_only[X_test_gm_only["[REDACTED_BY_SCRIPT]"] == 1]
    y_test_gm_only = y_test_gm_only.loc[X_test_gm_only.index]

    # --- CRITICAL FIX: Deduplicate Test Set Indices ---
    # We perform this unconditionally to ensure safety
    if X_test_gm_only.index.duplicated().any():
        print("[REDACTED_BY_SCRIPT]")
        # 1. Deduplicate Feature Matrix
        X_test_gm_only = X_test_gm_only.loc[~X_test_gm_only.index.duplicated(keep='first')]
        
        # 2. Align Target
        y_test_gm_only = y_test_gm_only.loc[X_test_gm_only.index]
        
        # 3. Align Prediction Components (Oracle, Arbiter, Solar, Stratified)
        # We reindex them to match the clean X_test_gm_only index
        preds_test_list = [p.loc[~p.index.duplicated(keep='first')].reindex(X_test_gm_only.index) for p in preds_test_list]

    p_oracle_te, p_arb_te, p_sol_te, p_sol_strat_te = preds_test_list

    # --- INJECTION: V16 SPECIALIST FEATURES (GM Only Test) ---
    print("[REDACTED_BY_SCRIPT]")
    test_gm_fis_v16 = engineer_gm_specialist_consolidated_final(X_test_gm_only)
    
    # Concatenate and immediately deduplicate columns to prevent ambiguity
    X_test_gm_only = pd.concat([X_test_gm_only, test_gm_fis_v16], axis=1)
    X_test_gm_only = X_test_gm_only.loc[:, ~X_test_gm_only.columns.duplicated()]
    # --- END INJECTION ---

    # 2. STRICT CASCADE: Calculate Residual for GM Specialist (R4 Target)
    # Target = True - (Oracle + Arbiter + SolarSpec + SolarStrat)
    # We must subtract ALL previous layers to find what's left for the GM Specialist to fix.
    current_prediction_train = p_oracle_tr + p_arb_tr + p_sol_tr + p_sol_strat_tr
    gm_resid_target_train = y_train_gm_only - current_prediction_train
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 3. Train GM Specialist (Phase K)
    # Feature Construction: Base Compressed Features + V16 GM Specialist FIs
    print("[REDACTED_BY_SCRIPT]")
    
    # A. Get Base Compressed Features (Clean Fork)
    # WARNING: X_train_gm_base_comp might have duplicate indices if not handled upstream.
    # We ensure alignment by index, but first we must ensure indices are unique to prevent reindexing errors.
    # --- ROBUST ALIGNMENT: Force Feature Matrix to Match Target Index ---
    # The target (gm_resid_target_train) is the ground truth for the rows we must train on.
    target_index_train = gm_resid_target_train.index

    # 1. Align Base Compressed Features
    if X_train_gm_base_comp.index.duplicated().any():
        X_train_gm_base_comp = X_train_gm_base_comp.loc[~X_train_gm_base_comp.index.duplicated(keep='first')]
    X_train_gm_comp_base = X_train_gm_base_comp.reindex(target_index_train)

    # 2. Align V16 Specialist Features
    if train_gm_fis_v16.index.duplicated().any():
        train_gm_fis_v16 = train_gm_fis_v16.loc[~train_gm_fis_v16.index.duplicated(keep='first')]
    X_train_gm_fis = train_gm_fis_v16.reindex(target_index_train)

    # 3. Concatenate and Deduplicate Columns
    # --- ROBUST ALIGNMENT: Force Feature Matrix to Match Target Index ---
    common_idx = X_train_gm_comp_base.index.intersection(X_train_gm_fis.index)
    X_train_gm_comp_base = X_train_gm_comp_base.loc[common_idx]
    X_train_gm_fis = X_train_gm_fis.loc[common_idx]

    X_train_gm_final = pd.concat([X_train_gm_comp_base, X_train_gm_fis], axis=1)
    X_train_gm_final = X_train_gm_final.loc[:, ~X_train_gm_final.columns.duplicated()]

    print(f"[REDACTED_BY_SCRIPT]")

    # 4. Align Weights Dataframe
    if gm_solar_df.index.duplicated().any():
         gm_solar_df_clean = gm_solar_df.loc[~gm_solar_df.index.duplicated(keep='first')]
         gm_train_subset_for_weights = gm_solar_df_clean.reindex(target_index_train)
    else:
         gm_train_subset_for_weights = gm_solar_df.reindex(target_index_train)

    gm_weights = calculate_maturity_weights(gm_train_subset_for_weights, target_col=y_general.name)

    print(f"[REDACTED_BY_SCRIPT]")
    
    # 3. Train GM Specialist (Phase K) - STRATIFIED
    groups_train_gm = groups_train.loc[X_train_gm_final.index]
    
    gm_stratified_models, gm_bins, gm_labels = phase_K_train_gm_specialist(
        X_train_gm_final, gm_resid_target_train, groups_train_gm, sample_weights=gm_weights
    )
    
    joblib.dump({'models': gm_stratified_models, 'bins': gm_bins, 'labels': gm_labels}, GM_SPECIALIST_PATH)
    
    # 4. Generate GM Specialist Predictions (Train & Test)
    gm_spec_pred_train = generate_stratified_predictions(X_train_gm_final, gm_stratified_models, gm_bins, gm_labels)
    
    # Prepare Test Set (Robust Alignment)
    target_index_test = X_test_gm_only.index
    
    if X_test_gm_base_comp.index.duplicated().any():
        X_test_gm_base_comp = X_test_gm_base_comp.loc[~X_test_gm_base_comp.index.duplicated(keep='first')]
    X_test_gm_comp_base = X_test_gm_base_comp.reindex(target_index_test)
    
    # V16 Features are generated on X_test_gm_only, so they match, but we reindex for safety
    X_test_gm_fis = engineer_gm_specialist_consolidated_final(X_test_gm_only)
    X_test_gm_fis = X_test_gm_fis.reindex(target_index_test)
    
    X_test_gm_final = pd.concat([X_test_gm_comp_base, X_test_gm_fis], axis=1)
    X_test_gm_final = X_test_gm_final.loc[:, ~X_test_gm_final.columns.duplicated()]
    
    # --- CRITICAL FIX: Align Test Features to Train Features ---
    # Ensure test set has exactly the same columns as training set to prevent LightGBM shape error
    # 1. Add missing columns (fill with 0)
    missing_cols = set(X_train_gm_final.columns) - set(X_test_gm_final.columns)
    for c in missing_cols:
        X_test_gm_final[c] = 0
        
    # 2. Drop extra columns
    extra_cols = set(X_test_gm_final.columns) - set(X_train_gm_final.columns)
    if extra_cols:
        print(f"[REDACTED_BY_SCRIPT]")
        X_test_gm_final = X_test_gm_final.drop(columns=list(extra_cols))
        
    # 3. Enforce exact column order
    X_test_gm_final = X_test_gm_final[X_train_gm_final.columns]
    
    gm_spec_pred_test = generate_stratified_predictions(X_test_gm_final, gm_stratified_models, gm_bins, gm_labels)
    
    # 5. Calculate Residual for GM Stratified
    # Target = True - ( ... + GM_Specialist)
    gm_strat_target_train = gm_resid_target_train - gm_spec_pred_train
    
    # --- MISSING BLOCK: DEFINE X_train_gm_comp_cap ---
    # We need to explicitly add the capacity column back to the compressed features
    # so the stratified model knows which bin to put the project in.
    X_train_gm_comp_cap = X_train_gm_comp_base.copy()
    
    # Check if capacity is already there (it might be in the Top N features)
    if '[REDACTED_BY_SCRIPT]' not in X_train_gm_comp_cap.columns:
        # If not, grab it from the raw GM-only training set
        X_train_gm_comp_cap['[REDACTED_BY_SCRIPT]'] = X_train_gm_only['[REDACTED_BY_SCRIPT]']
    # ------------------------------------------------
    
    # 6. Train GM Scale Corrector (Phase L - Consolidated)
    # We use the RAW GM training data (X_train_gm_only) for this, as it contains the raw scale features needed.
    gm_scale_corrector = phase_L_train_gm_scale_corrector(X_train_gm_only, gm_strat_target_train)
    joblib.dump(gm_scale_corrector, GM_STRATIFIED_PATH) # Re-using path variable for simplicity
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 7. Generate GM Scale Corrector Predictions
    # We use the RAW GM test data (X_test_gm_only)
    gm_scale_pred_test = gm_scale_corrector.predict(X_test_gm_only[gm_scale_corrector.feature_name_])

    # 9. Synthesize FINAL Ground Mount Prediction (THIS MUST HAPPEN BEFORE PHASE M)
    print("[REDACTED_BY_SCRIPT]")
    
    # ALIGNMENT GATE: Force all components to Series with the correct index
    target_index = y_test_gm_only.index
    
    # Layer 1-4: Already Series, but ensuring alignment
    v_oracle = p_oracle_te.reindex(target_index).fillna(0)
    v_arb = p_arb_te.reindex(target_index).fillna(0)
    v_sol = p_sol_te.reindex(target_index).fillna(0)
    v_sol_strat = p_sol_strat_te.reindex(target_index).fillna(0)
    
    # Layer 5: Numpy Array -> Series
    v_gm_spec = pd.Series(gm_spec_pred_test, index=target_index).fillna(0)
    
    # Layer 6: Series (Array -> Series)
    v_gm_scale = pd.Series(gm_scale_pred_test, index=target_index).fillna(0)
    
    # Summing all 6 layers
    final_gm_prediction = (
        v_oracle + 
        v_arb + 
        v_sol + 
        v_sol_strat + 
        v_gm_spec + 
        v_gm_scale
    )
    
    print(f"[REDACTED_BY_SCRIPT]")


    # ==============================================================================
    #  DEBUG EXPORT: Save Intermediate DataFrames to CSV
    # ==============================================================================
    
    # Define debug output path
    debug_dir = os.path.join(OUTPUT_DIR_V35, "debug_exports")
    os.makedirs(debug_dir, exist_ok=True)

    print(f"[REDACTED_BY_SCRIPT]")

    # 1. Export X_train_gm_final
    # Ensure this variable exists in your local scope at this point
    if 'X_train_gm_final' in locals():
        output_path_gm = os.path.join(debug_dir, "[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        X_train_gm_final.to_csv(output_path_gm)
        
        # Optional: Quick NaN check printed to console
        nan_counts = X_train_gm_final.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if not cols_with_nans.empty:
            print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")

    # 2. Export X_train_solar_comp
    # Ensure this variable exists in your local scope at this point
    if 'X_train_solar_comp' in locals():
        output_path_solar = os.path.join(debug_dir, "[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        X_train_solar_comp.to_csv(output_path_solar)
        
        # Optional: Quick NaN check printed to console
        nan_counts = X_train_solar_comp.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        if not cols_with_nans.empty:
            print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")


    # --- PHASE M: TRAIN TERMINAL CALIBRATOR (The Truth-Teller) ---
    # Now that 'final_gm_prediction' exists, we can train the calibrator.
    
    # Ensure Phase M function is defined in your script (add it before __main__ if you haven't).
    # X_test_gm_comp_cap contains the compressed features + capacity
    
    terminal_calibrator = phase_M_train_terminal_calibrator(
        X_test_gm_only, 
        y_test_gm_only, 
        final_gm_prediction
    )
    
    TERMINAL_CALIBRATOR_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
    joblib.dump(terminal_calibrator, TERMINAL_CALIBRATOR_PATH)
    print(f"[REDACTED_BY_SCRIPT]")

    # --- Reporting & Diagnostics ---
    
    rmse_gm = root_mean_squared_error(y_test_gm_only, final_gm_prediction)
    mae_gm = mean_absolute_error(y_test_gm_only, final_gm_prediction)
    r2_gm = r2_score(y_test_gm_only, final_gm_prediction)
    
    report_text_gm = (
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
    )
    print(report_text_gm)

    # Generate Error Distribution Plot (GM Only)
    error_gm = final_gm_prediction - y_test_gm_only
    plt.figure(figsize=(12, 7))
    sns.histplot(error_gm, kde=True, bins=30, color='green')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title(f"[REDACTED_BY_SCRIPT]", fontsize=16)
    plt.xlabel("[REDACTED_BY_SCRIPT]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plot_path = os.path.join(OUTPUT_DIR_V35, f"[REDACTED_BY_SCRIPT]")
    plt.savefig(plot_path)
    plt.close()

    # --- Save Dossier ---
    dossier_content.append("[REDACTED_BY_SCRIPT]")
    dossier_content.append("[REDACTED_BY_SCRIPT]")
    dossier_content.append(f"[REDACTED_BY_SCRIPT]")
    dossier_content.append(f"[REDACTED_BY_SCRIPT]")
    
    with open(DOSSIER_V35_PATH, 'w') as f:
        f.write("".join(dossier_content))
    print(f"[REDACTED_BY_SCRIPT]")

    # ==============================================================================
    # ARCHITECTURAL DIRECTIVE AD-AM-18: THE DIAGNOSTIC GAUNTLET
    # ==============================================================================
    print("[REDACTED_BY_SCRIPT]")
    
    # --- GLOBAL STATE ALIGNMENT ---
    # We explicitly define the variables required for the gauntlet using the 
    # definitive Ground Mount results we just generated.
    final_solar_prediction = final_gm_prediction
    y_test_gm = y_test_gm_only

    # Filter X_test_gm and oracle_preds_test to only include the GM holdout set
    if not isinstance(oracle_preds_test, pd.Series):
        oracle_preds_test = pd.Series(oracle_preds_test, index=X_test_gm.index)

    X_test_gm = X_test_gm.loc[y_test_gm.index]
    oracle_preds_test = oracle_preds_test.loc[y_test_gm.index]
    
    DIAGNOSTIC_DOSSIER_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
    diagnostic_dossier_content = ["[REDACTED_BY_SCRIPT]",
                                  "[REDACTED_BY_SCRIPT]"]

    # --- Setup for Diagnostics ---
    results_df = pd.DataFrame({
        'y_true': y_test_gm,
        'y_pred': final_solar_prediction
    }, index=y_test_gm.index)
    results_df['error'] = results_df['y_pred'] - results_df['y_true']
    results_df['residuals'] = results_df['y_true'] - results_df['y_pred']


    # --- Mandate 18.1 & 18.2: Decommissioned ---
    # The complex Arbiter model was removed per AD-AM-41.3. Therefore, SHAP analysis and
    # case studies on its behavior are no longer architecturally valid. The primary
    # diagnostic tool is now the SHAP analysis of the individual head classifiers.
    diagnostic_dossier_content.append("[REDACTED_BY_SCRIPT]")
    # A placeholder is needed for the hypothesis report
    importance_df = pd.DataFrame({'Head': ['COHORT_LPA_ALL']}) # Failsafe for hypothesis generator

    # --- Mandate 18.3: Systematic Bias Investigation ---
    bias_report = phase_D3_bias_investigation(results_df, OUTPUT_DIR_V35)
    diagnostic_dossier_content.append(bias_report)

    # --- Mandate 18.4: Feature Engineering Hypothesis Report ---
    hypothesis_report = phase_D4_generate_hypothesis_report(importance_df['Head'].iloc[0])
    diagnostic_dossier_content.append(hypothesis_report)

    # --- Mandate 18.5 (Addendum): All-Heads Granular SHAP Analysis ---
    all_heads_report = phase_D5_all_heads_shap_analysis(
        X_test_gm, cohorts, HEADS_V35_DIR, OUTPUT_DIR_V35, gcv, oracle_preds_test
    )
    diagnostic_dossier_content.append(all_heads_report)

    # ==============================================================================
    # ARCHITECTURAL DIRECTIVE AD-AM-31: THE FORENSIC DIAGNOSTIC GAUNTLET
    # ==============================================================================
    print("[REDACTED_BY_SCRIPT]")
    
    # Mandate 31.1: Decommissioned
    # The Arbiter model no longer exists, so waterfall plots of its decisions cannot be generated.
    
    # Mandate 31.2: Expert Correlation Analysis
    phase_D7_expert_correlation_heatmap(X_test_gm, cohorts, HEADS_V35_DIR, oracle_preds_test, OUTPUT_DIR_V35)

    # Mandate 31.3: Performance Stratification Report
    phase_D8_performance_stratification_report(results_df, X_test_gm)
    
    
    # # ==============================================================================
    # # DIRECTIVE: POISONOUS DATA EXCISION (POST-HOC)
    # # ==============================================================================
    # print("[REDACTED_BY_SCRIPT]")

    # # 1. Identify indices of samples from the holdout set with extreme residuals.
    # #    These are deemed "unlearnable" or "poisonous" by the project lead.
    # outlier_threshold = 100
    # # The 'error' column in results_df is defined as y_pred - y_true.
    # poisonous_indices = results_df[results_df['error'].abs() > outlier_threshold].index
    # print(f"[REDACTED_BY_SCRIPT]")

    # if not poisonous_indices.empty:
    #     # 2. Load the original, canonical target variable dataset.
    #     print(f"[REDACTED_BY_SCRIPT]")
    #     y_canonical = pd.read_csv(Y_REG_PATH, index_col=0).squeeze("columns")
    #     original_valid_count = y_canonical.notna().sum()
        
    #     # 3. Surgically nullify the labels for the identified poisonous samples.
    #     #    Setting the label to NaN ensures they will be dropped by the
    #     #    'valid_indices' filter in the next execution of the script's phase_1 function.
    #     y_canonical.loc[poisonous_indices] = np.nan
    #     print(f"[REDACTED_BY_SCRIPT]")
        
    #     # 4. Overwrite the canonical dataset. This action permanently alters the
    #     #    source data for all subsequent training runs.
    #     y_canonical.to_csv(Y_REG_PATH, header=True)
    #     final_valid_count = y_canonical.notna().sum()
    #     print(f"[REDACTED_BY_SCRIPT]")
    # else:
    #     print("[REDACTED_BY_SCRIPT]")


    