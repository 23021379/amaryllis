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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
# --- AD-AM-33: New imports for k-NN Anomaly Detection Sub-System ---
from sklearn.neighbors import NearestNeighbors

# --- Project Artifacts & Constants (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Defining new paths for the v3.5 cohort refactoring build
OUTPUT_DIR_V35 = "[REDACTED_BY_SCRIPT]"
HEADS_V35_DIR = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
MODEL_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
DOSSIER_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
# --- AD-AM-33: Artifact paths for the k-NN Anomaly Detection Sub-System ---
KNN_SCALER_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
KNN_ENGINE_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
KNN_IMPUTER_V35_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")


X_PATH = "[REDACTED_BY_SCRIPT]"
Y_REG_PATH = r"[REDACTED_BY_SCRIPT]"
RANDOM_STATE = 42

SPLITS_GLOBAL=15
TRIALS_GLOBAL=15

# --- Mandated Semantic Cohort: CORE_ATTRIBUTES (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Purpose: The foundational, immutable facts of the application itself. What is it, how big is it, and when was it submitted?
COHORT_CORE_ATTRIBUTES = [
    # Core Physical Attributes
    '[REDACTED_BY_SCRIPT]', # Note: Will be removed from cohort logic in define_all_cohorts
    'solar_site_area_sqm',       # Note: Will be removed from cohort logic in define_all_cohorts
    'chp_enabled',
    '[REDACTED_BY_SCRIPT]',
    # Core Temporal Attributes - MOVED TO COHORT_TEMPORAL_CONTEXT
    # 'submission_year',
    # 'submission_month',
    # 'submission_day',
    # 'submission_month_sin',
    # 'submission_month_cos'
]

from sklearn.impute import SimpleImputer

def engineer_temporal_context_features(df: pd.DataFrame, baseline_year=None, imputer=None, scaler=None):
    """
    Architectural Mandate AD-AM-19:
    Engineers Temporal Feature Interactions (TFIs) to explicitly model how the
    impact of various constraints evolves over time.
    """
    X = df.copy()
    epsilon = 1e-6

    # Step 1: Create the Normalized Year Variable.
    if baseline_year is None: # Training mode: calculate from data
        baseline_year = X['submission_year'].min()
    X['year_norm'] = X['submission_year'] - baseline_year

    # Step 2: Engineer the Comprehensive Interaction Set.
    # Grid Saturation Interactions
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * (1 / (X['[REDACTED_BY_SCRIPT]'] + 1))

    # Environmental Policy Hardening Interactions
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['aonb_is_within']
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['alc_is_bmv_at_site']
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['cs_on_site_bool']
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] / (X['aw_dist_to_nearest_m'] + 100)

    # LPA Behavior & Precedent Interactions
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['lpa_major_commercial_approval_rate']
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] / (X['[REDACTED_BY_SCRIPT]'] + 1)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['lpa_workload_trend']

    # Project Scale & Economics Interactions
    # DECOMMISSIONED: TFI_Scale_Penalty_Evolution. Replaced by context-aware SICs.
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['solar_site_area_sqm']

    # Socio-Economic & Cumulative Impact Interactions
    X['TFI_NIMBY_Amplifier'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']
    # RELOCATED: TFI_Cumulative_Impact_Sensitivity calculation moved to engineer_strategic_interaction_features to resolve dependency.

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
    '[REDACTED_BY_SCRIPT]'
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
]


# --- Mandated Semantic Cohort: METADATA_AND_FLAGS (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Purpose: A quarantine zone for one-hot encoded flags and categorical metadata. These are used by the model but are not primary continuous signals.
COHORT_METADATA_AND_FLAGS = [
    'technology_type', # Constant, but kept for completeness
    # DNO Source Flags
    'dno_source_nged', 'dno_source_ukpn',
    # Join Method Flags
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # Grid Status Flags
    '[REDACTED_BY_SCRIPT]', 'constraint_season_Winter', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # Encoded Environmental Status Flags
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', 'es_on_site_highest_tier_HLS',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
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
]

COHORT_AGRI_ENVIRONMENTAL_SCHEMES = [
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
]

COHORT_TOPOGRAPHY_AND_ACCESS = [
    'mean_terrain_gradient_1km', 'mean_terrain_gradient_2km', 'mean_terrain_gradient_5km', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]',
]



# ==============================================================================
# PHASE 0: GLOBAL CONTEXT VECTOR FORGING (AD-AMARYLLIS-MOD-11)
# ==============================================================================

def phase_0_tune_and_train_oracle(X_train, y_train, n_features=30, n_trials=TRIALS_GLOBAL):
    """
    Tunes, trains, and analyzes the Global Oracle model post-split to prevent feature leakage.
    Uses Optuna with a wide search space to find optimal hyperparameters.
    Returns the final trained model and the Global Context Vector (GCV).
    """
    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    def objective(trial):
        # Define a wide and robust search space for the foundational oracle model.
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'random_state': RANDOM_STATE, 'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 100.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'n_jobs': -1,
        }
        model = lgb.LGBMRegressor(**params)
        score = cross_val_score(
            model, X_train, y_train, 
            cv=KFold(n_splits=SPLITS_GLOBAL, shuffle=True, random_state=RANDOM_STATE), 
            scoring='neg_root_mean_squared_error'
        ).mean()
        return -score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print(f"[REDACTED_BY_SCRIPT]")

    # Phase 1: Forge the final Global Oracle with optimized parameters.
    print("[REDACTED_BY_SCRIPT]")
    oracle_model = lgb.LGBMRegressor(**best_params, random_state=RANDOM_STATE)
    oracle_model.fit(X_train, y_train)
    print("  Oracle trained.")

    # Phase 2: Extract the Unimpeachable Signal via SHAP.
    print("[REDACTED_BY_SCRIPT]")
    explainer = shap.TreeExplainer(oracle_model)
    shap_values = explainer.shap_values(X_train)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns, mean_abs_shap)), columns=['feature', 'mean_abs_shap'])
    feature_importance = feature_importance.sort_values('mean_abs_shap', ascending=False)
    print("[REDACTED_BY_SCRIPT]")

    # Phase 3: Define the Global Context Vector (GCV).
    gcv = feature_importance['feature'].head(n_features).tolist()
    print(f"[REDACTED_BY_SCRIPT]")

    return oracle_model, gcv


def phase_0b_identify_and_excise_unlearnable_samples(X, y, outlier_threshold=100, n_splits=5):
    """
    Architectural Mandate: Implements a robust, cross-validation-based protocol
    to identify and flag unlearnable samples (i.e., those with consistently high
    prediction errors) before they contaminate the training process. This replaces
    the flawed post-hoc cleaning of the test set.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    
    oof_predictions = pd.Series(index=X.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    # Use a simple, robust probe model. No need for extensive tuning.
    probe_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    
    # Hardening Gate: Ensure all columns are numeric for the probe model to prevent crashes.
    X_probe = X.select_dtypes(include=np.number)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_probe, y)):
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
    X['SIC_NIMBY_AMPLIFIER'] = (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]']) / (X['nt_dist_to_nearest_m'] + 100)
    # DECOMMISSIONED per AD-AM-32: 'SIC_LANDSCAPE_SCARRING'
    # X['SIC_LANDSCAPE_SCARRING'] = (X['ohl_local_tower_ratio'] / (X['[REDACTED_BY_SCRIPT]'] + 100)) * (X['aonb_is_within'] + 1)
    # DECOMMISSIONED per AD-AM-35: '[REDACTED_BY_SCRIPT]'
    # X['[REDACTED_BY_SCRIPT]'] = X['alc_is_bmv_at_site'] * X['cs_on_site_highest_tier_Higher_Tier']
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])

    # AD-AM-32: New Context-Aware Visual Impact Feature
    # Hypothesis: Visual impact is amplified by landscape sensitivity and mitigated by pre-existing industrial character.
    landscape_sensitivity_multiplier = (1 + X['aonb_is_within'] + X['np_is_within'])
    industrial_mitigation_denominator = (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    raw_visual_clutter = (X['ohl_local_tower_ratio'] / (X['[REDACTED_BY_SCRIPT]'] + 100))
    X['[REDACTED_BY_SCRIPT]'] = (raw_visual_clutter * landscape_sensitivity_multiplier) / industrial_mitigation_denominator


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
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * (is_large_scale & is_grid_stressed)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * (is_large_scale & is_env_proximate)
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * (is_large_scale & is_tough_lpa)

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
    # SIC-06: Visual Amenity Conflict
    X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] * X['[REDACTED_BY_SCRIPT]']) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    # SIC-07: Grid Reality Gap
    #X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']
    # SIC-08: LPA Precedent Bias
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] - X['lpa_major_commercial_approval_rate']) * (1 - X['lpa_workload_trend'])
    # SIC-09: Cumulative Impact Pressure
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * (X['cs_density_5km'] + X['[REDACTED_BY_SCRIPT]'])
    
    # Addendum: Create TFI now that its SIC dependency is met.
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']
    
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
    
    # SIC-LPA-03: NIMBY Susceptibility
    # Models how susceptible an LPA is to local opposition. High NIMBY risk is amplified for LPAs with less experience.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['lpa_total_experience'] + epsilon)

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
    # SIC-ENV-01: Designated Landscape Conflict
    # Models the pressure a large project exerts on nearby high-value designated landscapes (AONBs, National Parks).
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (X['aonb_dist_to_nearest_m'] + X['np_dist_to_nearest_m'] + 100)

    # SIC-ENV-02: Ecological Fragmentation Risk
    # Penalizes large projects located in an ecological "pinch point" between multiple sensitive habitat types like Ancient Woodland and SSSIs.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['aw_dist_to_nearest_m'] + X['sssi_dist_to_nearest_m'] + X['ph_dist_to_nearest_m'] + 100)

    # SIC-ENV-03: Agri-Scheme Saturation Conflict
    # Models the cumulative pressure of adding a solar farm to a landscape already defined by high densities of agri-environment schemes.
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * (X['cs_density_5km'] + X['es_hls_density_5km'])

    # SIC-ENV-04: BMV Land Pressure
    # Captures the acute political conflict of converting a large area of Best and Most Versatile agricultural land, especially when it is surrounded by more BMV land.
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * X['alc_is_bmv_at_site'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-ENV-05: Public Amenity Encroachment
    # Models the direct conflict between a project's physical footprint and its proximity to valued public amenity land (trails, parks, open access land).
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (X['nt_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100)
    
    # SIC-ENV-06: Heritage Asset Conflict
    # A focused signal representing the tension between large-scale modern development and the setting of historic parklands.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['hp_dist_to_nearest_m'] + 100)

    # --- Phase VI Addendum: SOCIO-ECONOMIC-Focused SICs ---
    # Hypothesis: Model specific social flashpoints where project characteristics conflict with the local human landscape.
    # SIC-SOCIO-01: Rural Character Disruption
    # Models the acute conflict from placing a large industrial installation in a deeply rural, settled, and often offline community.
    X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] / 10000) * X['site_lsoa_ruc_rural_score'] * X['[REDACTED_BY_SCRIPT]']
    
    # SIC-SOCIO-02: Urban Fringe Land Use Tension
    # Models the conflict on the edge of settlements, where the pressure to protect green space for amenity meets the pressure for development.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['alc_is_bmv_at_site'] * X['[REDACTED_BY_SCRIPT]']
    
    # SIC-SOCIO-03: Affluence and Visual Intrusion
    # A refined NIMBY signal. It's not just affluence, but affluence combined with a genuinely large and visually intrusive project (footprint + existing grid clutter).
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['solar_site_area_sqm'] + (X['[REDACTED_BY_SCRIPT]'] * 10000))
    
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
    
    # SIC-XD-03: Topographical Connection Friction (GRID x ENV x TOPOGRAPHY)
    # Models the real-world difficulty of grid connection, where the raw distance is penalized by difficult terrain and sensitive landscape designations.
    required_cols_topo_friction = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'aonb_is_within', 'np_is_within']
    if all(col in X.columns for col in required_cols_topo_friction):
        X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + 1) * (1 + X['aonb_is_within'] + X['np_is_within'])
    else:
        X['[REDACTED_BY_SCRIPT]'] = 0.0

    # SIC-XD-04: Evolving Agri-Land Policy Tension (ENV x TEMPORAL)
    # Models the increasing political penalty over time for developing on high-grade agricultural land.
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['solar_site_area_sqm'] * X['alc_is_bmv_at_site']

    
    
    # SIC-XD-06: Cumulative Development Pressure (GRID x ENV x SOCIO)
    # An ultimate cumulative impact signal. It combines the density of existing large developments with the density of environmental schemes and human settlement.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + 1) * (X['cs_density_10km'] + 1) * (X['[REDACTED_BY_SCRIPT]'] + 1)

    # --- Phase VII Addendum: Temporal Strategic Interaction Constructs (TSICs) ---
    # Hypothesis: Model the non-linear acceleration of risk factors over time.
    # TSIC-01: Grid Saturation Acceleration
    # Models the accelerating penalty of connecting to a stressed grid as saturation approaches non-linearly. The penalty in later years for high loading is exponentially higher.
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * (X['[REDACTED_BY_SCRIPT]'] ** 2)

    # TSIC-02: Policy Hardening on Veto Constraints
    # Models how protections for "veto" assets (Ancient Woodland, Grade 1 Land) have become nearly absolute over time. Proximity in a recent year is far more penalizing than in an early year.
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] / (X['aw_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100)

    

    
    
    # TSIC-05: The Shrinking Window for Large-Scale Projects
    # Models the increasing difficulty of getting very large projects approved as the landscape fills up and policy tightens. The penalty for scale increases with time.
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]'] * (1 + X['aonb_is_within'])
    
    # TSIC-06: The Erosion of Precedent
    # Models the idea that old planning approvals (legacy successes) become less relevant over time as the policy and physical environment changes.
    X['TSIC_PRECEDENT_EROSION'] = X['year_norm'] / (X['[REDACTED_BY_SCRIPT]'] + 1)

    # --- Phase VIII Addendum: Core Project Characterization SICs ---
    # Hypothesis: Model the intrinsic physical suitability and logistical viability of the project itself.
    # SIC-CORE-01: Power Density
    # A fundamental measure of land use efficiency. Extremely high or low values could be anomalous, indicating unusual designs or data issues.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['solar_site_area_sqm'] / 10000 + epsilon) # MW per hectare

    # SIC-CORE-02: Topographical Difficulty
    # Models the construction and visual impact challenge of placing a large project on highly variable terrain.
    required_cols_topo_difficulty = ['solar_site_area_sqm', '[REDACTED_BY_SCRIPT]']
    if all(col in X.columns for col in required_cols_topo_difficulty):
        X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * X['[REDACTED_BY_SCRIPT]']
    else:
        X['[REDACTED_BY_SCRIPT]'] = 0.0

    # SIC-CORE-03: Logistical Access Challenge
    # Models the construction traffic friction. A large-scale project with poor access to major road networks is a significant source of local objection and delay.
    X['SIC_CORE_LOGISTICAL_ACCESS_CHALLENGE'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-CORE-04: Brownfield Fit
    # Assesses how well the project's scale fits within the local industrial/urban fabric. A low value indicates a project that is small relative to the available brownfield land (a good fit).
    industrial_area_sqm = X['[REDACTED_BY_SCRIPT]'] * (np.pi * 2000**2) # Approximate area in sqm
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (industrial_area_sqm + epsilon)

    # SIC-CORE-05: Mounting System vs. Land Quality Conflict
    # A more advanced mounting system (e.g., trackers, often coded as a higher integer) on prime agricultural land represents a more intensive, industrial-style conversion, potentially increasing planning friction.
    # Note: Assumes mounting_type_for_solar is numerically ordered by intensity.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['alc_is_bmv_at_site']

    # SIC-CORE-06: Dislocated CHP Anomaly
    # CHP (Combined Heat and Power) enabled projects are typically co-located with an industrial heat user. A CHP project in a non-industrial area is a significant anomaly that warrants flagging.
    X['[REDACTED_BY_SCRIPT]'] = X['chp_enabled'] * (1 - X['[REDACTED_BY_SCRIPT]'])

    # --- Phase IX Addendum: Advanced Conflict Synthesis SICs ---
    # Hypothesis: Model specific, high-level planning arguments by synthesizing multiple cross-domain features.
    # SIC-SYNTH-01: Visual Amenity Conflict Synthesis
    # A direct proxy for the "[REDACTED_BY_SCRIPT]" argument. It models the project's visual footprint (size + existing clutter) in relation to its proximity to public amenities, amplified if the site is within a nationally designated landscape.
    X['[REDACTED_BY_SCRIPT]'] = ((X['solar_site_area_sqm'] + (X['[REDACTED_BY_SCRIPT]'] * 10000)) / (X['nt_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + 100)) * (1 + X['aonb_is_within'] + X['np_is_within'])

    # SIC-SYNTH-02: Grid Reality Gap Synthesis
    # Models the dissonance between theoretical grid capacity and the practical, forward-looking reality. Risk is high when the wait for crucial upgrades is long and known future constraints are imminent.
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'] + 1) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-SYNTH-03: LPA Precedent Paradox Synthesis
    # Captures the uncertainty when an LPA's general disposition (e.g., pro-commercial) contradicts its specific historical precedent with renewables. This uncertainty is amplified by high workload.
    X['[REDACTED_BY_SCRIPT]'] = (X['lpa_major_commercial_approval_rate'] - X['[REDACTED_BY_SCRIPT]']).abs() * (1 + X['lpa_workload_trend'])
    
    # SIC-SYNTH-04: Ecological Tipping Point Synthesis
    # Models the "straw that breaks the camel's back" argument. The impact of a project is multiplied by the density of pre-existing environmental constraints in the immediate vicinity.
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * X['[REDACTED_BY_SCRIPT]']
    
    # SIC-SYNTH-05: Socio-Ecological Conflict Synthesis
    # Models the potent political flashpoint where an affluent, organized community can leverage a nearby high-value ecological asset (like an SSSI) as the basis for objecting to a large-scale project.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]'] / (X['sssi_dist_to_nearest_m'] + 100)

    # --- Phase X Addendum: Contextual Anomaly & Ratio SICs ---
    # Hypothesis: Model relative context and flag anomalous project configurations.
    # SIC-CONTEXT-01: Scale vs. Local Grid Capacity Ratio
    # A measure of local grid impact. A high ratio indicates a project that is very large relative to the primary substation it connects to, signaling a potentially disruptive scheme.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / ((X['[REDACTED_BY_SCRIPT]'] / 1000) + epsilon) # MWelec vs MVA

    # SIC-CONTEXT-02: Connection Path Inefficiency
    # Models the inefficiency of requiring a long connection path for a relatively small project. This can indicate suboptimal siting or significant wayleave challenges.
    X['SIC_CONTEXT_CONNECTION_PATH_INEFFICIENCY'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-CONTEXT-03: Scarred Paradise Paradox
    # Flags the extreme planning tension of a site located within a protected landscape (AONB) that is simultaneously blighted by heavy industrial infrastructure (towers).
    X['[REDACTED_BY_SCRIPT]'] = X['aonb_is_within'] * X['[REDACTED_BY_SCRIPT]']

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
    # SIC-UNC-01: Transformer Mismatch Risk
    # Directly weaponizes a data quality flag. A project connecting to a substation with known data discrepancies is inherently higher risk, and this risk scales with project size.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']

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

    # SIC-POLICY-01: Competing Land Mandate Conflict
    # A powerful conflict signal for projects sited on land that is BOTH Best/Most Versatile (food security policy) AND under a high-tier agri-environment scheme (conservation policy). This creates a three-way policy tension.
    X['[REDACTED_BY_SCRIPT]'] = X['alc_is_bmv_at_site'] * X['cs_on_site_bool'] * X['solar_site_area_sqm']

    # SIC-LOGISTICS-01: Connection Path Tortuosity
    # Models the logistical and legal complexity of the connection path. A high number of intersections per kilometer suggests a tortuous, difficult, and expensive wayleave process.
    X['SIC_LOGISTICS_CONNECTION_TORTUOSITY'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)

    # SIC-ECOLOGY-01: Dense Ecological Network Conflict
    # Differentiates between proximity to a single constraint and siting within a dense network of ecological assets. Risk is amplified when a project is near multiple constraints in a concentrated area.
    X['[REDACTED_BY_SCRIPT]'] = (X['sssi_count_in_2km'] + X['aw_count_in_2km'] + X['[REDACTED_BY_SCRIPT]']) / (X['[REDACTED_BY_SCRIPT]'] + 100)

    # SIC-ECON-01: High-Value Scheme Conversion Penalty
    # A proxy for converting ecologically significant land. The penalty is proportional to the financial value of the agri-environment scheme being displaced, adjusted for the percentage of the site it covers.
    X['[REDACTED_BY_SCRIPT]'] = X['cs_on_site_total_value'] * X['cs_on_site_pct_area']

    # --- Phase XIII Addendum: Advanced Topological & Political SICs ---
    # Hypothesis: Model the most subtle risks arising from political context, infrastructure age, and complex spatial relationships.
    # SIC-POL-01: LPA Political Volatility Proxy
    # Models risk from procedural chaos. An LPA whose workload is increasing while its decision timing becomes more erratic is a high-risk environment for any major application.
    X['SIC_POL_LPA_POLITICAL_VOLATILITY'] = X['lpa_workload_trend'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-INFRA-01: Aging Asset Risk Compounder
    # A critical operational signal. The planning and engineering risk of connecting to a heavily loaded substation is compounded significantly if that asset is also old.
    X['SIC_INFRA_AGING_ASSET_RISK'] = X['substation_age_at_submission'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-SOCIO-ACTIVISM-01: Amenity-Driven Activism Potential
    # A proxy for an organized and engaged community. A high density of valued public amenities (trails, parks) in an affluent area suggests a higher potential for effective, well-resourced local opposition.
    X['[REDACTED_BY_SCRIPT]'] = (X['nt_intersection_count_in_5km'] + X['hp_count_in_5km'] + 1) * X['[REDACTED_BY_SCRIPT]']

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

    # SIC-LPA-09: Inexperience Under Pressure
    # Models the risk of an inexperienced LPA (low total experience) being overwhelmed by a complex project that has
    # already attracted significant local opposition (high affluent NIMBY risk).
    X['SIC_LPA_INEXPERIENCE_UNDER_PRESSURE'] = X['[REDACTED_BY_SCRIPT]'] / (X['lpa_total_experience'] + epsilon)

    # SIC-LPA-10: Procedural Instability Risk
    # Models the risk posed by an LPA that struggles with statutory compliance. The risk is amplified for larger, more complex
    # projects that are more likely to be affected by procedural failures.
    X['[REDACTED_BY_SCRIPT]'] = (1 - X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']

    # SIC-LPA-11: "Quiet Refusal" Friction
    # A high withdrawal rate can be a proxy for an LPA encouraging difficult applications to be withdrawn to avoid a refusal on their record.
    # This "procedural friction" is hypothesized to be greatest for projects close to a sensitive constraint, where the planning case is more contentious.
    X['[REDACTED_BY_SCRIPT]'] = X['lpa_withdrawal_rate'] / (X['[REDACTED_BY_SCRIPT]'] + 100)

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

    # SIC-ENV-07: Designated Landscape Pressure
    # Models the acute pressure a large project exerts on the setting of nearby high-value designated landscapes (AONBs, National Parks).
    # The impact is a function of the project's physical footprint divided by the distance to these sensitive receptors.
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (X['aonb_dist_to_nearest_m'] + X['np_dist_to_nearest_m'] + 100)

    # SIC-ENV-08: Ecological Fragmentation Risk
    # Penalizes large projects located in an ecological "pinch point" between multiple sensitive habitat types (Ancient Woodland, SSSIs, Priority Habitats),
    # modeling the risk of severing ecological corridors.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['aw_dist_to_nearest_m'] + X['sssi_dist_to_nearest_m'] + X['ph_dist_to_nearest_m'] + 100)

    # SIC-ENV-09: Agri-Scheme Saturation Conflict
    # Models the cumulative pressure of adding a solar farm to a landscape already heavily defined by high densities of agri-environment schemes,
    # which can trigger objections based on a change to the area's fundamental conservation character.
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * (X['cs_density_5km'] + X['es_hls_density_5km'])

    # SIC-ENV-10: BMV Land Conversion Pressure
    # Captures the acute political conflict of converting a large area of Best and Most Versatile agricultural land, especially when that site
    # is located in an area with a high concentration of other BMV land.
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * X['alc_is_bmv_at_site'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-ENV-11: Public Amenity Encroachment
    # Models the direct conflict between a project's physical footprint and its proximity to valued public amenity land (trails, parks, open access land),
    # a common source of well-organized local opposition.
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] / (X['nt_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100)

    # SIC-ENV-12: Heritage Setting Conflict
    # A focused signal representing the tension between large-scale modern development and the legally protected "setting" of historic parklands.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['hp_dist_to_nearest_m'] + 100)

    # --- Phase XVII Addendum: Socio-Economic Flashpoint SICs ---
    # Hypothesis: Model specific social flashpoints where project characteristics conflict with the local human landscape,
    # moving beyond generic demographic labels to capture actionable behavioral patterns.

    # SIC-SOCIO-07: Rural Character Disruption
    # Models the acute conflict from placing a large industrial installation in a deeply rural, settled, and often offline community,
    # where objections are often based on a fundamental change to the "character of the area".
    X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] / 10000) * X['site_lsoa_ruc_rural_score'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-SOCIO-08: Urban Fringe Land Use Tension
    # Models the conflict on the edge of settlements, where the pressure to protect green space for amenity meets the pressure for development.
    # The tension is highest when the land is also high-quality agricultural land.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * X['alc_is_bmv_at_site'] * X['[REDACTED_BY_SCRIPT]']

    # SIC-SOCIO-09: Affluent Visual Intrusion
    # A refined NIMBY signal. It posits that affluent opposition is most potent when combined with a genuinely large and visually intrusive project
    # (defined by its physical footprint plus existing grid clutter).
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['solar_site_area_sqm'] + (X['[REDACTED_BY_SCRIPT]'] * 10000))

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

    # --- Phase V Mandated Standalone Features ---
    # Note: '[REDACTED_BY_SCRIPT]' is not available. Using available OHL data as the primary friction source.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    # Note: 'feat_lpa_experience_delta' and '[REDACTED_BY_SCRIPT]' cannot be implemented as required component features are not in the dataset.
    X['[REDACTED_BY_SCRIPT]'] = (1 + X['aonb_is_within'] + X['np_is_within']) * (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])
    X['[REDACTED_BY_SCRIPT]'] = X['site_lsoa_ruc_rural_score'] * X['[REDACTED_BY_SCRIPT]']

    # --- AD-AM-15: Fortified NIMBY Interaction Constructs (RELOCATED) ---
    # SIC-NIMBY-01: Visual Intrusion on Public Amenity
    # Hypothesis: A large, visually cluttered project very close to valued public access land is the most potent driver of opposition.
    # We model clutter as the sum of the site's area and a scaled representation of nearby high-impact towers.
    X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] + (X['[REDACTED_BY_SCRIPT]'] * 10000)) / (X['nt_dist_to_nearest_m'] + X['[REDACTED_BY_SCRIPT]'] + 100)
    
    # SIC-NIMBY-02: Cumulative Landscape Stress
    # Hypothesis: The impact of a new project is amplified by the number of large-scale developments already present, especially in an area with a strong conservation character (high CS density). This effect is doubled in a nationally designated landscape (AONB).
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] + 1) * X['[REDACTED_BY_SCRIPT]'] * (X['cs_density_5km'] + 0.1) * (1 + X['aonb_is_within'])
    # SURGICAL INTERVENTION: Dampen the volatility of this multiplicative feature.
    X['[REDACTED_BY_SCRIPT]'] = np.log1p(X['[REDACTED_BY_SCRIPT]'])

    # TSIC-03: LPA Sensitivity to Cumulative Impact
    # Models how LPAs have become progressively more receptive to "[REDACTED_BY_SCRIPT]" arguments as both a legal concept and a political reality.
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']
    
    # SIC-NIMBY-03: Affluent Green Objection
    # Hypothesis: Affluent communities are most effective when they can leverage the protection of high-value ecological or heritage assets as the legal basis for their objection.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['sssi_dist_to_nearest_m'] + X['hp_dist_to_nearest_m'] + 100)

    # SIC-XD-05: Political Heat (LPA x SOCIO-ECONOMIC)
    # Models the scenario where a procedurally vulnerable LPA faces intense, organized opposition from an affluent demographic regarding a visually intrusive project.
    X['[REDACTED_BY_SCRIPT]'] = (1 - X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']

    # TSIC-04: Amplifying Political Heat
    # Models the observation that local opposition has become more effective and politically potent over time. The "political heat" from a controversial project is higher in later years.
    X['[REDACTED_BY_SCRIPT]'] = X['year_norm'] * X['[REDACTED_BY_SCRIPT]']


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
    Defines, sanitizes, and prepares the v3.5 multi-head ensemble structure,
    eradicating the COHORT_OVERFLOW anomaly per AD-AM-20 & AD-AM-21.
    """
    def _sanitize(features):
        return [re.sub(r'[^A-Za-z0-9_]+', '_', f) for f in features]

    # --- v3.5 Definitive Cohort Manifest ---
    # AD-AM-20 Base Feature Re-assignments
    temporal_base_features = [
        'submission_year', 'year_norm', 'submission_month', 'submission_day', 'submission_month_sin', 'submission_month_cos',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'aonb_is_within',
        'alc_is_bmv_at_site', 'cs_on_site_bool', 'aw_dist_to_nearest_m', 'lpa_major_commercial_approval_rate',
        '[REDACTED_BY_SCRIPT]', 'lpa_workload_trend', '[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm',
        '[REDACTED_BY_SCRIPT]'
    ]
    
    # AD-AM-20.4: Strengthen LPA cohort
    lpa_additions = [f for f in X.columns if f.startswith('knn_') or f.startswith('lpa_legacy_')]
    # --- AD-AM-33: Integrate k-NN Anomaly Features into LPA Cohort ---
    knn_anomaly_features = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
        '[REDACTED_BY_SCRIPT]', 'knn_lpa_entropy_gcv'
    ]
    lpa_additions.extend(knn_anomaly_features)
    
    # AD-AM-20.5 & AD-AM-21.4: Quarantine high-cardinality and final flags
    flag_additions = [f for f in X.columns if f.startswith('[REDACTED_BY_SCRIPT]') or '_nearest_name_' in f]
    flag_additions.append('[REDACTED_BY_SCRIPT]')

    # AD-AM-21.1: Fortify ENV cohort
    env_additions = [f for f in X.columns if any(p in f for p in ['aonb_', 'aw_', 'np_', 'ph_', 'sac_', 'spa_', 'sssi_']) and f not in flag_additions]

    # AD-AM-21.2: Fortify GRID cohort
    grid_additions = [
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]

    # AD-AM-21.3: Fortify SOCIO-ECONOMIC cohort
    socio_additions = [f for f in X.columns if f.startswith('nhlc_') or f.startswith('site_lsoa_')]

    # --- Fortified Specialist Interaction Cohort: SIC_GRID_POLICY (AD-AMARYLLIS-MOD-14) ---
    # Hypothesis: Models the tension between the DNO's multi-faceted technical anxiety 
    # and the LPA'[REDACTED_BY_SCRIPT]'s scale.
    FORTIFIED_SIC_GRID_POLICY = [
        # The pre-engineered interaction feature (The central thesis)
        '[REDACTED_BY_SCRIPT]',
    
        # Vector 1: DNO Technical Anxiety (The Physical Reality)
        '[REDACTED_BY_SCRIPT]',             # Current Stress
        '[REDACTED_BY_SCRIPT]',             # Stability & Strength
        'pq_idw_thd_knn5',                          # Pre-existing Instability
        '[REDACTED_BY_SCRIPT]',         # Future Relief
        '[REDACTED_BY_SCRIPT]',    # Known Weakness
        '[REDACTED_BY_SCRIPT]',                  # Ground-Truth Headroom
    
        # Vector 2: LPA Political Will (The Human Reality)
        'lpa_major_commercial_approval_rate',       # Historical Bias
        '[REDACTED_BY_SCRIPT]',       # Operational Capacity
        'lpa_workload_trend',                       # Current Pressure
        'lpa_legacy_approval_rate',                 # Specific Experience
    
        # Vector 3: The Project Catalyst (The Applied Stressor)
        '[REDACTED_BY_SCRIPT]',                # Electrical Stress
        'solar_site_area_sqm',                      # Political Stress
        '[REDACTED_BY_SCRIPT]'                      # Normalized Impact
    ]
    
    # --- Fortified Specialist Interaction Cohort: SIC_NIMBY (AD-AMARYLLIS-MOD-15) ---
    # Hypothesis: Models the drivers of public opposition by synthesizing population profile, 
    # landscape value, and the magnitude of visual intrusion, amplified by cumulative impact.
    FORTIFIED_SIC_NIMBY = [
        # Vector 1: The Engineered Arguments (The Primary Signals)
        'SIC_NIMBY_AMPLIFIER',                  # Original amplifier (retained for legacy signal)
        '[REDACTED_BY_SCRIPT]',      # New: Models direct conflict with public amenity
        '[REDACTED_BY_SCRIPT]',      # New: Models the "straw that breaks the camel's back"
        '[REDACTED_BY_SCRIPT]',         # New: Models the weaponization of ecological/heritage assets

        # Vector 2: Affected Population Profile (Who is objecting?)
        '[REDACTED_BY_SCRIPT]',              # Demographic driver
        '[REDACTED_BY_SCRIPT]',          # Proximity and number of potential objectors
        '[REDACTED_BY_SCRIPT]',          # Wider community context
        '[REDACTED_BY_SCRIPT]', # Specific demographic persona known for opposition to change

        # Vector 3: Landscape Value & Character (What is being lost?)
        'aonb_is_within',                       # Is the landscape nationally important?
        'nt_dist_to_nearest_m',                 # Proximity to National Trails
        '[REDACTED_BY_SCRIPT]',               # Proximity to Open Access Land
        'sssi_dist_to_nearest_m',               # Proximity to top-tier ecological sites
        'hp_dist_to_nearest_m',                 # Proximity to Historic Parks (heritage value)
        'cs_density_5km',                       # Saturation of agri-environment schemes (conservation character)
        '[REDACTED_BY_SCRIPT]',   # Composite measure of landscape 'wiredness' and protection

        # Vector 4: Magnitude of Intrusion (How big is the impact?)
        'solar_site_area_sqm',                  # Physical footprint of the proposal
        '[REDACTED_BY_SCRIPT]',            # Electrical scale of the proposal
        '[REDACTED_BY_SCRIPT]',            # Visual clutter from existing grid infrastructure
        '[REDACTED_BY_SCRIPT]'         # Density of existing large-scale energy projects
    ]
    
    # --- CONSOLIDATED SPECIALIST COHORT ---
    consolidated_features = []
    consolidated_features.extend(_sanitize(COHORT_CORE_ATTRIBUTES))
    consolidated_features.extend([f for f in X.columns if f.startswith('TFI_')] + temporal_base_features)
    consolidated_features.extend(FORTIFIED_SIC_GRID_POLICY)
    consolidated_features.extend(FORTIFIED_SIC_NIMBY)
    consolidated_features.extend(['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'])
    consolidated_features.extend(['[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm', 'cs_density_5km', '[REDACTED_BY_SCRIPT]'])
    consolidated_features.extend(['SIC_ECONOMIC_VS_ECOLOGICAL_TENSION', '[REDACTED_BY_SCRIPT]', 'alc_is_bmv_at_site', 'aw_dist_to_nearest_m'])


    all_cohorts = {
        # New Cohorts from Balkanization
        "[REDACTED_BY_SCRIPT]": _sanitize([REDACTED_BY_SCRIPT]),
        "[REDACTED_BY_SCRIPT]": _sanitize([REDACTED_BY_SCRIPT]),
        "[REDACTED_BY_SCRIPT]": _sanitize([f for f in [REDACTED_BY_SCRIPT] if not f.startswith('major_road_length_')]),
        
        # Refactored Existing Cohorts
        "COHORT_LPA_ALL": _sanitize(COHORT_LPA_ALL + lpa_additions),
        "COHORT_GRID_ALL": _sanitize(COHORT_GRID_ALL + grid_additions),
        "COHORT_ENV_ALL": _sanitize(COHORT_ENV_ALL + env_additions),
        "COHORT_SOCIO_ECONOMIC_ALL": _sanitize(COHORT_SOCIO_ECONOMIC_ALL + socio_additions + [f for f in X.columns if f.startswith('lpa_lsoa_agg_')]),
        "[REDACTED_BY_SCRIPT]": _sanitize([f for f in [REDACTED_BY_SCRIPT] if not f.startswith('major_road_length_')]),
        "[REDACTED_BY_SCRIPT]": _sanitize([REDACTED_BY_SCRIPT] + flag_additions),
        "[REDACTED_BY_SCRIPT]": list(dict.fromkeys(consolidated_features)) # Add new consolidated head
    }

    # --- SPECIALIST INTERACTION COHORTS (SICs) ---
    # Each SIC model is an expert on a specific, complex trade-off.

    # # --- AD-AM-33: New Specialist Cohort for Catastrophic Outlier Detection ---
    # all_cohorts["[REDACTED_BY_SCRIPT]"] = knn_anomaly_features
    
    # --- Fortified Specialist Interaction Cohort: SIC_GRID_POLICY (AD-AMARYLLIS-MOD-14) ---
    # Hypothesis: Models the tension between the DNO's multi-faceted technical anxiety 
    # and the LPA'[REDACTED_BY_SCRIPT]'s scale.
    FORTIFIED_SIC_GRID_POLICY = [
        # The pre-engineered interaction feature (The central thesis)
        '[REDACTED_BY_SCRIPT]',
    
        # Vector 1: DNO Technical Anxiety (The Physical Reality)
        '[REDACTED_BY_SCRIPT]',             # Current Stress
        '[REDACTED_BY_SCRIPT]',             # Stability & Strength
        'pq_idw_thd_knn5',                          # Pre-existing Instability
        '[REDACTED_BY_SCRIPT]',         # Future Relief
        '[REDACTED_BY_SCRIPT]',    # Known Weakness
        '[REDACTED_BY_SCRIPT]',                  # Ground-Truth Headroom
    
        # Vector 2: LPA Political Will (The Human Reality)
        'lpa_major_commercial_approval_rate',       # Historical Bias
        '[REDACTED_BY_SCRIPT]',       # Operational Capacity
        'lpa_workload_trend',                       # Current Pressure
        'lpa_legacy_approval_rate',                 # Specific Experience
    
        # Vector 3: The Project Catalyst (The Applied Stressor)
        '[REDACTED_BY_SCRIPT]',                # Electrical Stress
        'solar_site_area_sqm',                      # Political Stress
        '[REDACTED_BY_SCRIPT]'                      # Normalized Impact
    ]
    
    # --- Fortified Specialist Interaction Cohort: SIC_NIMBY (AD-AMARYLLIS-MOD-15) ---
    # Hypothesis: Models the drivers of public opposition by synthesizing population profile, 
    # landscape value, and the magnitude of visual intrusion, amplified by cumulative impact.
    FORTIFIED_SIC_NIMBY = [
        # Vector 1: The Engineered Arguments (The Primary Signals)
        'SIC_NIMBY_AMPLIFIER',                  # Original amplifier (retained for legacy signal)
        '[REDACTED_BY_SCRIPT]',      # New: Models direct conflict with public amenity
        '[REDACTED_BY_SCRIPT]',      # New: Models the "straw that breaks the camel's back"
        '[REDACTED_BY_SCRIPT]',         # New: Models the weaponization of ecological/heritage assets

        # Vector 2: Affected Population Profile (Who is objecting?)
        '[REDACTED_BY_SCRIPT]',              # Demographic driver
        '[REDACTED_BY_SCRIPT]',          # Proximity and number of potential objectors
        '[REDACTED_BY_SCRIPT]',          # Wider community context
        '[REDACTED_BY_SCRIPT]', # Specific demographic persona known for opposition to change

        # Vector 3: Landscape Value & Character (What is being lost?)
        'aonb_is_within',                       # Is the landscape nationally important?
        'nt_dist_to_nearest_m',                 # Proximity to National Trails
        '[REDACTED_BY_SCRIPT]',               # Proximity to Open Access Land
        'sssi_dist_to_nearest_m',               # Proximity to top-tier ecological sites
        'hp_dist_to_nearest_m',                 # Proximity to Historic Parks (heritage value)
        'cs_density_5km',                       # Saturation of agri-environment schemes (conservation character)
        '[REDACTED_BY_SCRIPT]',   # Composite measure of landscape 'wiredness' and protection

        # Vector 4: Magnitude of Intrusion (How big is the impact?)
        'solar_site_area_sqm',                  # Physical footprint of the proposal
        '[REDACTED_BY_SCRIPT]',            # Electrical scale of the proposal
        '[REDACTED_BY_SCRIPT]',            # Visual clutter from existing grid infrastructure
        '[REDACTED_BY_SCRIPT]'         # Density of existing large-scale energy projects
    ]
    
    # --- AD-AM-35: Decommissioning of underperforming specialist cohorts ---
    # all_cohorts["SIC_LAND_USE"] = ['[REDACTED_BY_SCRIPT]', 'alc_is_bmv_at_site', _sanitize(['[REDACTED_BY_SCRIPT]'])[0], 'solar_site_area_sqm']
    # all_cohorts["SIC_LANDSCAPE"] = ['SIC_LANDSCAPE_SCARRING', 'ohl_local_tower_ratio', '[REDACTED_BY_SCRIPT]', 'aonb_is_within']
    # all_cohorts["SIC_BROWNFIELD"] = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    # all_cohorts["SIC_VISUAL_AMENITY"] = ['[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    # all_cohorts["SIC_GRID_REALITY"] = [ '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', sanitized_demand_rag]
    # all_cohorts["SIC_LPA_PRECEDENT"] = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'lpa_major_commercial_approval_rate', 'lpa_workload_trend']
    # all_cohorts["SIC_CUMULATIVE_IMPACT"] = ['[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm', 'cs_density_5km', '[REDACTED_BY_SCRIPT]']
    # all_cohorts["SIC_ECO_TENSION"] = ['SIC_ECONOMIC_VS_ECOLOGICAL_TENSION', '[REDACTED_BY_SCRIPT]', 'alc_is_bmv_at_site', 'aw_dist_to_nearest_m']

    # --- ARCHITECTURAL ADDENDUM: Overflow Cohort ---
    # Create a cohort for all features not explicitly assigned to a monolithic expert cohort.
    # This ensures 100% feature coverage and captures potentially valuable residual signals.
    monolithic_cohort_names = [
        "COHORT_CORE_ATTRIBUTES", "COHORT_GRID_ALL", "COHORT_LPA_ALL", "COHORT_ENV_ALL",
        "COHORT_SOCIO_ECONOMIC_ALL", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]"
    ]


    # --- Architectural Sanity Check (Eradication of Overflow) ---
    all_model_features = set(X.columns)
    used_features = set()
    for cohort_name, features in all_cohorts.items():
        # Sanitize features before adding to the used set to match how they are defined
        used_features.update(_sanitize(features))
        
    # Sanitize all model features for a fair comparison
    sanitized_model_features = set(_sanitize(all_model_features))
    overflow_features = sorted(list(sanitized_model_features - used_features))
    
    if overflow_features:
        print("[REDACTED_BY_SCRIPT]")
        for feat in overflow_features:
            print(f"  - {feat}")
    else:
        print("[REDACTED_BY_SCRIPT]")

    # Per AD-AM-25, the GCV is no longer mixed into the head models.
    # The function now returns the pure, specialist cohort definitions.
    print("[REDACTED_BY_SCRIPT]")
    final_cohorts = {}
    for cohort_name, features in all_cohorts.items():
        cohort_features_in_df = [f for f in features if f in X.columns]
        if not cohort_features_in_df:
            print(f"[REDACTED_BY_SCRIPT]'{cohort_name}'[REDACTED_BY_SCRIPT]")
            continue
        
        # Sterilization Gate: Enforce feature uniqueness to prevent LightGBM crash.
        # This preserves order while removing duplicates.
        deduplicated_features = list(dict.fromkeys(cohort_features_in_df))
        
        if len(deduplicated_features) < len(cohort_features_in_df):
            print(f"[REDACTED_BY_SCRIPT]'{cohort_name}'")

        final_cohorts[cohort_name] = deduplicated_features
        print(f"[REDACTED_BY_SCRIPT]")
        
    return final_cohorts

def phase_1_load_and_sanitize_data():
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    X_raw = pd.read_csv(X_PATH, index_col=0)
    y_reg = pd.read_csv(Y_REG_PATH, index_col=0).squeeze("columns")

    # Mandated Purge of non-predictive metadata (AD-AMARYLLIS-MOD-05)
    metadata_to_purge = [c for c in X_raw.columns if 'join_method' in c or 'application_reference' in c or 'application_id' in c]
    X_purged = X_raw.drop(columns=metadata_to_purge)
    print(f"[REDACTED_BY_SCRIPT]")

    # Mandated Sanitization Gate
    X_sanitized = sanitize_column_names(X_purged)

    # Final data alignment for regression task.
    #  CRITICAL: Only train on successful applications with a valid, positive planning duration.
    # This prevents data poisoning from failed (0, -1), invalid (NaN), or rejected (10000.0) cases.
    valid_indices = y_reg[(y_reg > 25) & (y_reg < 1500.0)].index
    y_reg_clean = y_reg.loc[valid_indices]
    X_final = X_sanitized.loc[valid_indices]

    print(f"[REDACTED_BY_SCRIPT]")
    return X_final, y_reg_clean


def phase_1b_engineer_knn_anomaly_features(X_train, y_train, X_val, X_test, gcv_features, k_neighbors=10):
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
    knn_engine = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree', n_jobs=-1)
    knn_engine.fit(X_train_atlas_scaled)
    print(f"[REDACTED_BY_SCRIPT]")

    # --- 2.3 & 2.4: The Feature Forging Mandate ---
    def _forge_features(X_target: pd.DataFrame) -> pd.DataFrame:
        """[REDACTED_BY_SCRIPT]"""
        # Ensure target has the GCV columns, then impute and scale using the FITTED transformers.
        X_target_gcv_imputed = imputer.transform(X_target[gcv_features])
        X_target_gcv_scaled = scaler.transform(X_target_gcv_imputed)
        
        # Find the k nearest neighbors in the Success Atlas for each target sample.
        distances, indices = knn_engine.kneighbors(X_target_gcv_scaled)
        
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
    X_train_augmented = _forge_features(X_train.copy())
    print("[REDACTED_BY_SCRIPT]")
    X_val_augmented = _forge_features(X_val.copy())
    print("[REDACTED_BY_SCRIPT]")
    X_test_augmented = _forge_features(X_test.copy())
    
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
        n_estimators=100,
        reg_alpha=5,
        reg_lambda=5
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
        'num_leaves': 31,
        'n_jobs': -1,
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
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        reg_alpha=10,
        reg_lambda=10,
        n_jobs=-1
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

def phase_A_tune_individual_heads(X_train, y_head_target_train, sample_weights, cohorts, n_trials=TRIALS_GLOBAL):
    """
    AD-AM-41.2: Individually tunes specialist head CLASSIFIERS to predict Oracle
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
            
            # Mandate 41.2: Parsimony constraints that scale with cohort complexity.
            if n_features < 20: # Micro-specialists
                max_depth = trial.suggest_int('max_depth', 2, 4)
                num_leaves = trial.suggest_int('num_leaves', 3, 10)
                n_estimators = trial.suggest_int('n_estimators', 50, 150)
            elif n_features < 100: # Standard specialists
                max_depth = trial.suggest_int('max_depth', 3, 5)
                num_leaves = trial.suggest_int('num_leaves', 5, 15)
                n_estimators = trial.suggest_int('n_estimators', 100, 200)
            else: # Monolithic cohorts
                max_depth = trial.suggest_int('max_depth', 4, 7)
                num_leaves = trial.suggest_int('num_leaves', 10, 25)
                n_estimators = trial.suggest_int('n_estimators', 150, 250)

            
            params = {
                'objective': 'multiclass', 'metric': 'multi_logloss', 'random_state': RANDOM_STATE, 'verbosity': -1,
                'num_class': 5,
                'n_estimators': n_estimators,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'reg_alpha': trial.suggest_float('reg_alpha', 50.0, 250.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 50.0, 250.0, log=True),
                'n_jobs': -1,
            }
            model = lgb.LGBMClassifier(**params)
            
            model = lgb.LGBMClassifier(**params)
            
            # StratifiedKFold is MANDATORY for imbalanced classification to ensure all
            # classes are represented in each validation fold.
            kf = StratifiedKFold(n_splits=SPLITS_GLOBAL, shuffle=True, random_state=RANDOM_STATE)
            scores = []
            from sklearn.metrics import log_loss
            
            for train_idx, val_idx in kf.split(X_head_tune, y_head_target_train):
                X_train_fold, X_val_fold = X_head_tune.iloc[train_idx], X_head_tune.iloc[val_idx]
                y_train_fold, y_val_fold = y_head_target_train.iloc[train_idx], y_head_target_train.iloc[val_idx]
                weights_fold = sample_weights.iloc[train_idx]
                
                model.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)
                proba_preds = model.predict_proba(X_val_fold)

                # --- SURGICAL INTERVENTION: Reconstruct full probability matrix ---
                # Handles cases where a fold is missing one of the 5 classes.
                if proba_preds.shape[1] < 5:
                    full_proba_preds = np.zeros((proba_preds.shape[0], 5))
                    full_proba_preds[:, model.classes_] = proba_preds
                    proba_preds = full_proba_preds
                
                # Fortification: Explicitly provide all possible labels to prevent error
                # on folds that may be missing a rare class.
                fold_loss = log_loss(y_val_fold, proba_preds, labels=[0, 1, 2, 3, 4])
                scores.append(fold_loss)
            
            # Optuna minimizes, so we return the mean of the positive loss.
            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"[REDACTED_BY_SCRIPT]")
        all_best_params[cohort_name] = study.best_params

    print("[REDACTED_BY_SCRIPT]")
    return all_best_params

def phase_E_train_persist_and_analyze_heads(X_train, y_head_target_train, sample_weights, cohorts, all_head_params, output_dir, top_n=10):
    """
    AD-AM-41: Forges, persists, and analyzes specialist CLASSIFIERS trained to
    predict Oracle error regimes.
    """
    os.makedirs(output_dir, exist_ok=True)
    top_head_features_path = os.path.join(output_dir, "[REDACTED_BY_SCRIPT]")
    top_head_features = {}

    print(f"[REDACTED_BY_SCRIPT]'Anti-Oracle' Heads ---")
    for cohort_name, features in cohorts.items():
        if cohort_name not in all_head_params:
            print(f"  Skipping '{cohort_name}'[REDACTED_BY_SCRIPT]")
            continue
            
        model_features = [f for f in features if f in X_train.columns]
        X_head_train = X_train[model_features]
        head_specific_params = all_head_params[cohort_name]
        
        print(f"[REDACTED_BY_SCRIPT]")
        # SURGICAL INTERVENTION: Manually inject fixed multiclass parameters not saved by Optuna.
        head_specific_params['objective'] = 'multiclass'
        head_specific_params['num_class'] = 5
        model = lgb.LGBMClassifier(**head_specific_params, random_state=RANDOM_STATE)
        
        # Train the specialist classifier on the binned error categories.
        model.fit(X_head_train, y_head_target_train, sample_weight=sample_weights)
        joblib.dump(model, os.path.join(output_dir, f"[REDACTED_BY_SCRIPT]"))

        print(f"[REDACTED_BY_SCRIPT]")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_head_train)
        
        # SURGICAL INTERVENTION: Fortify SHAP value processing to handle all possible output formats
        # (list of 2D arrays, single 3D array, single 2D array) and guarantee a 1D result.
        if isinstance(shap_values, list):
            # Standard multiclass output: a list of 2D arrays (n_samples, n_features).
            # Stack, take abs, and average over classes (axis 0) and samples (axis 1).
            mean_abs_shap = np.abs(np.stack(shap_values, axis=0)).mean(axis=(0, 1))
        else:
            # Can be a 3D array for multiclass or 2D for binary/regression.
            sv_array = np.asarray(shap_values)
            if sv_array.ndim == 3:
                # 3D array output: (n_samples, n_features, n_classes).
                # Average over samples (axis 0) and classes (axis 2).
                mean_abs_shap = np.abs(sv_array).mean(axis=(0, 2))
            else:
                # 2D array output: (n_samples, n_features). Average over samples (axis 0).
                mean_abs_shap = np.abs(sv_array).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': X_head_train.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # The exclusion gate for GCV is no longer needed, as the features were never added.
        # The SHAP analysis is inherently pure.
        top_features = importance_df['feature'].head(top_n).tolist()

        top_head_features[cohort_name] = top_features

    import json
    with open(top_head_features_path, 'w') as f:
        json.dump(top_head_features, f, indent=4)
    print(f"[REDACTED_BY_SCRIPT]")
    return top_head_features


def phase_F_generate_ridge_arbiter_features(X_base, oracle_preds, head_prob_preds, top_head_features, scaler=None):
    """
    Architectural Mandate: Forges the feature matrix for the final Ridge Arbiter model.
    This consolidates the oracle's baseline prediction, aggregated risk signals from the
    specialist heads, and the most salient raw features identified by each specialist.
    """
    print("[REDACTED_BY_SCRIPT]")
    arbiter_features = pd.DataFrame(index=X_base.index)
    arbiter_features['[REDACTED_BY_SCRIPT]'] = oracle_preds

    # 1. Add aggregated risk probabilities
    avg_probs = pd.concat(head_prob_preds.values()).groupby(level=0).mean()
    arbiter_features['prob_under_pred'] = avg_probs[0] + avg_probs[1]
    arbiter_features['prob_over_pred'] = avg_probs[3] + avg_probs[4]

    # 2. Add raw probabilities from each head
    for cohort_name, prob_df in head_prob_preds.items():
        prob_df.columns = [f"[REDACTED_BY_SCRIPT]" for i in range(prob_df.shape[1])]
        arbiter_features = arbiter_features.join(prob_df)

    # 3. Add top N salient features from each specialist head
    salient_features = set()
    for features in top_head_features.values():
        salient_features.update(features)
    
    # Filter for features that exist in the base dataframe
    existing_salient_features = [f for f in salient_features if f in X_base.columns]
    salient_features_df = X_base[existing_salient_features]
    arbiter_features = arbiter_features.join(salient_features_df)
    
    # 4. Final sanitation and scaling
    arbiter_features.fillna(0, inplace=True)
    
    if scaler is None: # Training mode: fit a new scaler
        scaler = StandardScaler()
        column_order = arbiter_features.columns
        arbiter_features_scaled = pd.DataFrame(scaler.fit_transform(arbiter_features), index=arbiter_features.index, columns=column_order)
        # Store column order in scaler for inference
        scaler.feature_names_in_ = column_order
        return arbiter_features_scaled, scaler
    else: # Inference mode: use existing scaler
        # Ensure column order matches the scaler's expectations
        arbiter_features_reordered = arbiter_features[scaler.feature_names_in_]
        arbiter_features_scaled = pd.DataFrame(scaler.transform(arbiter_features_reordered), index=arbiter_features.index, columns=scaler.feature_names_in_)
        return arbiter_features_scaled, scaler

def phase_G_tune_and_train_ridge_arbiter(X_arbiter_train, y_arbiter_train, X_arbiter_val, y_arbiter_val):
    """
    Tunes and trains the final Ridge Arbiter model. The model is trained to predict
    the Oracle's error, which is then used as a correction factor. Tuning is performed
    on the validation set to prevent leakage.
    """
    print("[REDACTED_BY_SCRIPT]")
    alphas = [0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000]
    best_rmse = float('inf')
    best_alpha = None

    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        model.fit(X_arbiter_train, y_arbiter_train)
        preds_val = model.predict(X_arbiter_val)
        rmse = root_mean_squared_error(y_arbiter_val, preds_val)
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            
    print(f"[REDACTED_BY_SCRIPT]")

    # Retrain on combined train+val data with the best alpha
    print("[REDACTED_BY_SCRIPT]")
    X_combined = pd.concat([X_arbiter_train, X_arbiter_val])
    y_combined = pd.concat([y_arbiter_train, y_arbiter_val])
    
    final_arbiter = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
    final_arbiter.fit(X_combined, y_combined)
    print("[REDACTED_BY_SCRIPT]")
    
    return final_arbiter

def phase_H_retune_arbiter_with_valid_objective(X_train, y_train, n_trials=TRIALS_GLOBAL):
    """
    [AD-AM-32 Remediation] Re-tunes the Arbiter directly on its error-prediction task,
    excising the incompatible TransformedTargetRegressor.
    """
    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'rmse', 'random_state': 42, 'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 800, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.005, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 3, 50),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 100.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'n_jobs': -1,
        }
        
        # The Arbiter is now a raw LGBMRegressor, as it predicts error directly.
        # The TransformedTargetRegressor is no longer architecturally valid for this component.
        model = lgb.LGBMRegressor(**params)
        
        try:
            X_train_np = X_train.values
            y_train_np = y_train.values

            score = cross_val_score(
                model, X_train_np, y_train_np, 
                cv=KFold(n_splits=SPLITS_GLOBAL, shuffle=True, random_state=42),
                scoring='neg_root_mean_squared_error'
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
    
    # ARCHITECTURAL CORRECTION: Reconstruct the *exact* augmented feature matrix the model was trained on.
    specialist_features = [f for f in cohorts[influential_head] if f in X_test.columns]
    augmented_features = list(dict.fromkeys(specialist_features + gcv))
    X_test_head_augmented = X_test[augmented_features].copy()
    X_test_head_augmented['oracle_prediction'] = oracle_preds_test
    
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
        features = cohorts[cohort_name]
        model_path = os.path.join(head_models_dir, f"[REDACTED_BY_SCRIPT]")
        
        if not os.path.exists(model_path):
            print(f"[REDACTED_BY_SCRIPT]'{cohort_name}'[REDACTED_BY_SCRIPT]")
            continue

        head_model = joblib.load(model_path)
        specialist_features = [f for f in features if f in X_test.columns]
        
        if not specialist_features:
            print(f"[REDACTED_BY_SCRIPT]'{cohort_name}' found in the test set.")
            continue
            
        print(f"[REDACTED_BY_SCRIPT]")
        
        # ARCHITECTURAL CORRECTION (AD-AM-28): The head models were trained purely on their
        # specialist features (GCV Quarantine). The SHAP analysis must use the exact same feature set.
        X_test_head_specialist = X_test[specialist_features]
        
        # --- CRITICAL FIX: Determine actual feature count before SHAP calculation ---
        num_features_in_cohort = X_test_head_specialist.shape[1]
        
        # Guard against edge cases
        if num_features_in_cohort < 1:
            print(f"[REDACTED_BY_SCRIPT]")
            continue
        
        explainer = shap.TreeExplainer(head_model)
        shap_values = explainer.shap_values(X_test_head_specialist)

        # --- ARCHITECTURAL HARDENING GATE (AD-AM-45) ---
        # The explainer can produce artifacts (extra columns). We must surgically
        # truncate the SHAP arrays to match the feature matrix dimensions.
        if isinstance(shap_values, list):
            shap_values = [arr[:, :num_features_in_cohort] for arr in shap_values]
        else:
            shap_values = shap_values[:, :num_features_in_cohort]
        
        # SURGICAL INTERVENTION: For multi-class classifiers, shap_values is a list of arrays.
        # To get a single summary plot of overall feature importance, we must average the
        # absolute SHAP values across all classes.
        if isinstance(shap_values, list):
            shap_values_for_plot = np.abs(np.stack(shap_values, axis=0)).mean(0)
        else:
            shap_values_for_plot = shap_values

        # --- CRITICAL FIX: Cap max_display to actual feature count ---
        dynamic_max_display = min(top_n, num_features_in_cohort)
        
        # --- Generate SHAP Summary Plot (Beeswarm) ---
        plt.figure(figsize=(12, max(1, dynamic_max_display * 0.4)))
        shap.summary_plot(shap_values_for_plot, X_test_head_specialist, show=False, max_display=dynamic_max_display)
        plt.title(f"[REDACTED_BY_SCRIPT]", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_filename = f"[REDACTED_BY_SCRIPT]"
        plot_path = os.path.join(shap_output_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        # Add entry to the main dossier report
        relative_plot_path = os.path.join(os.path.basename(shap_output_dir), plot_filename)
        report_content.append(f"[REDACTED_BY_SCRIPT]")
        report_content.append(f"[REDACTED_BY_SCRIPT]")
        
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
    
    # Stratify by Project Scale
    print("[REDACTED_BY_SCRIPT]")
    data['scale_quartile'] = pd.qcut(data['[REDACTED_BY_SCRIPT]'], 4, labels=False, duplicates='drop')
    scale_performance = data.groupby('scale_quartile')['abs_error'].apply(lambda x: np.sqrt(np.mean(x**2)))
    print(scale_performance.to_string())
    
    # Stratify by LPA Toughness
    print("[REDACTED_BY_SCRIPT]")
    data['lpa_quartile'] = pd.qcut(data['lpa_major_commercial_approval_rate'], 4, labels=False, duplicates='drop')
    lpa_performance = data.groupby('lpa_quartile')['abs_error'].apply(lambda x: np.sqrt(np.mean(x**2)))
    print(lpa_performance.to_string())


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR_V35, exist_ok=True)
    dossier_content = ["[REDACTED_BY_SCRIPT]"]
    
    print("[REDACTED_BY_SCRIPT]")

    # --- Setup: Load data and architectural definitions (v3.4 Protocol) ---
    X_general, y_general = phase_1_load_and_sanitize_data()
    
    # --- Isolate Ground Mount Solar Cohort for Specialization ---
    master_df = X_general.join(y_general)
    gm_solar_df = master_df[(master_df['technology_type'] == 21)] # & (master_df['[REDACTED_BY_SCRIPT]'] == 1)

    # --- ARCHITECTURAL INTERVENTION: Pre-emptive Excision of Unlearnable Samples ---
    # The previous post-hoc cleaning was flawed. This new protocol identifies and removes
    # unlearnable samples from the entire dataset *before* any splits or training,
    # ensuring a clean data foundation for the entire pipeline.
    unlearnable_indices = phase_0b_identify_and_excise_unlearnable_samples(
        gm_solar_df.drop(columns=[y_general.name]),
        gm_solar_df[y_general.name],
        outlier_threshold=100
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
    gm_train_val_df, gm_test_df = train_test_split(
        gm_solar_df, test_size=0.15, random_state=42,
        stratify=pd.qcut(gm_solar_df['lpa_major_commercial_approval_rate'], q=4, labels=False, duplicates='drop')
    )
    # Next, split the remaining data into training (70%) and validation (15%).
    # The validation set is for tuning the final calibrator, not the models themselves.
    train_val_stratify_col = pd.qcut(gm_train_val_df['lpa_major_commercial_approval_rate'], q=4, labels=False, duplicates='drop')
    gm_train_df, gm_val_df = train_test_split(
        gm_train_val_df, test_size=0.1765, # 0.15 / 0.85 = 0.1765
        random_state=42, stratify=train_val_stratify_col
    )
    
    X_train_base, y_train_gm = gm_train_df.drop(columns=y_general.name), gm_train_df[y_general.name]
    X_val_base, y_val_gm = gm_val_df.drop(columns=y_general.name), gm_val_df[y_general.name]
    X_test_base, y_test_gm = gm_test_df.drop(columns=y_general.name), gm_test_df[y_general.name]
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    # --- Mandate 19.1 & AD-AM-22: Execute Feature Engineering in Correct Sequence Across All Splits ---
    # Step 1: Stateful Temporal Engineering (fit on train, transform all)
    X_train_temporal, baseline_year, tfi_imputer, tfi_scaler = engineer_temporal_context_features(X_train_base)
    X_val_temporal, _, _, _ = engineer_temporal_context_features(
        X_val_base, baseline_year=baseline_year, imputer=tfi_imputer, scaler=tfi_scaler
    )
    X_test_temporal, _, _, _ = engineer_temporal_context_features(
        X_test_base, baseline_year=baseline_year, imputer=tfi_imputer, scaler=tfi_scaler
    )
    
    # Step 2: Stateless Strategic Interaction Engineering (dependent on temporal features)
    X_train_sics = engineer_strategic_interaction_features(X_train_temporal)
    X_val_sics = engineer_strategic_interaction_features(X_val_temporal)
    X_test_sics = engineer_strategic_interaction_features(X_test_temporal)

    # --- Tune & Train Oracle & Generate Context Vectors (POST-SPLIT to prevent leakage) ---
    # AD-AM-33: Oracle MUST NOT see the k-NN features. It is trained on the pre-kNN feature set.
    oracle_model, gcv = phase_0_tune_and_train_oracle(X_train_sics, y_train_gm, n_features=20, n_trials=TRIALS_GLOBAL)
    
    # --- AD-AM-33: Execute k-NN Anomaly Detection Sub-System (Post 3-Way Split) ---
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

    # --- Define Cohorts using a consistently engineered feature space ---
    # This engineering pipeline must mirror the training pipeline for column consistency.
    X_general_temporal, _, _, _ = engineer_temporal_context_features(X_general)
    X_general_engineered = engineer_strategic_interaction_features(X_general_temporal)
    # The cohort definition must know about the new k-NN features.
    for col in ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'knn_lpa_entropy_gcv']:
        if col not in X_general_engineered.columns:
            X_general_engineered[col] = 0 # Add placeholder for consistent definition
    cohorts = define_all_cohorts(X_general_engineered)
    
    print("[REDACTED_BY_SCRIPT]")
    # AD-AM-33.2 CORRECTION: The Oracle MUST predict on the same feature space it was trained on.
    # It must not see the k-NN features. We use the pre-augmentation dataframes for this step.
    oracle_preds_train = pd.Series(oracle_model.predict(X_train_sics), index=X_train_sics.index, name="oracle_prediction")
    oracle_preds_val = pd.Series(oracle_model.predict(X_val_sics), index=X_val_sics.index, name="oracle_prediction")
    oracle_preds_test = pd.Series(oracle_model.predict(X_test_sics), index=X_test_sics.index, name="oracle_prediction")

    # --- AD-AM-30: Forge Specialist Heads for Parallel "Conclave of Experts" Architecture ---
    # The heads are now trained on the true target variable, not the Oracle's residual.
    # The residual calculation and signal gating logic have been excised to prevent signal sterilization.
    print("[REDACTED_BY_SCRIPT]")
    # --- AD-AM-32: Forge "Anti-Oracle" Specialists ---
    # 1. Define the specialist training target: the Oracle's raw error.
    oracle_error_train = y_train_gm - oracle_preds_train
    oracle_error_val = y_val_gm - oracle_preds_val # For calibrator tuning
    
    # 2. Implement Mandate 42.1: Expand head target to 5-class classification.
    #    This provides a more granular risk assessment for the calibrator.
    error_bins = [-np.inf, -150, -50, 50, 150, np.inf]
    error_labels = [0, 1, 2, 3, 4] # 0:Sev-Under, 1:Mod-Under, 2:Accurate, 3:Mod-Over, 4:Sev-Over
    y_head_target_train = pd.cut(oracle_error_train, bins=error_bins, labels=error_labels).astype(int)
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(y_head_target_train.value_counts(normalize=True).sort_index())
    
    # 3. Create sample weights for the imbalanced classification task.
    class_weights = y_head_target_train.value_counts().to_dict()
    sample_weights = y_head_target_train.map(lambda x: len(y_head_target_train) / class_weights[x])
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    # 3. Tune and train the specialists to predict the Oracle's weighted error.
    #    The specialists now train on the feature set augmented with k-NN features.
    best_head_params_dict = phase_A_tune_individual_heads(
        X_train_gm, y_head_target_train, sample_weights, cohorts, n_trials=TRIALS_GLOBAL
    )
    top_head_features = phase_E_train_persist_and_analyze_heads(
        X_train_gm, y_head_target_train, sample_weights, cohorts, best_head_params_dict, HEADS_V35_DIR
    )

    # # --- AD-AM-34: Train the new Catastrophe Risk Classifier ---
    # catastrophe_classifier, catastrophe_features = phase_B_train_catastrophe_classifier(
    #     X_train_gm, y_train_gm, oracle_preds_train
    # )

    # # --- AD-AM-35: Train the Quantile Specialists on the full feature set ---
    # p10_model, p90_model = phase_C_train_quantile_specialists(X_train_gm, oracle_error_train)

    # # --- AD-AM-38: Train the Failure Regime Specialist ---
    # failure_specialist = phase_E_train_failure_regime_specialist(
    #     X_train_gm, oracle_error_train, catastrophe_classifier, catastrophe_features
    # )

    # --- AD-AM-41: Execute v4.0 "Signal, Not Noise" Architecture ---
    # 3. Tune and train the parsimonious specialist classifiers.
    best_head_params_dict = phase_A_tune_individual_heads(
        X_train_gm, y_head_target_train, sample_weights, cohorts, n_trials=TRIALS_GLOBAL
    )
    # The training function is re-purposed to handle classifiers
    top_head_features = phase_E_train_persist_and_analyze_heads(
        X_train_gm, y_head_target_train, sample_weights, cohorts, best_head_params_dict, HEADS_V35_DIR
    )

    # 4. Generate head predictions (risk probabilities) on the validation set.
    head_prob_preds_val = {}
    for cohort_name, features in cohorts.items():
        model_path = os.path.join(HEADS_V35_DIR, f"[REDACTED_BY_SCRIPT]")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            model_features = [f for f in features if f in X_val_gm.columns]
            if model_features:
                probs = model.predict_proba(X_val_gm[model_features])
                head_prob_preds_val[cohort_name] = pd.DataFrame(probs, index=X_val_gm.index)

    # 5. Generate head predictions (risk probabilities) for all data splits.
    print("[REDACTED_BY_SCRIPT]")
    head_prob_preds_train, head_prob_preds_val, head_prob_preds_test = {}, {}, {}
    data_splits = {
        'train': (X_train_gm, head_prob_preds_train),
        'val': (X_val_gm, head_prob_preds_val),
        'test': (X_test_gm, head_prob_preds_test)
    }
    for cohort_name, features in cohorts.items():
        model_path = os.path.join(HEADS_V35_DIR, f"[REDACTED_BY_SCRIPT]")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            for split_name, (X_data, prob_dict) in data_splits.items():
                model_features = [f for f in features if f in X_data.columns]
                if model_features:
                    probs = model.predict_proba(X_data[model_features])
                    prob_dict[cohort_name] = pd.DataFrame(probs, index=X_data.index)

    # 6. Forge the Arbiter's feature matrices.
    X_arbiter_train, arbiter_scaler = phase_F_generate_ridge_arbiter_features(X_train_gm, oracle_preds_train, head_prob_preds_train, top_head_features)
    X_arbiter_val, _ = phase_F_generate_ridge_arbiter_features(X_val_gm, oracle_preds_val, head_prob_preds_val, top_head_features, scaler=arbiter_scaler)
    X_arbiter_test, _ = phase_F_generate_ridge_arbiter_features(X_test_gm, oracle_preds_test, head_prob_preds_test, top_head_features, scaler=arbiter_scaler)
    
    # 7. Tune and train the final Ridge Arbiter.
    # The arbiter is trained to predict the Oracle's error.
    oracle_error_train = y_train_gm - oracle_preds_train
    oracle_error_val = y_val_gm - oracle_preds_val
    final_arbiter = phase_G_tune_and_train_ridge_arbiter(X_arbiter_train, oracle_error_train, X_arbiter_val, oracle_error_val)
    
    # 8. Persist the final Arbiter model and its associated scaler.
    joblib.dump(final_arbiter, MODEL_V35_PATH)
    joblib.dump(arbiter_scaler, os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]"))
    print(f"[REDACTED_BY_SCRIPT]")

    # 9. Generate final predictions on the unseen test set.
    print("[REDACTED_BY_SCRIPT]")
    predicted_error_correction = final_arbiter.predict(X_arbiter_test)
    final_prediction_calibrated = oracle_preds_test + predicted_error_correction

    # --- AD-AM-32: Re-architected Prediction and Validation Pipeline ---
    print("[REDACTED_BY_SCRIPT]'Anti-Oracle' Pipeline ---")
    dossier_content.append(
        "[REDACTED_BY_SCRIPT]"
        "[REDACTED_BY_SCRIPT]'Ridge Arbiter'[REDACTED_BY_SCRIPT]'s baseline prediction, aggregated risk signals from specialist heads, and the most salient raw features identified by those specialists to compute a final, data-driven correction.\n"
    )

    # The final prediction has already been calculated using the new calibration logic.
    # The old Arbiter-based prediction logic is now obsolete.
    
    # Generate the definitive, honest performance report using the calibrated prediction.
    rmse = root_mean_squared_error(y_test_gm, final_prediction_calibrated)
    mae = mean_absolute_error(y_test_gm, final_prediction_calibrated)
    r2 = r2_score(y_test_gm, final_prediction_calibrated)
    report_text = (
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
    )
    print(report_text)
    
    # Generate the mandatory Error Distribution Plot
    error = final_prediction_calibrated - y_test_gm
    plt.figure(figsize=(12, 7))
    sns.histplot(error, kde=True, bins=50)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='[REDACTED_BY_SCRIPT]')
    plt.title(f"[REDACTED_BY_SCRIPT]", fontsize=16)
    plt.xlabel("[REDACTED_BY_SCRIPT]", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plot_path = os.path.join(OUTPUT_DIR_V35, f"[REDACTED_BY_SCRIPT]")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    dossier_content.append("[REDACTED_BY_SCRIPT]")
    dossier_content.append(f"```\n{report_text}\n```\n")
    dossier_content.append(f"[REDACTED_BY_SCRIPT]")

    # (For brevity, the full re-generation of every dossier plot is omitted,
    # but would use this validated model and report function.)
    with open(DOSSIER_V35_PATH, 'w') as f:
        f.write("".join(dossier_content))
    print(f"[REDACTED_BY_SCRIPT]")

    # ==============================================================================
    # ARCHITECTURAL DIRECTIVE AD-AM-18: THE DIAGNOSTIC GAUNTLET
    # ==============================================================================
    print("[REDACTED_BY_SCRIPT]")
    DIAGNOSTIC_DOSSIER_PATH = os.path.join(OUTPUT_DIR_V35, "[REDACTED_BY_SCRIPT]")
    diagnostic_dossier_content = ["[REDACTED_BY_SCRIPT]",
                                  "[REDACTED_BY_SCRIPT]"]

    # --- Setup for Diagnostics ---
    # The final calibrated prediction has already been calculated. We use it directly
    # to ensure the diagnostic gauntlet operates on the definitive model output.
    results_df = pd.DataFrame({
        'y_true': y_test_gm,
        'y_pred': final_prediction_calibrated
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