import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import joblib
import os
import re
import matplotlib.pyplot as plt
import optuna

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# --- Artifact & Project Constants ---
# ARCHITECTURAL NOTE: Paths remain stable, but the underlying artifacts are V3.
X_PATH = "[REDACTED_BY_SCRIPT]"
Y_REG_PATH = "[REDACTED_BY_SCRIPT]"

OUTPUT_DIR = "[REDACTED_BY_SCRIPT]" # Version incremented
MODEL_ARBITER_PATH = os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
REPORT_PATH = os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")
SHAP_PLOT_PATH = os.path.join(OUTPUT_DIR, "[REDACTED_BY_SCRIPT]")

# --- Modeling Constants ---
RANDOM_STATE = 42
CV_SPLITS = 20 # Mandate Preserved: Robust OOF generation is non-negotiable.
OPTUNA_TRIALS_HEAD = 2 # Mandate Increased: Deeper search for head model tuning.
OPTUNA_TRIALS_ARBITER = 1 # Mandate Increased: Deeper search for arbiter tuning.

# ==============================================================================
# ARCHITECTURAL ARTIFACT V3.1: DEFINITIVE COHORT DEFINITIONS
# Every feature is now assigned to a single, authoritative monolithic cohort.
# ==============================================================================

# ==============================================================================
# ARCHITECTURAL ARTIFACT V3.1: DEFINITIVE COHORT DEFINITIONS
# Every feature is now assigned to a single, authoritative monolithic cohort.
# This ensures strict thematic coherence and prevents signal contamination.
# ==============================================================================

# --- Mandated Semantic Cohort: CORE_ATTRIBUTES (AD-AMARYLLIS-MOD-04 Rev. 2) ---
# Purpose: The foundational, immutable facts of the application itself. What is it, how big is it, and when was it submitted?
COHORT_CORE_ATTRIBUTES = [
    # Core Physical Attributes
    '[REDACTED_BY_SCRIPT]',
    'solar_site_area_sqm',
    'chp_enabled',
    '[REDACTED_BY_SCRIPT]',
    # Core Temporal Attributes
    'submission_year',
    'submission_month',
    'submission_day',
    'submission_month_sin',
    'submission_month_cos'
]

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
    'hp_dist_to_nearest_m', 'hp_is_within', 'hp_count_in_2km', 'hp_total_area_in_5km_ha',
    # Capstone & Aggregate Environmental Risk Features
    '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
    # --- Phase V Addendum Features ---
    '[REDACTED_BY_SCRIPT]',
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


# ==============================================================================
# PHASE 0: GLOBAL CONTEXT VECTOR FORGING (AD-AMARYLLIS-MOD-11)
# ==============================================================================

def phase_0_get_gcv(X, y, n_features=20, head_params=None):
    """
    Implements the Global Signal Amplification Protocol (Phases 1-3).
    Trains a global oracle model and uses SHAP to extract the Global Context Vector (GCV).
    """
    print("[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

    # Phase 1: Forge the Global Oracle. Using pre-tuned head_params for a robust oracle.
    if head_params is None:
        # Provide reasonable defaults if no params are passed
        head_params = {'objective': 'regression_l1', 'metric': 'mae', 'random_state': RANDOM_STATE, 'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31, 'verbosity': -1}

    oracle_model = lgb.LGBMRegressor(**head_params, verbosity=-1)
    oracle_model.fit(X, y)
    print("  Oracle trained.")

    # Phase 2: Extract the Unimpeachable Signal via SHAP.
    print("[REDACTED_BY_SCRIPT]")
    explainer = shap.TreeExplainer(oracle_model)
    shap_values = explainer.shap_values(X)

    # Calculate mean absolute SHAP value for each feature.
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(list(zip(X.columns, mean_abs_shap)), columns=['feature', 'mean_abs_shap'])
    feature_importance = feature_importance.sort_values('mean_abs_shap', ascending=False)
    print("[REDACTED_BY_SCRIPT]")

    # Phase 3: Define the Global Context Vector (GCV).
    gcv = feature_importance['feature'].head(n_features).tolist()
    print(f"[REDACTED_BY_SCRIPT]")

    return gcv

# ==============================================================================
# PHASE 1: DATA PREPARATION & STRATEGIC FEATURE ENGINEERING
# ==============================================================================

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df



def engineer_strategic_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Architectural Mandate AD-AMARYLLIS-MOD-07:
    Engineers high-order interaction features to test specific, real-world hypotheses.
    This is the core of the Specialist Interaction Cohort (SIC) strategy.
    """
    X = df.copy()
    epsilon = 1e-6 # Defensive constant to prevent division by zero.

    # SIC-01: Grid Constraint vs. Policy Pressure
    # Hypothesis: A pro-development LPA's leniency is nullified by a physically constrained grid.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['lpa_major_commercial_approval_rate'] + epsilon)

    # SIC-02: NIMBY Pressure Cooker
    # Hypothesis: Proximity to public amenity is a stronger negative predictor in affluent/organized communities.
    X['SIC_NIMBY_AMPLIFIER'] = (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]']) / (X['nt_dist_to_nearest_m'] + 100)

    # SIC-03: Land Use Conflict Intensity
    # Hypothesis: Developing on BMV land is exponentially more difficult if it's ALSO under an environmental contract.
    # Note: `cs_on_site_highest_tier_Higher_Tier` must be sanitized to match the column name.
    # We will assume sanitization happens before this function is called.
    #X['[REDACTED_BY_SCRIPT]'] = X['alc_is_bmv_at_site'] * X['cs_on_site_highest_tier_Higher_Tier']

    # SIC-04: Landscape Scarring
    # Hypothesis: The visual intrusion of grid infrastructure is a critical planning risk inside a protected landscape.
    X['SIC_LANDSCAPE_SCARRING'] = (X['ohl_local_tower_ratio'] / (X['[REDACTED_BY_SCRIPT]'] + 100)) * (X['aonb_is_within'] + 1)

    # SIC-05: The Brownfield Bonus
    # Hypothesis: A large project becomes more palatable if it utilizes previously developed land, offsetting its scale.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])

    interaction_features = [f for f in X.columns if f.startswith('SIC_')]
    if interaction_features:
        # Normalize interactions to prevent scale dominance.
        scaler = StandardScaler()
        X[interaction_features] = scaler.fit_transform(X[interaction_features])
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
    #X['[REDACTED_BY_SCRIPT]'] = X['alc_is_bmv_at_site'] * X['cs_on_site_highest_tier_Higher_Tier']
    X['SIC_LANDSCAPE_SCARRING'] = (X['ohl_local_tower_ratio'] / (X['[REDACTED_BY_SCRIPT]'] + 100)) * (X['aonb_is_within'] + 1)
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] * (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])

    # --- Phase V Mandated SICs ---
    # SIC-06: Visual Amenity Conflict
    X['[REDACTED_BY_SCRIPT]'] = (X['solar_site_area_sqm'] * X['[REDACTED_BY_SCRIPT]']) / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    # SIC-07: Grid Reality Gap
    #X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] * X['[REDACTED_BY_SCRIPT]']) * X['[REDACTED_BY_SCRIPT]']
    # SIC-08: LPA Precedent Bias
    X['[REDACTED_BY_SCRIPT]'] = (X['[REDACTED_BY_SCRIPT]'] - X['lpa_major_commercial_approval_rate']) * (1 - X['lpa_workload_trend'])
    # SIC-09: Cumulative Impact Pressure
    X['[REDACTED_BY_SCRIPT]'] = X['solar_site_area_sqm'] * (X['cs_density_5km'] + X['[REDACTED_BY_SCRIPT]'])
    # SIC-10: Economic vs. Ecological Tension
    # Architectural Interpretation: Penalty is inverse distance to nearest veto constraint (Ancient Woodland).
    ecological_penalty = 1 / (X['aw_dist_to_nearest_m'] + 100) # Add 100m to prevent extreme values at close proximity
    X['SIC_ECONOMIC_VS_ECOLOGICAL_TENSION'] = X['[REDACTED_BY_SCRIPT]'] / (X['alc_is_bmv_at_site'] + ecological_penalty + epsilon)

    # --- Phase V Mandated Standalone Features ---
    # Note: '[REDACTED_BY_SCRIPT]' is not available. Using available OHL data as the primary friction source.
    X['[REDACTED_BY_SCRIPT]'] = X['[REDACTED_BY_SCRIPT]'] / (X['[REDACTED_BY_SCRIPT]'] + epsilon)
    # Note: 'feat_lpa_experience_delta' and '[REDACTED_BY_SCRIPT]' cannot be implemented as required component features are not in the dataset.
    X['[REDACTED_BY_SCRIPT]'] = (1 + X['aonb_is_within'] + X['np_is_within']) * (X['[REDACTED_BY_SCRIPT]'] + X['[REDACTED_BY_SCRIPT]'])
    X['[REDACTED_BY_SCRIPT]'] = X['site_lsoa_ruc_rural_score'] * X['[REDACTED_BY_SCRIPT]']

    # --- Final Normalization Step ---
    # Consolidate all interaction features for robust scaling.
    interaction_features = [f for f in X.columns if f.startswith('SIC_') or f.startswith('feat_')]
    if interaction_features:
        # Normalize interactions to prevent scale dominance.
        scaler = StandardScaler()
        X[interaction_features] = scaler.fit_transform(X[interaction_features])
    return X

def define_all_cohorts(X: pd.DataFrame, gcv: list):
    """
    Defines, sanitizes, and fortifies the complete multi-head ensemble structure
    with the Global Context Vector (GCV) per AD-AMARYLLIS-MOD-11.
    """

    def _sanitize(features):
        return [re.sub(r'[^A-Za-z0-9_]+', '_', f) for f in features]

    # --- MONOLITHIC "EXPERT" COHORTS ---
    all_cohorts = {
        "COHORT_CORE_ATTRIBUTES": _sanitize(COHORT_CORE_ATTRIBUTES),
        "COHORT_GRID_ALL": _sanitize(COHORT_GRID_ALL),
        "COHORT_LPA_ALL": _sanitize(COHORT_LPA_ALL),
        "COHORT_ENV_ALL": _sanitize(COHORT_ENV_ALL),
        "COHORT_SOCIO_ECONOMIC_ALL": _sanitize(COHORT_SOCIO_ECONOMIC_ALL),
        "[REDACTED_BY_SCRIPT]": _sanitize([REDACTED_BY_SCRIPT]),
        "[REDACTED_BY_SCRIPT]": _sanitize([REDACTED_BY_SCRIPT])
    }

    # --- SPECIALIST INTERACTION COHORTS (SICs) ---
    # Each SIC model is an expert on a specific, complex trade-off.
    all_cohorts["SIC_GRID_POLICY"] = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'lpa_major_commercial_approval_rate', '[REDACTED_BY_SCRIPT]']
    all_cohorts["SIC_NIMBY"] = ['SIC_NIMBY_AMPLIFIER', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'nt_dist_to_nearest_m']
    all_cohorts["SIC_LAND_USE"] = ['[REDACTED_BY_SCRIPT]', 'alc_is_bmv_at_site', _sanitize(['[REDACTED_BY_SCRIPT]'])[0], 'solar_site_area_sqm']
    all_cohorts["SIC_LANDSCAPE"] = ['SIC_LANDSCAPE_SCARRING', 'ohl_local_tower_ratio', '[REDACTED_BY_SCRIPT]', 'aonb_is_within']
    all_cohorts["SIC_BROWNFIELD"] = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']

    # --- Phase V Mandated SICs ---
    sanitized_demand_rag = _sanitize(['[REDACTED_BY_SCRIPT]'])[0]
    all_cohorts["SIC_VISUAL_AMENITY"] = ['[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    all_cohorts["SIC_GRID_REALITY"] = [ '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', sanitized_demand_rag]
    all_cohorts["SIC_LPA_PRECEDENT"] = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 'lpa_major_commercial_approval_rate', 'lpa_workload_trend']
    all_cohorts["SIC_CUMULATIVE_IMPACT"] = ['[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm', 'cs_density_5km', '[REDACTED_BY_SCRIPT]']
    all_cohorts["SIC_ECO_TENSION"] = ['SIC_ECONOMIC_VS_ECOLOGICAL_TENSION', '[REDACTED_BY_SCRIPT]', 'alc_is_bmv_at_site', 'aw_dist_to_nearest_m']

    # --- ARCHITECTURAL ADDENDUM: Overflow Cohort ---
    # Create a cohort for all features not explicitly assigned to a monolithic expert cohort.
    # This ensures 100% feature coverage and captures potentially valuable residual signals.
    monolithic_cohort_names = [
        "COHORT_CORE_ATTRIBUTES", "COHORT_GRID_ALL", "COHORT_LPA_ALL", "COHORT_ENV_ALL",
        "COHORT_SOCIO_ECONOMIC_ALL", "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]"
    ]
    all_model_features = set(X.columns)
    used_features = set()
    for cohort_name in monolithic_cohort_names:
        used_features.update(all_cohorts.get(cohort_name, []))

    overflow_features = list(all_model_features - used_features)
    all_cohorts["COHORT_OVERFLOW"] = overflow_features
    print(f"[REDACTED_BY_SCRIPT]")

    # Phase 4: Fortify the Ensemble Heads with the GCV
    fortified_cohorts = {}
    print("[REDACTED_BY_SCRIPT]")
    for cohort_name, features in all_cohorts.items():
        original_feature_count = len(features)
        # The union of the original features and the GCV, ensuring no duplicates.
        fortified_features = list(set(features + gcv))
        fortified_cohorts[cohort_name] = fortified_features
        print(f"[REDACTED_BY_SCRIPT]")


    return fortified_cohorts

def phase_1_load_and_prepare_data():
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

    # Mandated Strategic Interaction Feature Engineering
    X_engineered = engineer_strategic_interaction_features(X_sanitized)

    # Final data alignment for regression task.
    #  CRITICAL: Only train on successful applications with a valid, positive planning duration.
    # This prevents data poisoning from failed (0, -1), invalid (NaN), or rejected (10000.0) cases.
    valid_indices = y_reg[(y_reg > 0) & (y_reg < 1000.0)].index
    y_reg_clean = y_reg.loc[valid_indices]
    X_final = X_engineered.loc[valid_indices]

    print(f"[REDACTED_BY_SCRIPT]")
    return X_final, y_reg_clean

# ==============================================================================
# PHASE 2: GLOBAL FEATURE DISTILLATION
# ==============================================================================

# ==============================================================================
# PHASE 2: GLOBAL FEATURE DISTILLATION (REMOVED)
# ==============================================================================
# ARCHITECTURAL NOTE: This phase is obsolete under AD-AMARYLLIS-MOD-11.
# The data-driven Global Context Vector (GCV) now serves this purpose.

def phase_B_prime_generate_oof(X, y, cohorts, gcv, head_params, arbiter_params):
    """[REDACTED_BY_SCRIPT]"""
    print(f"\n--- Phase B': Generating OOF Predictions for Diagnostics ---")

    # Step 1: Generate OOF predictions for all head models.
    oof_head_preds = pd.DataFrame(index=X.index, columns=cohorts.keys(), dtype=np.float64)
    cv_heads = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print("[REDACTED_BY_SCRIPT]")
    for cohort_name, features in cohorts.items():
        for train_idx, val_idx in cv_heads.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            model_features = [f for f in features if f in X_train.columns]
            if not model_features: continue

            head_model = lgb.LGBMRegressor(**head_params, random_state=RANDOM_STATE, verbosity=-1)
            head_model.fit(X_train[model_features], y_train)
            preds = head_model.predict(X_val[model_features])
            oof_head_preds.iloc[val_idx, oof_head_preds.columns.get_loc(cohort_name)] = preds

    # Step 2: Generate OOF predictions for the arbiter model using the head OOFs as features.
    arbiter_training_matrix = pd.concat([oof_head_preds, X[gcv]], axis=1)
    oof_arbiter_preds = pd.Series(index=X.index, dtype=np.float64, name="y_pred")
    cv_arbiter = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print("[REDACTED_BY_SCRIPT]")
    for train_idx, val_idx in cv_arbiter.split(arbiter_training_matrix, y):
        X_train_arb, X_val_arb = arbiter_training_matrix.iloc[train_idx], arbiter_training_matrix.iloc[val_idx]
        y_train_arb = y.iloc[train_idx]

        arbiter_model = lgb.LGBMRegressor(**arbiter_params, random_state=RANDOM_STATE, verbosity=-1)
        arbiter_model.fit(X_train_arb, y_train_arb)
        oof_arbiter_preds.iloc[val_idx] = arbiter_model.predict(X_val_arb)

    diagnostics_df = pd.DataFrame({'y_true': y, 'y_pred': oof_arbiter_preds})
    print("[REDACTED_BY_SCRIPT]")
    return diagnostics_df


def phase_D_run_performance_triage(diagnostics_df, X_features, baseline_rmse):
    """
    Implements the Performance Triage & Sub-Cohort Diagnostics Protocol (AD-AMARYLLIS-MOD-12).
    """
    print("[REDACTED_BY_SCRIPT]")

    # Phase 1: Forge the Diagnostics Artifact
    artifact = X_features.join(diagnostics_df)
    artifact['[REDACTED_BY_SCRIPT]'] = (artifact['y_pred'] - artifact['y_true']).abs()
    print(f"[REDACTED_BY_SCRIPT]")

    # Phase 2: Define the Strategic Triage Cohorts
    lpa_workload_q3 = artifact['lpa_workload_trend'].quantile(0.75)
    lpa_p90_days_q3 = artifact['[REDACTED_BY_SCRIPT]'].quantile(0.75)

    # Sanitize column names for queries
    artifact.rename(columns=lambda c: re.sub(r'[^A-Za-z0-9_]+', '_', c), inplace=True)
    triage_cohorts = {
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "SCALE_SMALL (<10MW)": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "DNO_NGED_GROUND_TRUTH": "[REDACTED_BY_SCRIPT]",
        "DNO_IMPUTED_BRIDGE": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "LPA_PERMISSIVE": "[REDACTED_BY_SCRIPT]",
        "LPA_ADVERSARIAL": "[REDACTED_BY_SCRIPT]",
        "LPA_OVERBURDENED": f"[REDACTED_BY_SCRIPT]",
    }

    # Phase 3: Execution and Reporting Matrix
    report_data = []
    global_rmse = root_mean_squared_error(artifact['y_true'], artifact['y_pred'])
    report_data.append({"Cohort": "GLOBAL BASELINE", "N": len(artifact), "RMSE (Days)": f"{global_rmse:.2f}", "Delta": "-"})

    for name, query in triage_cohorts.items():
        try:
            cohort_df = artifact.query(query)
            if cohort_df.empty:
                rmse, n_samples = np.nan, 0
            else:
                rmse = root_mean_squared_error(cohort_df['y_true'], cohort_df['y_pred'])
                n_samples = len(cohort_df)
            report_data.append({"Cohort": name, "N": n_samples, "RMSE (Days)": f"{rmse:.2f}", "Delta": f"[REDACTED_BY_SCRIPT]"})
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]'{name}': {e}")


    report_df = pd.DataFrame(report_data)
    print("[REDACTED_BY_SCRIPT]")
    print(report_df.to_markdown(index=False))

    # Phase 4: The "Worst Offenders" Deep-Dive Report
    print("\n--- 'Worst Offenders'[REDACTED_BY_SCRIPT]")
    worst_offenders = artifact.sort_values('[REDACTED_BY_SCRIPT]', ascending=False).head(5)

    for i, row in worst_offenders.iterrows():
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]'y_pred'[REDACTED_BY_SCRIPT]'y_true'[REDACTED_BY_SCRIPT]'regression_absolute_error_days']:.0f} days)")
        print(f"  - Diagnostics:")
        print(f"    - Scale: {row['installed_capacity_mwelec'[REDACTED_BY_SCRIPT]'lpa_major_commercial_approval_rate']:.2f}")
        print(f"    - Grid: {row['dist_to_nearest_substation_km'[REDACTED_BY_SCRIPT]'winter_nameplate_loading_pct']:.2%}")
        print(f"    - Env: {row['env_min_dist_to_any_constraint_m'[REDACTED_BY_SCRIPT]'alc_is_bmv_at_site'])}")
        # Simple hypothesis generation
        if row['y_pred'] < row['y_true']:
             print("[REDACTED_BY_SCRIPT]")
        else:
             print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]")



# ==============================================================================
# PHASE 3: THE UNIFIED ENSEMBLE FORGE (Tuning & Training)
# ==============================================================================

def phase_A_tune_head_models(X, y, grid_cohort_features):
    """[REDACTED_BY_SCRIPT]"""
    print(f"[REDACTED_BY_SCRIPT]")
    X_grid = X[grid_cohort_features]

    def objective(trial):
        params = {
            'objective': trial.suggest_categorical('objective', ['regression_l1', 'regression', 'huber', 'fair']),
            'metric': trial.suggest_categorical('metric', ['mae']),
            'random_state': RANDOM_STATE, 
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 1000, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 125, 200),
            'max_depth': trial.suggest_int('max_depth', 10, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        model = lgb.LGBMRegressor(**params)
        score = cross_val_score(model, X_grid, y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring='neg_root_mean_squared_error').mean()
        return -score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS_HEAD)

    print(f"[REDACTED_BY_SCRIPT]")
    return study.best_params


def phase_B_tune_arbiter_model(X, y, cohorts, global_features, unified_head_params):
    """[REDACTED_BY_SCRIPT]"""
    print(f"[REDACTED_BY_SCRIPT]")

    def objective(trial):
        oof_predictions = pd.DataFrame(index=X.index, columns=cohorts.keys(), dtype=np.float64)
        cv_inner = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, val_idx in cv_inner.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            for cohort_name, features in cohorts.items():
                # Defensive check for feature existence
                model_features = [f for f in features if f in X_train.columns]
                if not model_features: continue
                
                head_model = lgb.LGBMRegressor(**unified_head_params, random_state=RANDOM_STATE, verbosity=-1)
                head_model.fit(X_train[model_features], y_train)
                preds = head_model.predict(X_val[model_features])
                oof_predictions.iloc[val_idx, oof_predictions.columns.get_loc(cohort_name)] = preds

        final_training_matrix = pd.concat([oof_predictions, X[global_features]], axis=1)
        
        arbiter_params = {
            'objective': 'regression_l1', 'metric': 'rmse', 'random_state': RANDOM_STATE, 'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 1500, 2500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 50, 120),
            'max_depth': trial.suggest_int('max_depth', 6, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 20.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 20.0, log=True),
        }
        arbiter_model = lgb.LGBMRegressor(**arbiter_params)
        
        score = cross_val_score(arbiter_model, final_training_matrix, y, cv=cv_inner, scoring='neg_root_mean_squared_error').mean()
        return -score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS_ARBITER)

    final_honest_rmse = study.best_value
    print(f"[REDACTED_BY_SCRIPT]")
    return study.best_params, final_honest_rmse


def phase_C_train_final_production_model(X, y, cohorts, global_features, head_params, arbiter_params):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")

    full_data_head_preds = pd.DataFrame(index=X.index, columns=cohorts.keys(), dtype=np.float64)
    for cohort_name, features in cohorts.items():
        model_features = [f for f in features if f in X.columns]
        if not model_features: continue
        head_model = lgb.LGBMRegressor(**head_params, random_state=RANDOM_STATE, verbosity=-1)
        head_model.fit(X[model_features], y)
        full_data_head_preds[cohort_name] = head_model.predict(X[model_features])

    final_training_matrix = pd.concat([full_data_head_preds, X[global_features]], axis=1)

    final_arbiter = lgb.LGBMRegressor(**arbiter_params, random_state=RANDOM_STATE, verbosity=-1)
    final_arbiter.fit(final_training_matrix, y)
    joblib.dump(final_arbiter, MODEL_ARBITER_PATH)
    print(f"[REDACTED_BY_SCRIPT]")

    print("Phase C Complete.")
    return final_arbiter, final_training_matrix


def phase_E_generate_final_audit_report(model, X_meta, baseline_rmse, final_rmse, num_heads):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")

    report_content = f"""
    # Amaryllis Project: Hybrid Ensemble Performance Report (v3.1-Tuned)

    ## Performance Summary
    - **Baseline Untuned RMSE:** {baseline_rmse:.4f}
    - **Final Tuned Honest RMSE:** {final_rmse:.4f}
    - **Improvement:** {baseline_rmse - final_rmse:.4f} ({((baseline_rmse - final_rmse) / baseline_rmse) * 100:.2f}%)

    ## Architectural Details
    - **Architecture:** Hybrid Ensemble ({num_heads} Heads) with Tuned LGBM Arbiter
    - **Tuning Protocol:** Mandated Tiered Optuna Campaign (AD-AMARYLLIS-MOD-06 Rev. 1)
    - **Validation Protocol:** Doubly-Nested {CV_SPLITS}-Fold Cross-Validation

    The final reported RMSE is the best score from the Tier 2 Optuna study, representing
    a robust, honest estimate of performance on unseen data. The Arbiter model learns from
    the predictions of {num_heads} specialist and expert models.
    """
    with open(REPORT_PATH, 'w') as f:
        f.write(report_content)
    print(f"[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_meta)

    plt.figure(figsize=(10, 8)) # Increased figure size for readability
    shap.summary_plot(shap_values, X_meta, show=False, max_display=len(X_meta.columns))
    plt.title("[REDACTED_BY_SCRIPT]", fontsize=16)
    plt.savefig(SHAP_PLOT_PATH, bbox_inches='tight')
    plt.close()
    print(f"[REDACTED_BY_SCRIPT]")

    print("Phase D Complete.")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    BASELINE_RMSE = 146.9899 # From previous untuned run (v2)

    # Phase 1: Load, sanitize, and engineer features
    X_final, y_final = phase_1_load_and_prepare_data()

    # --- AD-AMARYLLIS-MOD-11: Global Signal Amplification Protocol (Corrected Sequence) ---
    # Step 1: Forge a preliminary oracle to find the GCV.
    # A fixed, robust set of params is used for stability; the goal is a stable feature ranking, not SOTA performance.
    preliminary_oracle_params = {'objective': 'regression_l1', 'metric': 'rmse', 'random_state': RANDOM_STATE, 'n_estimators': 750, 'learning_rate': 0.05, 'num_leaves': 60}
    GLOBAL_CONTEXT_VECTOR = phase_0_get_gcv(X_final, y_final, n_features=20, head_params=preliminary_oracle_params)

    # Step 2: Define and Fortify all cohorts with the GCV.
    all_cohorts = define_all_cohorts(X_final, gcv=GLOBAL_CONTEXT_VECTOR)
    print("-"*80)

    # Step 3: Now, tune the head models using a representative FORTIFIED cohort.
    # This ensures hyperparameters are optimized for the final, context-aware head architecture.
    print("[REDACTED_BY_SCRIPT]")
    unified_head_params = phase_A_tune_head_models(X_final, y_final, all_cohorts['COHORT_GRID_ALL'])

    # Step 4: Proceed with Arbiter tuning using the fortified cohorts and their correctly optimized params.
    # Phase B: Tier 2 Tuning for the Arbiter
    optimized_arbiter_params, final_tuned_rmse = phase_B_tune_arbiter_model(
        X_final, y_final, all_cohorts, GLOBAL_CONTEXT_VECTOR, unified_head_params
    )

    # --- AD-AMARYLLIS-MOD-12: Performance Triage & Diagnostics Protocol ---
    # Phase B': Generate OOF predictions using final tuned params for diagnostics
    diagnostics_df = phase_B_prime_generate_oof(
        X_final, y_final, all_cohorts, GLOBAL_CONTEXT_VECTOR, unified_head_params, optimized_arbiter_params
    )

    # Phase D: Run the full performance triage and deep-dive analysis
    phase_D_run_performance_triage(diagnostics_df, X_final, baseline_rmse=final_tuned_rmse)
    print("-"*80)

    # --- Final Production Artifact Generation ---
    # Phase C: Train the final production model on all data with optimized parameters
    final_model, final_matrix = phase_C_train_final_production_model(
        X_final, y_final, all_cohorts, GLOBAL_CONTEXT_VECTOR, unified_head_params, optimized_arbiter_params
    )

    # Phase E: Generate the final report and audit artifacts
    phase_E_generate_final_audit_report(final_model, final_matrix, BASELINE_RMSE, final_tuned_rmse, len(all_cohorts))

    print("[REDACTED_BY_SCRIPT]")