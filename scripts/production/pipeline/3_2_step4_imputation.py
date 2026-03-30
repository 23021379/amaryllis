"""

This script can be improved.
At the moment it performs a fine-tuned sequential imputation for each target column. 
Some features may perform poorly due to their postion in the sequence.
After the first pass, a second pass could be implemented to re-impute columns that had poor performance the first time around.
We can even increase the search space since there would be fewer columns to impute in the second pass.
This would likely improve overall imputation quality since we have more data, and better search space.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --- 1.0 Operational Environment: Mandated Constants & Artifacts (AD-IMP-02) ---
print("[REDACTED_BY_SCRIPT]")

# Mandatory constant for reproducibility
RANDOM_STATE = 42

# Holdout validation constant# Holdout validation constant
HOLDOUT_SET_SIZE = 0.1

SENTINEL_VALUE = -1.0
# Two-pass imputation tuning constants
R2_REFINEMENT_THRESHOLD = 0.50
INITIAL_OPTUNA_TRIALS = 3
REFINED_OPTUNA_TRIALS = 25 # Allocate more compute for difficult features

# Input Artifact
PRE_FORGE_DATA_PATH = r"[REDACTED_BY_SCRIPT]"

# Output Artifact
FINAL_IMPUTED_ARTIFACT_PATH = r"[REDACTED_BY_SCRIPT]"
FINAL_METRICS_REPORT_PATH = r"[REDACTED_BY_SCRIPT]"

# Holdout validation constant
HOLDOUT_SET_SIZE = 0.1

def execute_imputation_protocol(data_path):
    """
    Executes a resilient, iterative per-column production imputation protocol.
    """
    # --- Phase 1: Ingestion and Target Identification ---
    print("[REDACTED_BY_SCRIPT]")
    df_master = pd.read_csv(data_path)
    print(f"[REDACTED_BY_SCRIPT]")

    # General data sanitization: Replace infinities with NaN
    df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("[REDACTED_BY_SCRIPT]")

    target_cols_raw = [col for col in df_master.columns if (df_master[col] == SENTINEL_VALUE).any()]
    
    # Defensive check: ensure targets are fundamentally numeric before proceeding
    target_cols = []
    for col in target_cols_raw:
        # Create a series for inspection, excluding the sentinel value
        series_to_check = df_master[col][df_master[col] != SENTINEL_VALUE].dropna()
        
        # Attempt to convert the series to numeric. If any value fails, it will remain as a non-numeric type (e.g., string).
        numeric_series = pd.to_numeric(series_to_check, errors='coerce')
        
        # Check if there are any non-numeric values left after coercion (i.e., where coercion produced NaN)
        if numeric_series.isna().sum() > 0:
            print(f"WARNING: Column '{col}'[REDACTED_BY_SCRIPT]")
        else:
            target_cols.append(col)

    assert len(target_cols) > 0, "[REDACTED_BY_SCRIPT]"
    print(f"[REDACTED_BY_SCRIPT]")

    # --- Phase 2: Define Universal Predictor Set ---
    print("[REDACTED_BY_SCRIPT]")
    metadata_leakage_cols = [
        'amaryllis_id', 'solar_farm_id', 'application_id', 'dno_source',
        'submission_date', 'datecommissioned', 'assessmentdate',
        'permission_granted', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
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
    # Defensive check: Ensure metadata columns are not targeted for imputation
    original_target_count = len(target_cols)
    target_cols = [col for col in target_cols if col not in metadata_leakage_cols]
    if len(target_cols) < original_target_count:
        print(f"[REDACTED_BY_SCRIPT]")

    df_master.replace(SENTINEL_VALUE, np.nan, inplace=True)
    #df_master.replace(0.0, np.nan, inplace=True)
    print("[REDACTED_BY_SCRIPT]")
    
    # Predictors are all columns that are NOT targets and NOT metadata/leakage
    base_predictor_cols = [
        col for col in df_master.columns if col not in target_cols and col not in metadata_leakage_cols
    ]

    # Separate numeric and categorical predictors
    numeric_predictors = df_master[base_predictor_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_predictors = df_master[base_predictor_cols].select_dtypes(exclude=np.number).columns.tolist()

    print(f"[REDACTED_BY_SCRIPT]")

    # Handle categorical predictors: one-hot encode low cardinality, drop high cardinality
    MAX_CARDINALITY_FOR_OHE = 10 
    ohe_cols = []
    for col in categorical_predictors:
        cardinality = df_master[col].nunique()
        if cardinality <= MAX_CARDINALITY_FOR_OHE:
            print(f"    One-hot encoding '{col}'[REDACTED_BY_SCRIPT]")
            ohe_cols.append(col)
        else:
            print(f"    Dropping '{col}'[REDACTED_BY_SCRIPT]")

    if ohe_cols:
        df_master = pd.get_dummies(df_master, columns=ohe_cols, prefix=ohe_cols, dummy_na=False)
        print("[REDACTED_BY_SCRIPT]")
        # The original ohe_cols are gone, replaced by new one-hot encoded columns.
        # We need to update our list of predictors to include these new columns.
        # The simplest way is to re-evaluate the numeric columns from the transformed dataframe.
        final_predictor_cols = df_master.select_dtypes(include=np.number).columns.drop(target_cols, errors='ignore').tolist()
        # And remove any lingering metadata columns that might be numeric
        final_predictor_cols = [p for p in final_predictor_cols if p not in metadata_leakage_cols]
    else:
        final_predictor_cols = numeric_predictors

    print(f"[REDACTED_BY_SCRIPT]")

    # --- Phase 3: Preprocessing, Hyperparameter Tuning, and Imputation ---
    print("[REDACTED_BY_SCRIPT]")
    df_imputed = df_master.copy() # Use the potentially OHE dataframe
    unimputable_cols = []
    metrics_report = []

    for i, target_col in enumerate(target_cols):
        print(f"[REDACTED_BY_SCRIPT]'{target_col}'")
        
        known_data_mask = df_imputed[target_col].notna()
        impute_mask = df_imputed[target_col].isna()

        if impute_mask.sum() == 0:
            print("[REDACTED_BY_SCRIPT]")
            continue
        
        if known_data_mask.sum() < 150: # Need a reasonable number of samples for tuning + validation
            print(f"[REDACTED_BY_SCRIPT]'{target_col}'[REDACTED_BY_SCRIPT]")
            unimputable_cols.append(target_col)
            continue

        X_known = df_imputed.loc[known_data_mask, final_predictor_cols]
        y_known = df_imputed.loc[known_data_mask, target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X_known, y_known, test_size=HOLDOUT_SET_SIZE, random_state=RANDOM_STATE
        )

        # --- Optuna Objective Function ---
        # --- Optuna Objective Function ---
        def objective(trial, refinement_pass=False):
            if refinement_pass:
                # Wider, more intensive search space for difficult features
                params = {
                    'objective': 'huber',
                    'metric': 'mae',
                    'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 30, 400),
                    'max_depth': trial.suggest_int('max_depth', 7, 20),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                }
            else:
                # Standard, rapid search space for initial screening
                params = {
                    'objective': 'huber',
                    'metric': 'mae',
                    'n_estimators': trial.suggest_int('n_estimators', 1000, 1500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 125, 200),
                    'max_depth': trial.suggest_int('max_depth', 10, 15),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }
            
            model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **params)
            pipeline = Pipeline([
                ('variance_threshold', VarianceThreshold(threshold=0.0)),
                ('scaler', StandardScaler()),
                ('transformer', PowerTransformer(method='yeo-johnson')),
                ('regressor', model)
            ])
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            return rmse

        # --- Pass 1: Initial Screening Search ---
        print("[REDACTED_BY_SCRIPT]")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, refinement_pass=False), n_trials=INITIAL_OPTUNA_TRIALS, n_jobs=-1)
        print(f"[REDACTED_BY_SCRIPT]")
        
        best_params = study.best_params

        # --- Provisional Validation & Triage ---
        provisional_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **best_params)
        provisional_pipeline = Pipeline([
            ('variance_threshold', VarianceThreshold(threshold=0.0)),
            ('scaler', StandardScaler()),
            ('transformer', PowerTransformer(method='yeo-johnson')),
            ('regressor', provisional_model)
        ])
        provisional_pipeline.fit(X_train, y_train)
        provisional_pred = provisional_pipeline.predict(X_test)
        provisional_r2 = r2_score(y_test, provisional_pred)

        # --- Pass 2: Refined Search (if needed) ---
        if provisional_r2 < R2_REFINEMENT_THRESHOLD:
            print(f"[REDACTED_BY_SCRIPT]")
            refined_study = optuna.create_study(direction='minimize')
            refined_study.optimize(lambda trial: objective(trial, refinement_pass=True), n_trials=REFINED_OPTUNA_TRIALS, n_jobs=-1)
            print(f"[REDACTED_BY_SCRIPT]")
            best_params = refined_study.best_params # Overwrite with superior parameters
        else:
            print(f"[REDACTED_BY_SCRIPT]")

        # --- Final Holdout Validation ---
        final_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **best_params)
        final_pipeline = Pipeline([
            ('variance_threshold', VarianceThreshold(threshold=0.0)),
            ('scaler', StandardScaler()),
            ('transformer', PowerTransformer(method='yeo-johnson')),
            ('regressor', final_model)
        ])
        final_pipeline.fit(X_train, y_train)
        y_pred = final_pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        metrics_report.append({
            'feature': target_col, 'R2': r2, 'MAE': mae, 'RMSE': rmse_final, 'MAPE_%': mape
        })
        print(f"[REDACTED_BY_SCRIPT]")

        # --- Production Imputation (Retrain on ALL known data) ---
        production_pipeline = Pipeline([
            ('variance_threshold', VarianceThreshold(threshold=0.0)),
            ('scaler', StandardScaler()),
            ('transformer', PowerTransformer(method='yeo-johnson')),
            ('regressor', lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params))
        ])
        production_pipeline.fit(X_known, y_known)
        
        X_to_impute = df_imputed.loc[impute_mask, final_predictor_cols]
        predicted_values = production_pipeline.predict(X_to_impute)
        df_imputed.loc[impute_mask, target_col] = predicted_values
        print(f"[REDACTED_BY_SCRIPT]")

   # --- Phase 4: Metrics Reporting and Finalization ---
    print("[REDACTED_BY_SCRIPT]")

    metrics_df = pd.DataFrame(metrics_report).sort_values(by='R2', ascending=False)
    print("[REDACTED_BY_SCRIPT]")
    print(metrics_df.head(10).to_string())
    metrics_df.to_csv(FINAL_METRICS_REPORT_PATH, index=False)
    print(f"[REDACTED_BY_SCRIPT]")

    print("[REDACTED_BY_SCRIPT]")
    assert df_imputed.shape[0] == df_master.shape[0], f"[REDACTED_BY_SCRIPT]"

    cols_with_nulls = df_imputed[target_cols].columns[df_imputed[target_cols].isnull().any()].tolist()
    
    assert set(cols_with_nulls) == set(unimputable_cols), (
        f"[REDACTED_BY_SCRIPT]"
        f"[REDACTED_BY_SCRIPT]"
    )
    
    print("[REDACTED_BY_SCRIPT]")
    if unimputable_cols:
        print(f"[REDACTED_BY_SCRIPT]")
        for col in unimputable_cols:
            print(f"  - {col}")
    else:
        print("[REDACTED_BY_SCRIPT]")

    df_imputed.to_csv(FINAL_IMPUTED_ARTIFACT_PATH, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    return df_imputed

if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")

    # The main protocol function handles the entire workflow
    final_artifact = execute_imputation_protocol(PRE_FORGE_DATA_PATH)
    
    print("\n========= PRODUCTION IMPUTATION COMPLETE =========")
    print("[REDACTED_BY_SCRIPT]")
    print(final_artifact.head())