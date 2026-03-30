
import logging
import pandas as pd
import numpy as np
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Constants ---
RANDOM_STATE = 42
SENTINEL_VALUE = -1.0
IMPUTATION_MODELS_DIR = r"[REDACTED_BY_SCRIPT]"

def execute(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Executes production imputation using pre-trained joblib models.
    
    This executor loads pre-trained imputation models from disk and uses them
    to impute missing values. For features without pre-trained models, it falls
    back to median imputation.
    
    Args:
        df_master (pd.DataFrame): The dataframe with missing values to be imputed.
        
    Returns:
        pd.DataFrame: The dataframe with imputed values.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Phase 1: Ingestion and Target Identification ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    logging.info("[REDACTED_BY_SCRIPT]")

    # Replace all forms of missing/sentinel values with np.nan for unified processing
    # This includes the original sentinel, explicit NaNs, and 0.0 as per user requirement
    df_master.replace([SENTINEL_VALUE, 0.0, 0], np.nan, inplace=True)
    print(f"[REDACTED_BY_SCRIPT]")

    target_cols_raw = [col for col in df_master.columns if df_master[col].isnull().any()]
    
    target_cols = []
    for col in target_cols_raw:
        series_to_check = df_master[col].dropna()
        numeric_series = pd.to_numeric(series_to_check, errors='coerce')
        if numeric_series.isna().sum() > 0:
            logging.warning(f"Column '{col}'[REDACTED_BY_SCRIPT]")
        else:
            target_cols.append(col)

    if not target_cols:
        logging.warning("[REDACTED_BY_SCRIPT]")
        return df_master

    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- Phase 2: Define Universal Predictor Set ---
    logging.info("[REDACTED_BY_SCRIPT]")
    metadata_leakage_cols = [
        'amaryllis_id', 'solar_farm_id', 'application_id', 'dno_source',
        'submission_date', 'datecommissioned', 'assessmentdate',
        'permission_granted', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm', 'chp_enabled',
        '[REDACTED_BY_SCRIPT]', 'submission_year', 'submission_month',
        'submission_day', 'submission_month_sin', 'submission_month_cos'
    ]
    
    original_target_count = len(target_cols)
    target_cols = [col for col in target_cols if col not in metadata_leakage_cols]
    if len(target_cols) < original_target_count:
        logging.info(f"[REDACTED_BY_SCRIPT]")

    df_master.replace(SENTINEL_VALUE, np.nan, inplace=True)
    logging.info("[REDACTED_BY_SCRIPT]")
    
    base_predictor_cols = [col for col in df_master.columns if col not in target_cols and col not in metadata_leakage_cols]
    numeric_predictors = df_master[base_predictor_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_predictors = df_master[base_predictor_cols].select_dtypes(exclude=np.number).columns.tolist()

    MAX_CARDINALITY_FOR_OHE = 10 
    ohe_cols = [col for col in categorical_predictors if df_master[col].nunique() <= MAX_CARDINALITY_FOR_OHE]
    
    if ohe_cols:
        df_master = pd.get_dummies(df_master, columns=ohe_cols, prefix=ohe_cols, dummy_na=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")
    
    final_predictor_cols = [p for p in df_master.select_dtypes(include=np.number).columns.drop(target_cols, errors='ignore').tolist() if p not in metadata_leakage_cols]
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- Phase 3: Load Pre-Trained Models and Impute ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_imputed = df_master.copy()
    models_loaded = 0
    models_missing = 0
    median_fallback_count = 0

    for i, target_col in enumerate(target_cols):
        logging.info(f"[REDACTED_BY_SCRIPT]'{target_col}'")
        
        impute_mask = df_imputed[target_col].isna()

        if impute_mask.sum() == 0:
            logging.info("[REDACTED_BY_SCRIPT]")
            continue
        
        # Construct the model filepath
        model_filename = f"[REDACTED_BY_SCRIPT]"
        model_path = os.path.join(IMPUTATION_MODELS_DIR, model_filename)
        
        # Try to load the pre-trained model
        if os.path.exists(model_path):
            try:
                logging.info(f"[REDACTED_BY_SCRIPT]")
                production_pipeline = joblib.load(model_path)
                models_loaded += 1
                
                # CRITICAL: Extract the feature names the model expects
                # The model was trained on specific features - we must use exactly those
                if hasattr(production_pipeline, 'feature_names_in_'):
                    # Sklearn pipeline has this attribute
                    expected_features = production_pipeline.feature_names_in_.tolist()
                elif hasattr(production_pipeline, 'named_steps'):
                    # Try to get from the last step of the pipeline
                    last_step = list(production_pipeline.named_steps.values())[-1]
                    if hasattr(last_step, 'feature_names_in_'):
                        expected_features = last_step.feature_names_in_.tolist()
                    else:
                        # Fallback: use all available numeric features
                        expected_features = final_predictor_cols
                else:
                    # Fallback: use all available numeric features
                    expected_features = final_predictor_cols
                
                # Check which expected features are available in our data
                available_features = [f for f in expected_features if f in df_imputed.columns]
                missing_features = [f for f in expected_features if f not in df_imputed.columns]
                
                if missing_features:
                    logging.warning(f"[REDACTED_BY_SCRIPT]")
                
                # Get the data to impute, using ONLY the features the model expects
                X_to_impute = df_imputed.loc[impute_mask, available_features]
                
                # If we're missing features, we need to add them with fill values
                if missing_features:
                    for missing_feat in missing_features:
                        X_to_impute[missing_feat] = np.nan  # Will be handled by model's internal imputation
                    # Reorder to match expected order
                    X_to_impute = X_to_impute[expected_features]
                
                # Predict using the aligned feature set
                try:
                    predicted_values = production_pipeline.predict(X_to_impute)
                    df_imputed.loc[impute_mask, target_col] = predicted_values
                    logging.info(f"[REDACTED_BY_SCRIPT]")
                    logging.info(f"[REDACTED_BY_SCRIPT]")
                except Exception as e:
                    logging.warning(f"[REDACTED_BY_SCRIPT]'{target_col}': {e}")
                    logging.info(f"[REDACTED_BY_SCRIPT]'{target_col}'.")
                    median_value = df_imputed.loc[~impute_mask, target_col].median()
                    df_imputed.loc[impute_mask, target_col] = median_value
                    median_fallback_count += 1
                    logging.info(f"[REDACTED_BY_SCRIPT]")
                    
            except Exception as e:
                logging.warning(f"[REDACTED_BY_SCRIPT]")
                logging.info(f"[REDACTED_BY_SCRIPT]'{target_col}'.")
                median_value = df_imputed.loc[~impute_mask, target_col].median()
                df_imputed.loc[impute_mask, target_col] = median_value
                median_fallback_count += 1
                logging.info(f"[REDACTED_BY_SCRIPT]")
        else:
            logging.warning(f"[REDACTED_BY_SCRIPT]'{target_col}' at {model_path}")
            logging.info(f"[REDACTED_BY_SCRIPT]'{target_col}'.")
            models_missing += 1
            median_value = df_imputed.loc[~impute_mask, target_col].median()
            df_imputed.loc[impute_mask, target_col] = median_value
            median_fallback_count += 1
            logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- Phase 4: Reporting and Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")

    logging.info("[REDACTED_BY_SCRIPT]")
    return df_imputed
