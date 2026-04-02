"""
Architectural Directive AD-AM-INF-01 Phase 1: Quantile Model Training

Trains P10 and P90 quantile regression models using LightGBM to provide
confidence intervals for the Decision Dossier uncertainty quantification protocol.

These models predict the 10th and 90th percentiles of the planning duration
distribution, enabling the system to communicate prediction uncertainty
transparently to end users.

Author: Amaryllis Decision Intelligence Team
Date: 2025-11-21
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- Configuration ---
ARTIFACTS_DIR = r"[REDACTED_BY_SCRIPT]"
X_TRAIN_PATH = r"[REDACTED_BY_SCRIPT]"
Y_TRAIN_PATH = r"[REDACTED_BY_SCRIPT]"

# Output paths
P10_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
P90_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

def sanitize_column_names(df):
    """[REDACTED_BY_SCRIPT]"""
    import re
    sanitized_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = sanitized_columns
    return df

def load_training_data():
    """Load and prepare training data"""
    logging.info("Loading training data...")
    
    X_train = pd.read_csv(X_TRAIN_PATH, index_col=0)
    y_train = pd.read_csv(Y_TRAIN_PATH, index_col=0).squeeze("columns")
    
    # Sanitize column names to remove special characters
    X_train = sanitize_column_names(X_train)
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Remove NaN targets
    valid_mask = y_train.notna()
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Filter for ground-mount solar only (technology_type == 21)
    if 'technology_type' in X_train.columns:
        gm_solar_mask = X_train['technology_type'] == 21
        X_train = X_train[gm_solar_mask]
        y_train = y_train[gm_solar_mask]
        logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return X_train, y_train

def train_quantile_model(X_train, y_train, X_val, y_val, alpha, model_name):
    """
    Train a quantile regression model for the specified percentile.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        alpha: Quantile to predict (0.1 for P10, 0.9 for P90)
        model_name: Name for logging
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'metric': 'quantile',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'max_depth': 10,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='quantile',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return model

def validate_quantile_models(X_val, y_val, p10_model, p90_model):
    """
    Validate quantile models by checking coverage and calibration.
    
    Coverage: What percentage of actual values fall within [P10, P90]?
    Target: ~80% (since we're capturing the 10th to 90th percentile range)
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    p10_preds = p10_model.predict(X_val)
    p90_preds = p90_model.predict(X_val)
    
    # Coverage: how many actual values fall in the interval?
    within_interval = ((y_val >= p10_preds) & (y_val <= p90_preds))
    coverage = within_interval.mean()
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Calibration check: are the quantiles actually at the right levels?
    p10_calibration = (y_val < p10_preds).mean()  # Should be ~0.10
    p90_calibration = (y_val < p90_preds).mean()  # Should be ~0.90
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Interval width statistics
    interval_widths = p90_preds - p10_preds
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Quality check
    if coverage < 0.70 or coverage > 0.90:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
    else:
        logging.info(f"[REDACTED_BY_SCRIPT]")
    
    if abs(p10_calibration - 0.10) > 0.05 or abs(p90_calibration - 0.90) > 0.05:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
    else:
        logging.info(f"[REDACTED_BY_SCRIPT]")
    
    return {
        'coverage': coverage,
        'p10_calibration': p10_calibration,
        'p90_calibration': p90_calibration,
        'mean_interval_width': interval_widths.mean()
    }

def main():
    logging.info("="*70)
    logging.info("[REDACTED_BY_SCRIPT]")
    logging.info("="*70)
    
    # Ensure output directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Load data
    X, y = load_training_data()
    
    # Split into train/val for proper validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Train P10 model (10th percentile - lower bound)
    p10_model = train_quantile_model(X_train, y_train, X_val, y_val, alpha=0.1, model_name="P10")
    
    # Train P90 model (90th percentile - upper bound)
    p90_model = train_quantile_model(X_train, y_train, X_val, y_val, alpha=0.9, model_name="P90")
    
    # Validate
    validation_metrics = validate_quantile_models(X_val, y_val, p10_model, p90_model)
    
    # Save models
    logging.info("[REDACTED_BY_SCRIPT]")
    joblib.dump(p10_model, P10_MODEL_PATH)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    joblib.dump(p90_model, P90_MODEL_PATH)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Save validation report
    report_path = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")
    with open(report_path, 'w') as f:
        f.write("[REDACTED_BY_SCRIPT]")
        f.write("="*50 + "\n\n")
        f.write(f"[REDACTED_BY_SCRIPT]")
        f.write(f"[REDACTED_BY_SCRIPT]")
        f.write(f"[REDACTED_BY_SCRIPT]")
        f.write(f"[REDACTED_BY_SCRIPT]'coverage']:.2%}\n")
        f.write(f"[REDACTED_BY_SCRIPT]'p10_calibration']:.2%}\n")
        f.write(f"[REDACTED_BY_SCRIPT]'p90_calibration']:.2%}\n")
        f.write(f"[REDACTED_BY_SCRIPT]'mean_interval_width']:.1f} days\n")
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    logging.info("\n" + "="*70)
    logging.info("[REDACTED_BY_SCRIPT]")
    logging.info("="*70)

if __name__ == '__main__':
    main()
