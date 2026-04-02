import pandas as pd
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# This dictionary defines which categorical columns should be one-hot encoded.
# The keys are the source column names, and the values are prefixes for the new encoded columns.
# The executor will automatically find all unique values in the column and create a new
# binary feature for each one.
CATEGORICAL_FEATURES_TO_ENCODE = {
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'sssi_unit_nearest_condition': 'sssi_unit_nearest_condition',
    'np_nearest_name': 'np_nearest_name',
    'sac_nearest_name': 'sac_nearest_name',
    'spa_nearest_name': 'spa_nearest_name',
    'hp_nearest_name': 'hp_nearest_name',
    'ph_nearest_name': 'ph_nearest_name',
    'cs_highest_tier_on_site': 'cs_highest_tier_on_site',
    'es_highest_scheme_on_site': 'es_highest_scheme_on_site',
    # Add any other categorical columns that need encoding
}

def execute(master_gdf):
    """
    Performs one-hot encoding on specified categorical features.
    This executor should run late in the pipeline, after all source categorical
    features have been created and synthesized.
    It ensures the 'hex_id' index is preserved throughout the operation.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Ensure hex_id is the index for alignment.
    if master_gdf.index.name != 'hex_id':
        if 'hex_id' in master_gdf.columns:
            master_gdf.set_index('hex_id', inplace=True)
        else:
            logging.error("Critical error: 'hex_id'[REDACTED_BY_SCRIPT]")
            raise KeyError("'hex_id'[REDACTED_BY_SCRIPT]")

    # It's safer to work with a copy to avoid SettingWithCopyWarning
    df = master_gdf.copy()

    for column, prefix in CATEGORICAL_FEATURES_TO_ENCODE.items():
        if column in df.columns:
            logging.info(f"Encoding column: '{column}' with prefix: '{prefix}'")
            
            # Ensure the column is of a type that can be handled (e.g., string)
            # and fill NaNs with a placeholder string.
            df[column] = df[column].astype(str).fillna('unknown')
            
            # Use pandas get_dummies for one-hot encoding. This respects the index.
            dummies = pd.get_dummies(df[column], prefix=prefix, prefix_sep='_')
            
            # Clean up dummy column names to be more readable and valid
            dummies.columns = [
                col.replace(' ', '_').replace('-', '_').replace('.', '').replace('__', '_').upper()
                for col in dummies.columns
            ]

            logging.info(f"[REDACTED_BY_SCRIPT]")
            
            # Join the new dummy columns back to the main dataframe. This operation aligns on the index.
            df = df.join(dummies)
            
            # Drop the original categorical column as it's now redundant
            df.drop(column, axis=1, inplace=True)
            logging.info(f"[REDACTED_BY_SCRIPT]'{column}'.")

        else:
            logging.warning(f"Categorical column '{column}'[REDACTED_BY_SCRIPT]")

    # --- Finalization ---
    # The index 'hex_id' is preserved by the get_dummies and join operations.
    # No need to set it again if the initial check passed and operations were correct.
    if df.index.name != 'hex_id':
         logging.error("Critical error: 'hex_id'[REDACTED_BY_SCRIPT]")
         # Attempt to restore it, though this indicates a problem in the logic.
         df.index = master_gdf.index

    logging.info("[REDACTED_BY_SCRIPT]")
    
    return df
