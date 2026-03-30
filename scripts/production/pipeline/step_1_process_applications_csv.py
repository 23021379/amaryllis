import pandas as pd
import logging
import sys

# MANDATE 3.5: Configure Auditable Logging
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# --- CONFIGURATION ---
# NOTE: The user must replace this with the actual path to their source data.
SOURCE_CSV_PATH = r'[REDACTED_BY_SCRIPT]'
OUTPUT_CSV_PATH = r'[REDACTED_BY_SCRIPT]'

# MANDATE 3.1: Define required columns and their new snake_case names
COLUMN_MAPPING = {
    'Planning Authority': 'planning_authority',
    'Technology Type': 'technology_type',
    'Storage Type': 'storage_type',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'CHP Enabled': 'chp_enabled',
    '[REDACTED_BY_SCRIPT]': 'ro_banding_roc_mwh',
    'FiT Tariff (p/kWh)': 'fit_tariff_p_kwh',
    'CfD Capacity (MW)': 'cfd_capacity_mw',
    'Turbine Capacity': 'turbine_capacity',
    'No. of Turbines': 'no_of_turbines',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'X-coordinate': 'easting',  # MANDATE 3.4
    'Y-coordinate': 'northing', # MANDATE 3.4
    'Secretary of State Reference': 'secretary_of_state_reference',
    '[REDACTED_BY_SCRIPT]': 'type_of_sos_intervention',
    'Judicial Review': 'judicial_review',
    'Offshore Wind Round': 'offshore_wind_round',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'Appeal Lodged': 'appeal_lodged',
    'Appeal Withdrawn': 'appeal_withdrawn',
    'Appeal Refused': 'appeal_refused',
    'Appeal Granted': 'appeal_granted',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    '[REDACTED_BY_SCRIPT]': 'sos_intervened',
    '[REDACTED_BY_SCRIPT]': 'sos_refusal',
    '[REDACTED_BY_SCRIPT]': 'sos_granted',
    '[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]',
    'Under Construction': 'under_construction',
    'Operational': 'operational',
    'Heat Network Ref': 'heat_network_ref',
    '[REDACTED_BY_SCRIPT]': 'solar_site_area_sqm'
}

# Define which columns should be dates or numeric for later conversion
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

def ingest_repd_data(source_path: str) -> pd.DataFrame:
    """
    Loads, subsets, and renames data from the source REPD CSV file.
    Implements Mandates 3.1 and 3.2 (initial load).
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    try:
        # MANDATE 3.1: Use `usecols` to load only required data.
        # MANDATE 3.2: Use `dtype=str` to prevent pandas type inference errors.
        df = pd.read_csv(
            source_path,
            usecols=COLUMN_MAPPING.keys(),
            dtype=str,
            encoding='latin-1'
        )
        logging.info(f"[REDACTED_BY_SCRIPT]")
        # MANDATE 3.1: Rename columns to snake_case.
        df = df.rename(columns=COLUMN_MAPPING)
        return df
    except FileNotFoundError:
        logging.error(f"[REDACTED_BY_SCRIPT]'{source_path}'. Terminating.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

def clean_and_normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs string normalization and paranoid type casting.
    Implements Mandates 3.2, 3.3, and 3.5 (logging).
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE 3.3: String Normalisation
    for col in df.columns:
        # Ensure we only apply string operations to object columns
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\r?\n', ' ', regex=True)

    # MANDATE 3.2: Paranoid Type Casting & MANDATE 3.5: Auditable Logging
    for col in NUMERIC_COLS:
        initial_nulls = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        final_nulls = df[col].isnull().sum()
        coerced_count = final_nulls - initial_nulls
        if coerced_count > 0:
            logging.warning(f"Column '{col}'[REDACTED_BY_SCRIPT]")

    for col in DATE_COLS:
        initial_nulls = df[col].isnull().sum()
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
        final_nulls = df[col].isnull().sum()
        coerced_count = final_nulls - initial_nulls
        if coerced_count > 0:
            logging.warning(f"Column '{col}'[REDACTED_BY_SCRIPT]")

    logging.info("[REDACTED_BY_SCRIPT]")
    return df

def main():
    """
    Main execution pipeline for the ingestion script.
    """
    df = ingest_repd_data(SOURCE_CSV_PATH)
    df_cleaned = clean_and_normalize_data(df)

    # Final logging and output
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    try:
        df_cleaned.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        logging.info(f"[REDACTED_BY_SCRIPT]'{OUTPUT_CSV_PATH}'.")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

if __name__ == "__main__":
    main()