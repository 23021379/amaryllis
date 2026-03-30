import geopandas as gpd
from pathlib import Path
import random
import logging
import os

# Set the SHAPE_RESTORE_SHX config option to YES to allow recreating the .shx file if it's missing
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
# Adjusted RAW_DATA_ROOT to match the provided nested directory structure.
RAW_DATA_ROOT = Path(r"[REDACTED_BY_SCRIPT]")
REPORT_PATH = Path("[REDACTED_BY_SCRIPT]")
TARGET_CRS = "EPSG:27700"
SAMPLE_SIZE = 5

import re

# Canonical lists updated for internal consistency. All asset types are now lowercase.
VOLTAGES = ["11KV", "33KV", "66KV", "132KV"]
ASSET_TYPES = ["cable", "ohl", "poles", "substation", "tower", "transformer"]

# --- SCRIPT ---
def audit_asset_group(voltage, asset_type):
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    voltage_dir = RAW_DATA_ROOT / voltage
    if not voltage_dir.exists():
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return [], []

    # Use a broad glob and then a precise, case-insensitive regex to filter.
    # This is more robust than a simple case-sensitive glob pattern.
    all_files_in_dir = voltage_dir.glob(f"*_{voltage}.shp")
    
    # The regex pattern will match `GRIDREF_asset_VOLTAGE.shp`, ignoring case for the asset part.
    pattern = re.compile(f"[REDACTED_BY_SCRIPT]", re.IGNORECASE)
    
    file_paths = [f for f in all_files_in_dir if pattern.match(f.name)]

    if not file_paths:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return [], []

    # Sample files as per the blueprint mandate
    sample_paths = random.sample(file_paths, min(len(file_paths), SAMPLE_SIZE))
    
    failures = []
    warnings = []
    
    # Establish the reference schema from the first file in the sample
    try:
        # CORRECTED: Pass only the first file path to read_file.
        reference_gdf = gpd.read_file(sample_paths[0], engine="pyogrio")
        reference_schema = reference_gdf.columns.to_list()
        reference_dtypes = reference_gdf.dtypes.to_dict()
    except Exception as e:
        msg = f"[REDACTED_BY_SCRIPT]"
        logging.error(msg)
        failures.append(msg)
        return failures, warnings

    for fp in sample_paths:
        try:
            gdf = gpd.read_file(fp, engine="pyogrio")

            # Defensive CRS Assignment: If CRS is missing, assign the known project standard.
            if gdf.crs is None:
                logging.warning(f"[REDACTED_BY_SCRIPT]")
                gdf.set_crs(TARGET_CRS, inplace=True)
            
            # 1. CRS Verification
            if gdf.crs != TARGET_CRS:
                msg = f"[REDACTED_BY_SCRIPT]"
                logging.error(msg)
                failures.append(msg)

            # 2. Schema Consistency Verification
            current_schema = gdf.columns.to_list()
            if current_schema != reference_schema:
                msg = f"[REDACTED_BY_SCRIPT]"
                logging.error(msg)
                failures.append(msg)

            # 3. Geometry Integrity Verification
            if gdf.geometry.isnull().any():
                msg = f"[REDACTED_BY_SCRIPT]"
                logging.warning(msg)
                warnings.append(msg)
            if not gdf.geometry.is_valid.all():
                msg = f"[REDACTED_BY_SCRIPT]"
                logging.warning(msg)
                warnings.append(msg)

        except Exception as e:
            msg = f"[REDACTED_BY_SCRIPT]"
            logging.error(msg)
            failures.append(msg)
            
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return failures, warnings

def main():
    """[REDACTED_BY_SCRIPT]"""
    REPORT_PATH.parent.mkdir(exist_ok=True)
    all_failures = []
    all_warnings = []

    for voltage in VOLTAGES:
        for asset in ASSET_TYPES:
            failures, warnings = audit_asset_group(voltage, asset)
            all_failures.extend(failures)
            all_warnings.extend(warnings)
    
    with open(REPORT_PATH, 'w') as f:
        f.write("[REDACTED_BY_SCRIPT]")
        if not all_failures:
            f.write("STATUS: SUCCESS\n")
            f.write("[REDACTED_BY_SCRIPT]")
        else:
            f.write("[REDACTED_BY_SCRIPT]")
            f.write("[REDACTED_BY_SCRIPT]")
            f.write("[REDACTED_BY_SCRIPT]")
            for failure in all_failures:
                f.write(f"- {failure}\n")
            f.write("\n")

        if all_warnings:
            f.write("--- WARNINGS ---\n")
            f.write("[REDACTED_BY_SCRIPT]")
            for warning in all_warnings:
                f.write(f"- {warning}\n")
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    if all_failures:
        logging.error("[REDACTED_BY_SCRIPT]")
    else:
        logging.info("AUDIT SUCCESSFUL.")

if __name__ == "__main__":
    main()