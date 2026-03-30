"""
Directive 034 (Preparation): SSSI Unit L1 Artifact Creation

This script ingests the raw SSSI Unit Condition data, performs critical
sanitization, and saves it as a trusted, project-standard L1 artifact.

Its primary function is to create the `condition_ordinal` feature, which
quantifies the ecological health of each unit based on the mandated scale.
"""

import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"

# Mandated Ordinal Scale from Directive 034
CONDITION_ORDINAL_MAP = {
    'Favourable': 1,
    '[REDACTED_BY_SCRIPT]': 2,
    '[REDACTED_BY_SCRIPT]': 3,
    '[REDACTED_BY_SCRIPT]': 4,
    'Partially Destroyed': 5,
    'Destroyed': 5
}

logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    sssi_unit_raw_path = r"[REDACTED_BY_SCRIPT]"
    output_path = r"[REDACTED_BY_SCRIPT]"

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(sssi_unit_raw_path)

    # MANDATE 1: Paranoid CRS Verification.
    if gdf.crs != PROJECT_CRS:
        logging.warning(f"[REDACTED_BY_SCRIPT]'{PROJECT_CRS}', found '{gdf.crs}'. Forcing standard.")
        gdf = gdf.to_crs(PROJECT_CRS)
    else:
        logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE 2: Proactive Schema Normalization.
    gdf.columns = [col.lower().strip() for col in gdf.columns]

    # MANDATE 3: Geometric Validation & Repair.
    invalid_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_count > 0:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        gdf['geometry'] = gdf.geometry.buffer(0)

    # MANDATE 4: Create Ordinal Condition Feature.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf['condition_ordinal'] = gdf['condition'].map(CONDITION_ORDINAL_MAP).fillna(0).astype(int)
    
    # MANDATE 5: Performance Optimization.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(output_path, driver='GPKG', layer='sssi_units_england')

    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()