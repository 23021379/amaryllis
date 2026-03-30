import logging
import geopandas as gpd
import pandas as pd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
# As per directive, Grades 1, 2, and 3 are considered Best and Most Versatile (BMV)
BMV_GRADES = [1, 2, 3]
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

def main():
    """
    Orchestrates the creation of the ALC L1 artifact per Directive 038.
    This script ingests the raw ALC data, performs critical sanitization,
    enriches it with policy-aligned features, and saves it as a trusted L1 artifact.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Define paths
    raw_data_path = Path(r"[REDACTED_BY_SCRIPT]")
    output_path = Path(r"[REDACTED_BY_SCRIPT]")

    # --- Ingestion ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(raw_data_path)

    # --- Sanitization & Enrichment Mandates ---

    # MANDATE 1: Uncompromising CRS Unification.
    logging.info(f"Source CRS is '{gdf.crs}'[REDACTED_BY_SCRIPT]'{PROJECT_CRS}'.")
    gdf = gdf.to_crs(PROJECT_CRS)
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE 2: Proactive Schema Normalization.
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # MANDATE 3: Geometric Validation & Repair.
    invalid_geom_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_geom_count > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf['geometry'] = gdf.geometry.buffer(0)
    else:
        logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE 4: Synthetic Feature Enrichment.
    logging.info("[REDACTED_BY_SCRIPT]")
    # Extract numeric grade from string (e.g., "Grade 1" -> 1)
    gdf['alc_grade'] = gdf['alc_grade'].str.extract(r'(\d)').astype(float).astype('Int64')
    # Create the critical policy-aligned BMV binary flag
    gdf['is_bmv'] = gdf['alc_grade'].isin(BMV_GRADES).astype(int)
    logging.info(f"[REDACTED_BY_SCRIPT]'is_bmv'[REDACTED_BY_SCRIPT]")

    # MANDATE 5: Performance Optimization.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    # --- Persistence ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(output_path, driver='GPKG', layer='alc_england')

    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()