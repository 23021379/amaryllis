"""
Directive 033 (Preparation): SSSI L1 Artifact Creation

This script ingests the raw Sites of Special Scientific Interest (SSSI) data,
performs critical sanitization, and saves it as a trusted, project-standard L1
artifact.

This adheres to the Amaryllis Doctrine by:
1.  Paranoid CRS Verification: Verifying the stated EPSG:27700 CRS.
2.  Defensive Data Handling: Repairing geometries and normalizing schemas.
3.  Artifact Integrity: Creating a clean, reusable GeoPackage as the single
    source of truth for SSSI data.
"""

import logging
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CRS = "EPSG:27700"

logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    sssi_raw_path = r"[REDACTED_BY_SCRIPT]"
    output_path = r"[REDACTED_BY_SCRIPT]"

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(sssi_raw_path)

    # MANDATE 1: Paranoid CRS Verification.
    if gdf.crs != PROJECT_CRS:
        logging.warning(f"[REDACTED_BY_SCRIPT]'{PROJECT_CRS}', found '{gdf.crs}'[REDACTED_BY_SCRIPT]")
        gdf = gdf.to_crs(PROJECT_CRS)
    else:
        logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE 2: Proactive Schema Normalization.
    gdf.columns = [col.lower().strip() for col in gdf.columns]

    # MANDATE 3: Geometric Validation & Repair.
    logging.info("[REDACTED_BY_SCRIPT]")
    invalid_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_count > 0:
        gdf['geometry'] = gdf.geometry.buffer(0)
        logging.warning(f"[REDACTED_BY_SCRIPT]")
    
    # MANDATE 4: Performance Optimization.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(output_path, driver='GPKG', layer='sssi_england')

    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()