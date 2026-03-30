import logging
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

def main():
    """
    Orchestrates the creation of the National Trails L1 artifact per Directive 039.
    This script ingests the raw National Trails data, performs critical sanitization,
    and saves it as a trusted, project-standard L1 GeoPackage artifact.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Define paths
    raw_data_path = Path(r"[REDACTED_BY_SCRIPT]")
    output_path = Path(r"[REDACTED_BY_SCRIPT]")

    # --- Ingestion ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(raw_data_path)

    # --- Sanitization Mandates ---

    # MANDATE 1: Uncompromising CRS Unification.
    logging.info(f"Source CRS is '{gdf.crs}'[REDACTED_BY_SCRIPT]'{PROJECT_CRS}'.")
    gdf = gdf.to_crs(PROJECT_CRS)
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE 2: Proactive Schema Normalization and Distillation.
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    # Distill to only necessary columns as per directive
    gdf = gdf[['name', 'geometry']].copy()
    gdf.rename(columns={'name': 'nt_name'}, inplace=True)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # MANDATE 3: Performance Optimization.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    # --- Persistence ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(output_path, driver='GPKG', layer='national_trails_england')

    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()