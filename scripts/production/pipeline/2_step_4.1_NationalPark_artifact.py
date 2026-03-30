import logging
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

def main():
    """
    Orchestrates the creation of the National Parks L1 artifact.
    This script ingests the raw National Park data, performs critical
    sanitization (CRS unification, schema normalization, geometric repair),
    and saves it as a trusted, project-standard L1 GeoPackage artifact.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Define paths using Path objects for robustness
    raw_data_path = Path(r"[REDACTED_BY_SCRIPT]")
    output_path = Path(r"[REDACTED_BY_SCRIPT]")

    # --- Ingestion ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(raw_data_path)

    # --- Sanitization Mandates ---

    # MANDATE 1: Uncompromising CRS Unification. Neutralize hostile source CRS.
    logging.info(f"Source CRS is '{gdf.crs}'[REDACTED_BY_SCRIPT]'{PROJECT_CRS}'.")
    gdf = gdf.to_crs(PROJECT_CRS)
    logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE 2: Proactive Schema Normalization.
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # MANDATE 3: Geometric Validation & Repair. Mitigate risk from complex polygons.
    invalid_geoms = gdf.geometry.is_valid.sum()
    if invalid_geoms < len(gdf):
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf['geometry'] = gdf.geometry.buffer(0)
    else:
        logging.info("[REDACTED_BY_SCRIPT]")

    # MANDATE 4: Performance Optimization. Critical for large, complex polygons.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    # --- Persistence ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    # Using 'name' as the layer name for consistency
    gdf.to_file(output_path, driver='GPKG', layer='[REDACTED_BY_SCRIPT]')

    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()