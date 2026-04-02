import logging
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

def main():
    """
    Orchestrates the creation of the CRoW Access Layer L1 artifact per Directive 040.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Define paths
    raw_data_path = Path(r"[REDACTED_BY_SCRIPT]")
    output_path = Path(r"[REDACTED_BY_SCRIPT]")

    # --- Ingestion & Sanitization ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(raw_data_path)

    logging.info(f"Source CRS is '{gdf.crs}'[REDACTED_BY_SCRIPT]'{PROJECT_CRS}'.")
    gdf = gdf.to_crs(PROJECT_CRS)

    gdf.columns = [col.lower().strip() for col in gdf.columns]
    logging.info("Schema normalized.")

    invalid_geom_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_geom_count > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf['geometry'] = gdf.geometry.buffer(0)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    # --- Persistence ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(output_path, driver='GPKG', layer='crow_access_england')

    logging.info("[REDACTED_BY_SCRIPT]")


def main2():
    """
    Orchestrates the creation of the CRoW Section 4 (RCL) L1 artifact per Directive 040.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Define paths
    raw_data_path = Path(r"[REDACTED_BY_SCRIPT]")
    output_path = Path(r"[REDACTED_BY_SCRIPT]")

    # --- Ingestion & Sanitization ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(raw_data_path)

    logging.info(f"Source CRS is '{gdf.crs}'[REDACTED_BY_SCRIPT]'{PROJECT_CRS}'.")
    gdf = gdf.to_crs(PROJECT_CRS)

    gdf.columns = [col.lower().strip() for col in gdf.columns]
    logging.info("Schema normalized.")
    
    invalid_geom_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_geom_count > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf['geometry'] = gdf.geometry.buffer(0)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    # --- Persistence ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(output_path, driver='GPKG', layer='crow_s4_rcl_england')

    logging.info("[REDACTED_BY_SCRIPT]")


def main3():
    """
    Orchestrates the creation of the CRoW Section 15 L1 artifact per Directive 040.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Define paths
    raw_data_path = Path(r"[REDACTED_BY_SCRIPT]")
    output_path = Path(r"[REDACTED_BY_SCRIPT]")

    # --- Ingestion & Sanitization ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(raw_data_path)

    logging.info(f"Source CRS is '{gdf.crs}'[REDACTED_BY_SCRIPT]'{PROJECT_CRS}'.")
    gdf = gdf.to_crs(PROJECT_CRS)

    gdf.columns = [col.lower().strip() for col in gdf.columns]
    logging.info("Schema normalized.")

    invalid_geom_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_geom_count > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf['geometry'] = gdf.geometry.buffer(0)

    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    # --- Persistence ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(output_path, driver='GPKG', layer='crow_s15_england')

    logging.info("[REDACTED_BY_SCRIPT]")


if __name__ == "__main__":
    main()
    main2()
    main3()