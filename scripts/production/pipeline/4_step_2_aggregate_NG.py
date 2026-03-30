import geopandas as gpd
from pathlib import Path
import logging
import pandas as pd
import re
import fiona

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
# Corrected RAW_DATA_ROOT to match the true nested directory structure.
RAW_DATA_ROOT = Path(r"[REDACTED_BY_SCRIPT]")
OUTPUT_DIR = Path("[REDACTED_BY_SCRIPT]")
TARGET_CRS = "EPSG:27700"
VERSION = "1.0"

# Updated lists for internal consistency and to match source data reality.
VOLTAGES = ["11KV", "33KV", "66KV", "132KV"]
ASSET_TYPES = ["cable", "ohl", "poles", "substation", "tower", "transformer"]

# --- SCRIPT ---
def to_snake_case(name):
    """[REDACTED_BY_SCRIPT]"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def aggregate_and_synthesize():
    """[REDACTED_BY_SCRIPT]"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for voltage in VOLTAGES:
        for asset in ASSET_TYPES:
            logging.info(f"[REDACTED_BY_SCRIPT]")
            
            # Hardened, case-insensitive file discovery logic.
            voltage_dir = RAW_DATA_ROOT / voltage
            if not voltage_dir.exists():
                logging.warning(f"[REDACTED_BY_SCRIPT]")
                continue
            
            all_files_in_dir = voltage_dir.glob(f"*_{voltage}.shp")
            pattern = re.compile(f"[REDACTED_BY_SCRIPT]", re.IGNORECASE)
            file_paths = [f for f in all_files_in_dir if pattern.match(f.name)]

            if not file_paths:
                logging.warning(f"[REDACTED_BY_SCRIPT]")
                continue

            try:
                # Use Fiona's environment context manager to ensure the SHX fix is applied.
                with fiona.Env(SHAPE_RESTORE_SHX="YES"):
                    # Refactored ingestion loop to handle CRS assignment before unification.
                    list_of_gdfs = []
                    for fp in file_paths:
                        # Explicitly use the 'fiona' engine to respect the fiona.Env context.
                        gdf = gpd.read_file(fp, engine="fiona")
                        
                        # Defensive CRS Assignment: Correct missing metadata on the fly.
                        if gdf.crs is None:
                            logging.warning(f"[REDACTED_BY_SCRIPT]")
                            gdf.set_crs(TARGET_CRS, inplace=True)
                        
                        # Uncompromising CRS Unification: Ensure all data conforms.
                        gdf = gdf.to_crs(TARGET_CRS)
                        list_of_gdfs.append(gdf)

                # Concatenate into a single master GeoDataFrame
                master_gdf = pd.concat(list_of_gdfs, ignore_index=True)
                
                if master_gdf.empty:
                    logging.warning(f"[REDACTED_BY_SCRIPT]")
                    continue

                # --- Synthesis & Indexing (Phase 3) ---
                
                # 1. Proactive Schema Normalization
                master_gdf.columns = [to_snake_case(col) for col in master_gdf.columns]

                # 2. Mandated Spatial Indexing
                _ = master_gdf.sindex


                # 4. Artifact Serialization
                asset_snake_case = to_snake_case(asset)
                output_path = OUTPUT_DIR / f"[REDACTED_BY_SCRIPT]"

                # 4. Save the GeoDataFrame to a GeoParquet file
                master_gdf.to_parquet(output_path)

                logging.info(f"[REDACTED_BY_SCRIPT]")

            except Exception as e:
                logging.error(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    aggregate_and_synthesize()