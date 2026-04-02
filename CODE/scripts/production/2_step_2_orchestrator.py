import sys
import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
import traceback
import json

# --- Project Setup ---
# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Executor Imports ---
# Import all the new, refactored executor modules.
from scripts.production.executors import (
    # --- Block 1: Environmental, Land Use & Planning Constraints ---
    exec_021_env_aw,
    exec_022_env_aonb,
    exec_024_env_sssi,
    exec_025_env_sssi_units,
    exec_027_nationalpark,
    exec_028_sac,
    exec_029_spa,
    exec_030_alc,
    exec_031_nationaltrail,
    exec_032_crow,
    exec_033_historicpark,
    exec_034_priority,
    exec_035_stewardship,
    exec_036_ess,
    exec_037_nhlc,

    # --- Block 2: DNO Stratification & Initial Enrichment ---
    exec_038_dno_stratification,

    # --- Block 3: National Grid (NG) Network Features ---
    exec_041_ng_assets,
    exec_042_ng_substation,
    exec_043_ng_distribution_substation,
    exec_044_ng_lct,
    exec_045_ng_dfes,

    # --- Block 4: UK Power Networks (UKPN) Network Features ---
    exec_051_ukpn_dfes,
    exec_052_ukpn_earthing,
    exec_053_ukpn_capacity,
    exec_054_ukpn_transformers,
    exec_055_ukpn_idno,
    exec_056_ukpn_ltds,
    exec_057_ukpn_lct,
    exec_058_ukpn_dnoa,
    exec_059_ukpn_pq,
    exec_060_ukpn_primary_transformers,

    # --- Block 5: Demographic & Infrastructure Features ---
    exec_061_demographic_lsoa,
    exec_062_demographic_oa,
    exec_063_infrastructure_proxies,
    exec_064_ohl,

    # --- Block 6: Historical Precedent ---
    exec_065_historical_precedent,
    exec_066_grid_synthesis,
    exec_067_land_use,
    exec_068_env_synthesis,


    # --- Block 7: Schema Cleaning & Imputation ---
    exec_070_clean_header,
    exec_040_imputation
)

# --- Configuration ---
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = r"[REDACTED_BY_SCRIPT]"
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{RUN_TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

LOG_FILE = os.path.join(RUN_DIR, "orchestrator_v4.log")
FINAL_OUTPUT_CSV = os.path.join(RUN_DIR, "[REDACTED_BY_SCRIPT]")
FINAL_METRICS_REPORT = os.path.join(RUN_DIR, "[REDACTED_BY_SCRIPT]") # Assuming imputation script generates this

# --- V4.1 State Management & Resumption Configuration ---
STATE_FILE = os.path.join(os.path.dirname(__file__), 'orchestrator_state.json')
# V6: No longer saving a separate recovery artifact on failure. We resume from the last successful checkpoint.

# --- V7 Debug Mode ---
# If set to a number, it will only process the first N rows of the dataset.
# Set to None for a full run.
DEBUG_SAMPLE_SIZE = None 

PROJECT_CRS = "EPSG:27700"
INPUT_GEOPACKAGE = r"[REDACTED_BY_SCRIPT]"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[REDACTED_BY_SCRIPT]',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def update_state(state):
    """[REDACTED_BY_SCRIPT]"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def read_state():
    """[REDACTED_BY_SCRIPT]"""
    if not os.path.exists(STATE_FILE):
        return {'resume_run_id': None, 'last_successful_executor': None, '[REDACTED_BY_SCRIPT]': None}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logging.warning("[REDACTED_BY_SCRIPT]")
        return {'resume_run_id': None, 'last_successful_executor': None, '[REDACTED_BY_SCRIPT]': None}
    

# --- V5 Pipeline Definition ---
# This defines the sequential order of execution for the entire feature engineering process.
PIPELINE_STAGES = [
    # --- Block 1: DNO Stratification, Environmental, Land Use & Planning Constraints ---
    exec_038_dno_stratification,
    exec_021_env_aw,
    exec_022_env_aonb,
    exec_024_env_sssi,
    exec_025_env_sssi_units,
    exec_027_nationalpark,
    exec_028_sac,
    exec_029_spa,
    exec_030_alc,
    exec_031_nationaltrail,
    exec_032_crow,
    exec_033_historicpark,
    exec_034_priority,
    exec_035_stewardship,
    exec_036_ess,
    exec_037_nhlc,
    
    # --- Block 2: Enrichment & DNO-Specific Processing ---
    #exec_039_enrichment,
    
    # --- NG Features ---
    exec_041_ng_assets,
    exec_042_ng_substation,
    exec_043_ng_distribution_substation,
    exec_044_ng_lct,
    exec_045_ng_dfes,

    # --- UKPN Features ---
    exec_051_ukpn_dfes,
    exec_052_ukpn_earthing,
    exec_053_ukpn_capacity,
    exec_054_ukpn_transformers,
    exec_055_ukpn_idno,
    exec_056_ukpn_ltds,
    exec_057_ukpn_lct,
    exec_058_ukpn_dnoa,
    exec_059_ukpn_pq,
    exec_060_ukpn_primary_transformers,

    # --- Demographic & Infrastructure Features ---
    exec_061_demographic_lsoa,
    exec_063_infrastructure_proxies,
    exec_064_ohl,

    exec_065_historical_precedent,
    exec_066_grid_synthesis,
    exec_067_land_use,
    exec_068_env_synthesis,
    
    # --- Block 3: Schema Cleaning & Imputation ---
    exec_070_clean_header,
    exec_040_imputation,
]

def main():
    """
    V4 Main Entry Point: A fully refactored, sequential feature engineering cascade.
    """
    global RUN_TIMESTAMP, RUN_DIR, LOG_FILE # Allow modification for resume
    
    master_gdf = None
    pipeline_to_run = PIPELINE_STAGES
    
    # --- State-Aware Startup Protocol ---
    state = read_state()
    if state.get('resume_run_id'):
        # --- RESUME MODE ---
        resume_run_id = state['resume_run_id']
        last_success_module_name = state['last_successful_executor']
        [REDACTED_BY_SCRIPT] = state.get('[REDACTED_BY_SCRIPT]') # Use .get for safety

        # Override current run identifiers with the one we are resuming
        RUN_TIMESTAMP = resume_run_id.replace('run_', '')
        RUN_DIR = os.path.join(BASE_OUTPUT_DIR, resume_run_id)
        LOG_FILE = os.path.join(RUN_DIR, "orchestrator_v4.log") # Append to existing log

        logging.info(f"[REDACTED_BY_SCRIPT]")

        # V6 RESUMPTION LOGIC: Resume from the last successful checkpoint.
        if not last_checkpoint_path or not os.path.exists(last_checkpoint_path):
            logging.critical(f"[REDACTED_BY_SCRIPT]")
            # Clear state to prevent loop
            update_state({'resume_run_id': None, 'last_successful_executor': None, '[REDACTED_BY_SCRIPT]': None})
            return

        logging.info(f"[REDACTED_BY_SCRIPT]")
        # V5.2 CSV RESUME FIX: When resuming from a CSV, the 'geometry' column is just a WKT string.
        # We must load it with pandas and then explicitly convert it to a GeoDataFrame.
        temp_df = pd.read_csv(last_checkpoint_path)
        from shapely import wkt
        # Ensure the geometry column is not empty and contains valid WKT strings
        if 'geometry' in temp_df.columns and not temp_df['geometry'].isnull().all():
            geometries = temp_df['geometry'].dropna().apply(wkt.loads)
            master_gdf = gpd.GeoDataFrame(temp_df, geometry=geometries, crs=PROJECT_CRS)
            logging.info("[REDACTED_BY_SCRIPT]")
        else:
            # If there's no geometry, just load as a DataFrame. This might happen in later stages.
            master_gdf = temp_df
            logging.info("[REDACTED_BY_SCRIPT]")

        # V13 FIX: The order of operations was incorrect. We must first validate that the key exists
        # in the columns, and only *after* all validation and restoration is complete, set it as the index.

        # --- V5 Authoritative Key Resilience (for Resumption) ---
        # Ensure the authoritative key exists, even when resuming from an older artifact
        # that was created before the 'fid' -> 'hex_id' rename step was added.
        if 'hex_id' not in master_gdf.columns:
            if 'fid' in master_gdf.columns:
                master_gdf.rename(columns={'fid': 'hex_id'}, inplace=True)
                logging.info("[REDACTED_BY_SCRIPT]'hex_id' from 'fid'[REDACTED_BY_SCRIPT]")
            else:
                # This is a critical failure if neither key is present in the recovery file.
                logging.critical("[REDACTED_BY_SCRIPT]'hex_id' and 'fid'. Cannot proceed.")
                # Clear state to prevent a restart loop on this broken artifact.
                update_state({'resume_run_id': None, 'last_successful_executor': None, '[REDACTED_BY_SCRIPT]': None})
                raise KeyError("[REDACTED_BY_SCRIPT]'hex_id' and 'fid'. Cannot proceed.")

        # # V12 FIX (Corrected Order): The authoritative key 'hex_id' must be set as the index during resumption.
        # # Many downstream executors rely on this index for joins and lookups.
        # if 'hex_id' in master_gdf.columns:
        #     master_gdf.set_index('hex_id', inplace=True)
        #     logging.info("Set 'hex_id'[REDACTED_BY_SCRIPT]")

        # V6.2 GEOMETRY RESILIENCE: If the checkpoint is a CSV that has lost its geometry,
        # we need to restore it by merging with the original grid centroids.
        if isinstance(master_gdf, pd.DataFrame) and 'geometry' not in master_gdf.columns:
            logging.warning("[REDACTED_BY_SCRIPT]'geometry'[REDACTED_BY_SCRIPT]")
            try:
                initial_grid = gpd.read_file(INPUT_GEOPACKAGE)
                if 'fid' in initial_grid.columns:
                    initial_grid.rename(columns={'fid': 'hex_id'}, inplace=True)
                
                # Keep only the essential columns for the merge
                initial_grid_geom = initial_grid[['hex_id', 'geometry']]

                # Perform a left merge to add the geometry back to the master dataframe
                master_gdf = pd.merge(master_gdf, initial_grid_geom, on='hex_id', how='left')
                
                # Convert back to a GeoDataFrame
                master_gdf = gpd.GeoDataFrame(master_gdf, geometry='geometry', crs=PROJECT_CRS)
                logging.info("[REDACTED_BY_SCRIPT]'geometry'[REDACTED_BY_SCRIPT]")

            except Exception as e:
                logging.critical(f"[REDACTED_BY_SCRIPT]'geometry' column. Error: {e}")
                update_state({'resume_run_id': None, 'last_successful_executor': None, '[REDACTED_BY_SCRIPT]': None})
                raise

        # --- V5 Authoritative Key Resilience (for Resumption) --- - THIS BLOCK IS NOW MOVED UP
        
        if last_success_module_name:
            logging.info(f"[REDACTED_BY_SCRIPT]'{last_success_module_name}'[REDACTED_BY_SCRIPT]")
            last_success_index = -1
            for i, module in enumerate(PIPELINE_STAGES):
                if module.__name__.split('.')[-1] == last_success_module_name:
                    last_success_index = i
                    break
            
            if last_success_index != -1:
                pipeline_to_run = PIPELINE_STAGES[last_success_index + 1:]
                logging.info(f"[REDACTED_BY_SCRIPT]'{pipeline_to_run[0].__name__.split('.')[-1]}'")
            else:
                logging.warning(f"[REDACTED_BY_SCRIPT]'{last_success_module_name}' in pipeline. Running full pipeline.")
        
    else:
        # --- FRESH RUN MODE ---
        logging.info(f"[REDACTED_BY_SCRIPT]")
        logging.info(f"[REDACTED_BY_SCRIPT]")
        
        logging.info(f"[REDACTED_BY_SCRIPT]")
        master_gdf = gpd.read_file(INPUT_GEOPACKAGE)
        
        # --- V5 Authoritative Key Creation ---
        # The 'fid' from the L1 grid is the definitive unique identifier.
        # We rename it to 'hex_id' to establish a clear, authoritative join key
        # that will be used by all subsequent enrichment modules.
        if 'hex_id' in master_gdf.columns:
            logging.info("[REDACTED_BY_SCRIPT]'hex_id' already exists.")
        elif 'fid' in master_gdf.columns:
            master_gdf.rename(columns={'fid': 'hex_id'}, inplace=True)
            logging.info("[REDACTED_BY_SCRIPT]'hex_id' from source 'fid'.")
        elif 'id' in master_gdf.columns:
            master_gdf.rename(columns={'id': 'hex_id'}, inplace=True)
            logging.info("[REDACTED_BY_SCRIPT]'hex_id' from source 'id'.")
        else:
            # This is a critical failure condition. If a key is not present, enrichment is impossible.
            logging.critical("[REDACTED_BY_SCRIPT]'hex_id', 'fid', or 'id'[REDACTED_BY_SCRIPT]")
            raise KeyError("Source GDF lacks 'hex_id', 'fid', or 'id'[REDACTED_BY_SCRIPT]")

        # --- V7 Debug Mode ---
        if DEBUG_SAMPLE_SIZE is not None:
            logging.warning(f"[REDACTED_BY_SCRIPT]")
            master_gdf = master_gdf.head(DEBUG_SAMPLE_SIZE)

        # V6.4: Set authoritative key as index for robust joins throughout the pipeline.
        master_gdf.set_index('hex_id', inplace=True)
        logging.info("Set 'hex_id'[REDACTED_BY_SCRIPT]")

        if master_gdf.crs.to_string() != PROJECT_CRS:
            logging.info(f"[REDACTED_BY_SCRIPT]")
            master_gdf = master_gdf.to_crs(PROJECT_CRS)
        logging.info(f"[REDACTED_BY_SCRIPT]")

        # Set the state for the new run
        update_state({'resume_run_id': f"run_{RUN_TIMESTAMP}", 'last_successful_executor': None, '[REDACTED_BY_SCRIPT]': None})

    try:
        # --- Stage 1-N: Sequential Feature Cascade ---
        for stage_module in tqdm(pipeline_to_run, desc="[REDACTED_BY_SCRIPT]"):
            stage_name = stage_module.__name__.split('.')[-1]
            logging.info(f"[REDACTED_BY_SCRIPT]")
            
            # Pass the dataframe to the executor and receive the transformed version.
            master_gdf = stage_module.execute(master_gdf)

            # --- V5.1 Column Integrity Check ---
            # After each execution, ensure the authoritative join key has not been dropped.
            # Some pandas operations, especially within executors, might inadvertently
            # return a dataframe without the original index/columns.
            if 'hex_id' not in master_gdf.columns:
                logging.warning(f"Authoritative key 'hex_id' was lost after stage '{stage_name}'.")
                # Attempt to restore it from the index, assuming the index is still the key.
                if master_gdf.index.name == 'hex_id' or master_gdf.index.name == 'fid':
                     master_gdf.reset_index(inplace=True)
                     logging.info("[REDACTED_BY_SCRIPT]'hex_id'[REDACTED_BY_SCRIPT]")
                # This is a fallback if the index is just a range. It's not guaranteed to be correct.
                elif 'fid' in master_gdf.columns:
                    master_gdf.rename(columns={'fid': 'hex_id'}, inplace=True)
                    logging.info("[REDACTED_BY_SCRIPT]'hex_id' by renaming 'fid'.")
                else:
                    logging.error(f"[REDACTED_BY_SCRIPT]'hex_id' after stage '{stage_name}'. Aborting.")
                    raise KeyError(f"Authoritative key 'hex_id'[REDACTED_BY_SCRIPT]")
            
            logging.info(f"[REDACTED_BY_SCRIPT]")

            # V6 ATOMIC CHECKPOINT: Save successful state and update the state file.
            checkpoint_path = os.path.join(RUN_DIR, f"[REDACTED_BY_SCRIPT]")
            master_gdf.to_csv(checkpoint_path, index=False)
            logging.info(f"[REDACTED_BY_SCRIPT]")

            update_state({
                'resume_run_id': f"run_{RUN_TIMESTAMP}",
                'last_successful_executor': stage_name,
                '[REDACTED_BY_SCRIPT]': checkpoint_path
            })

        # --- Finalization on Success ---
        logging.info("[REDACTED_BY_SCRIPT]")
        
        if 'geometry' in master_gdf.columns:
            master_gdf = master_gdf.drop(columns=['geometry'])
            
        master_gdf.to_csv(FINAL_OUTPUT_CSV, index=False)
        logging.info(f"[REDACTED_BY_SCRIPT]")

        # Clear state to signal a successful run.
        update_state({'resume_run_id': None, 'last_successful_executor': None, '[REDACTED_BY_SCRIPT]': None})

    except Exception as e:
        logging.critical(f"[REDACTED_BY_SCRIPT]")
        logging.critical(traceback.format_exc())

        # V6: No longer save a failure artifact. The state file already points to the last *successful* run.
        # The user can simply re-run the script to resume from the last good checkpoint.
        logging.error("[REDACTED_BY_SCRIPT]")
        return

    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()