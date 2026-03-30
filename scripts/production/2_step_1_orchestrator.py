import sys
import os

# This adjustment ensures that the script can find the 'scripts' package
# when run from its location, resolving the ModuleNotFoundError.
# It adds the project's root directory ('renewables') to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pyarrow
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import os
from datetime import datetime
from scripts.production.executors import (
    exec_001_dfes_headroom,
    exec_002_earthing,
    exec_003_capacity,
    exec_004_transformers,
    exec_005_idnos,
    exec_006_ltds,
    exec_008_lct,
    exec_008_lct_2,
    exec_009_dnoa,
    exec_010_power_quality,
    exec_011_transformers,
    exec_012_gandp,
    exec_013_ohl,
    exec_014_tandp,
    exec_015_tandp,
    exec_016_ohl_tandp_hv,
    exec_017_ohl_tandp_lv,
    exec_018_sub_headroom,
    exec_019_secondary_sites,
    exec_020_secondary_sites_2,
    # Environmental & Land Use Executors
    exec_021_env_aw,
    exec_022_env_aonb,
    exec_024_env_sssi,
    exec_025_integrate_sssi,
    exec_026_integrate_sssi_units,
    exec_027_nationalpark,
    exec_028_sac,
    exec_029_spa,
    exec_030_alc,
    exec_031_national_trail,
    exec_032_crow,
    exec_032_historicpark,
    exec_033_priority,
    exec_034_stewardship,
    exec_035_ess,
    exec_036_nhlc,
    exec_037_ng_transformers_tandp_ohl,
    exec_038_ng_substation,
    exec_039_ng_substations_2,
    exec_040_ng_substations_3,
    exec_042_ng_lct,
    exec_043_ng_dfes,
    exec_044_demo_lsoa,
    exec_045_demo_oa,
    exec_046_demo_infra_2_1,
    exec_047_demo_infra_2_2,
    exec_048_demo_infra_2_3,
    exec_049_demo_infra_2_4,
    # Capstone Synthesis (Must run after all other env modules)
    exec_023_env_synthesis
)
import traceback


# --- Configuration ---
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = r"[REDACTED_BY_SCRIPT]"
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, f"run_{RUN_TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True) # Create the unique directory for this run

LOG_FILE = os.path.join(RUN_DIR, "orchestrator.log")
FINAL_OUTPUT_CSV = os.path.join(RUN_DIR, "[REDACTED_BY_SCRIPT]")
CHECKPOINT_CSV = os.path.join(RUN_DIR, "checkpoint.csv") # For resumption

# --- Source Data Configuration ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
GRID_CENSUS_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")
PROJECT_CRS = "EPSG:27700"
NUM_WORKERS = 1

# --- Logging Setup (Mandate 2) ---
# This now logs to a run-specific file with process info.
logging.basicConfig(
    level=logging.INFO,
    format='[REDACTED_BY_SCRIPT]',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() # Also print to console
    ]
)


# --- Worker Initialization (Mandate 1) ---
def initialize_worker():
    """
    This function is called ONCE per worker process.
    It loads all necessary data into the process's global memory.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    # Each executor's _initialize_module_state() function will be called here.
    # This ensures data is loaded only once per process, not per task.
    exec_001_dfes_headroom._initialize_module_state()
    exec_002_earthing._initialize_module_state()
    exec_003_capacity._initialize_module_state()
    exec_004_transformers._initialize_module_state()
    exec_005_idnos._initialize_module_state()
    exec_006_ltds._initialize_module_state()
    exec_008_lct._initialize_module_state()
    exec_008_lct_2._initialize_module_state()
    exec_009_dnoa._initialize_module_state()
    exec_010_power_quality._initialize_module_state()
    exec_011_transformers._initialize_module_state()
    exec_012_gandp._initialize_module_state()
    exec_013_ohl._initialize_module_state()
    exec_014_tandp._initialize_module_state()
    exec_015_tandp._initialize_module_state()
    exec_016_ohl_tandp_hv._initialize_module_state()
    exec_017_ohl_tandp_lv._initialize_module_state()
    exec_018_sub_headroom._initialize_module_state()
    exec_019_secondary_sites._initialize_module_state()
    exec_020_secondary_sites_2._initialize_module_state()
    exec_021_env_aw._initialize_module_state()
    exec_022_env_aonb._initialize_module_state()
    exec_024_env_sssi._initialize_module_state()
    exec_025_integrate_sssi._initialize_module_state()
    exec_026_integrate_sssi_units._initialize_module_state()
    exec_027_nationalpark._initialize_module_state()
    exec_028_sac._initialize_module_state()
    exec_029_spa._initialize_module_state()
    exec_030_alc._initialize_module_state()
    exec_031_national_trail._initialize_module_state()
    exec_032_crow._initialize_module_state()
    exec_032_historicpark._initialize_module_state()
    exec_033_priority._initialize_module_state()
    exec_034_stewardship._initialize_module_state()
    exec_035_ess._initialize_module_state()
    exec_036_nhlc._initialize_module_state()
    exec_037_ng_transformers_tandp_ohl._initialize_module_state()
    exec_038_ng_substation._initialize_module_state()
    exec_039_ng_substations_2._initialize_module_state()
    exec_040_ng_substations_3._initialize_module_state()
    exec_042_ng_lct._initialize_module_state()
    exec_043_ng_dfes._initialize_module_state()
    exec_044_demo_lsoa._initialize_module_state()
    exec_045_demo_oa._initialize_module_state()
    exec_046_demo_infra_2_1._initialize_module_state()
    exec_047_demo_infra_2_2._initialize_module_state()
    exec_048_demo_infra_2_3._initialize_module_state()
    # exec_049 does not have an initializer as it only synthesizes existing features.
    # exec_023 does not have an initializer as it only synthesizes existing features.
    logging.info("[REDACTED_BY_SCRIPT]")


def setup_task_list(completed_hex_ids_to_skip=None):
    """
    Loads the L1 prediction grid and generates the list of state dictionaries.
    This list can be filtered to skip already-completed tasks for run resumption.
    """
    if completed_hex_ids_to_skip is None:
        completed_hex_ids_to_skip = set()

    logging.info(f"[REDACTED_BY_SCRIPT]")
    grid_gdf = gpd.read_file(GRID_CENSUS_ARTIFACT)
    
    assert grid_gdf.crs.to_string() == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"

    temporal_anchor_date = datetime(2025, 6, 30)

    tasks = []
    for _, row in grid_gdf.iterrows():
        if row['hex_id'] not in completed_hex_ids_to_skip:
            tasks.append({
                'hex_id': row['hex_id'],
                'input_geom': row['geometry'],
                'dno': row['dno'],
                'date_validated': temporal_anchor_date,
                'features': {},
                'errors': []
            })
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return tasks

# The full, 36-stage feature engineering pipeline sequence.
PIPELINE_EXECUTORS = [
    # --- Grid & Network Topology Block ---
    exec_001_dfes_headroom.execute,
    exec_002_earthing.execute,
    exec_003_capacity.execute,
    exec_004_transformers.execute,
    exec_005_idnos.execute,
    exec_006_ltds.execute,
    exec_008_lct.execute,
    exec_008_lct_2.execute,
    exec_009_dnoa.execute,
    exec_010_power_quality.execute,
    exec_011_transformers.execute,
    exec_012_gandp.execute,
    exec_013_ohl.execute,
    exec_014_tandp.execute,
    exec_015_tandp.execute,
    exec_016_ohl_tandp_hv.execute,
    exec_017_ohl_tandp_lv.execute,
    exec_018_sub_headroom.execute,
    exec_019_secondary_sites.execute,
    exec_020_secondary_sites_2.execute,

    # --- Environmental, Land Use & Planning Constraints Block ---
    # Individual constraints must run before the final synthesis.
    exec_021_env_aw.execute,
    exec_022_env_aonb.execute,
    exec_024_env_sssi.execute,
    exec_025_integrate_sssi.execute,
    exec_026_integrate_sssi_units.execute,
    exec_027_nationalpark.execute,
    exec_028_sac.execute,
    exec_029_spa.execute,
    exec_030_alc.execute,
    exec_031_national_trail.execute,
    exec_032_crow.execute,
    exec_032_historicpark.execute,
    exec_033_priority.execute,
    exec_034_stewardship.execute,
    exec_035_ess.execute,
    exec_036_nhlc.execute,

    # --- NG Asset Data Block ---
    exec_037_ng_transformers_tandp_ohl.execute,
    exec_038_ng_substation.execute,
    exec_039_ng_substations_2.execute,
    exec_040_ng_substations_3.execute,
    exec_042_ng_lct.execute,
    exec_043_ng_dfes.execute,

    # --- Demographic & Infrastructure Block ---
    exec_044_demo_lsoa.execute,
    exec_045_demo_oa.execute,
    exec_046_demo_infra_2_1.execute,
    exec_047_demo_infra_2_2.execute,
    exec_048_demo_infra_2_3.execute,
    exec_049_demo_infra_2_4.execute,
    
    # Capstone synthesis runs LAST, using features from all preceding env modules.
    exec_023_env_synthesis.execute
]


def process_single_point(initial_state: dict) -> dict:
    """
    Worker function that processes a single point through the entire executor pipeline.
    This function is executed by each process in the multiprocessing pool.
    """
    current_state = initial_state
    try:
        for executor_func in PIPELINE_EXECUTORS:
            current_state = executor_func(current_state)
    except Exception:
        # Capture the full traceback for precise debugging.
        error_msg = f"[REDACTED_BY_SCRIPT]'hex_id'[REDACTED_BY_SCRIPT]"
        logging.error(error_msg)
        current_state['errors'].append(error_msg)
    
    # Return just the essential results to manage memory
    return {
        'hex_id': current_state['hex_id'],
        'features': current_state['features'],
        'errors': current_state['errors']
    }

def main():
    """
    Main entry point for the Amaryllis On-Demand Feature Factory.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- Task Setup with Resumption Logic (Mandate 3) ---
    completed_hex_ids = set()
    if os.path.exists(CHECKPOINT_CSV):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        try:
            df_completed = pd.read_csv(CHECKPOINT_CSV)
            if 'hex_id' in df_completed.columns:
                completed_hex_ids = set(df_completed['hex_id'])
                logging.info(f"[REDACTED_BY_SCRIPT]")
            else:
                logging.warning("[REDACTED_BY_SCRIPT]'hex_id'[REDACTED_BY_SCRIPT]")
        except pd.errors.EmptyDataError:
            logging.warning("[REDACTED_BY_SCRIPT]")

    tasks = setup_task_list(completed_hex_ids_to_skip=completed_hex_ids)
    
    if not tasks:
        logging.info("[REDACTED_BY_SCRIPT]")
        return
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # --- Multiprocessing Pool with Initializer (Mandate 1 & 3) ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Open the checkpoint file in append mode. Write header if it's new.
    write_header = not os.path.exists(CHECKPOINT_CSV) or os.path.getsize(CHECKPOINT_CSV) == 0
    
    try:
        with open(CHECKPOINT_CSV, 'a', newline='') as checkpoint_file, \
             Pool(processes=NUM_WORKERS, initializer=initialize_worker) as pool:
            
            chunksize = max(1, len(tasks) // (NUM_WORKERS * 4))
            with tqdm(total=len(tasks), desc="[REDACTED_BY_SCRIPT]") as pbar:
                for result in pool.imap_unordered(process_single_point, tasks, chunksize=chunksize):
                    if not result['errors']:
                        res_df = pd.DataFrame([result['features']])
                        res_df['hex_id'] = result['hex_id']
                        
                        # Ensure hex_id is the first column
                        cols = ['hex_id'] + [col for col in res_df.columns if col != 'hex_id']
                        res_df = res_df[cols]

                        res_df.to_csv(checkpoint_file, header=write_header, index=False)
                        write_header = False # Only write header once per run
                        checkpoint_file.flush() # Ensure it's written to disk
                    else:
                        log_failure(result)
                    pbar.update()
    except Exception as e:
        logging.critical(f"[REDACTED_BY_SCRIPT]")
        logging.critical(traceback.format_exc())
        return # Terminate execution

    logging.info("[REDACTED_BY_SCRIPT]")
    
    if os.path.exists(CHECKPOINT_CSV):
        os.rename(CHECKPOINT_CSV, FINAL_OUTPUT_CSV)
        logging.info(f"Final artifact '{os.path.basename(FINAL_OUTPUT_CSV)}' saved.")
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")

    logging.info("[REDACTED_BY_SCRIPT]")


def log_failure(result):
    """[REDACTED_BY_SCRIPT]"""
    error_log_path = os.path.join(RUN_DIR, "[REDACTED_BY_SCRIPT]")
    with open(error_log_path, "a") as f:
        f.write(f"Hex ID: {result['hex_id']}\n")
        for error in result['errors']:
            # Indent multi-line tracebacks for readability
            indented_error = "\n\t  ".join(error.splitlines())
            f.write(f"[REDACTED_BY_SCRIPT]")
        f.write("-" * 80 + "\n")

if __name__ == "__main__":
    main()