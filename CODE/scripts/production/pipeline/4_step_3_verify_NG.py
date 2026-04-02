import geopandas as gpd
from pathlib import Path
import logging
import time

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
ARTIFACT_DIR = Path("[REDACTED_BY_SCRIPT]")
TARGET_CRS = "EPSG:27700"
PERFORMANCE_THRESHOLD_S = 10.0

# Define a test case: a known solar farm location in NG territory (e.g., East Anglia)
TEST_POINT_BNG = {"name": "test_solar_farm", "x": 605000, "y": 280000}
TEST_BUFFER_M = 5000

# --- SCRIPT ---
def verify_artifact(artifact_path):
    """[REDACTED_BY_SCRIPT]"""
    if not artifact_path.exists():
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return False, f"[REDACTED_BY_SCRIPT]"

    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    try:
        start_time = time.time()
        gdf = gpd.read_parquet(artifact_path)
        load_time = time.time() - start_time
        logging.info(f"[REDACTED_BY_SCRIPT]")

        if gdf.crs != TARGET_CRS:
            msg = f"[REDACTED_BY_SCRIPT]"
            logging.error(f"  -> {msg}")
            return False, msg
        logging.info(f"[REDACTED_BY_SCRIPT]")

        if not gdf.empty:
            test_point_gdf = gpd.GeoDataFrame(
                [TEST_POINT_BNG],
                geometry=gpd.points_from_xy([TEST_POINT_BNG['x']], [TEST_POINT_BNG['y']]),
                crs=TARGET_CRS
            )
            test_buffer = test_point_gdf.buffer(TEST_BUFFER_M)

            possible_matches_idx = list(gdf.sindex.intersection(test_buffer.total_bounds))
            possible_matches = gdf.iloc[possible_matches_idx]
            precise_matches = possible_matches[possible_matches.intersects(test_buffer.unary_union)]
            
            query_time = time.time() - start_time - load_time
            logging.info(f"[REDACTED_BY_SCRIPT]")
            
            total_time = time.time() - start_time
            if total_time > PERFORMANCE_THRESHOLD_S:
                msg = f"[REDACTED_BY_SCRIPT]"
                logging.error(f"  -> {msg}")
                return False, msg
            logging.info(f"[REDACTED_BY_SCRIPT]")

            if precise_matches.empty:
                logging.warning(f"[REDACTED_BY_SCRIPT]")
            elif len(precise_matches) == len(gdf):
                msg = f"[REDACTED_BY_SCRIPT]"
                logging.error(f"  -> {msg}")
                return False, msg
            else:
                logging.info(f"[REDACTED_BY_SCRIPT]")
        else:
            logging.warning("[REDACTED_BY_SCRIPT]")

    except Exception as e:
        msg = f"[REDACTED_BY_SCRIPT]"
        logging.error(f"  -> {msg}")
        return False, msg
        
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return True, "Success"

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    artifacts_to_test = list(ARTIFACT_DIR.glob("*.geoparquet"))
    if not artifacts_to_test:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return

    logging.info(f"[REDACTED_BY_SCRIPT]")
    failures = []

    for artifact_path in artifacts_to_test:
        is_success, reason = verify_artifact(artifact_path)
        if not is_success:
            failures.append(f"[REDACTED_BY_SCRIPT]")
    
    if not failures:
        logging.info("[REDACTED_BY_SCRIPT]")
    else:
        logging.error("[REDACTED_BY_SCRIPT]")
        logging.error("--- SUMMARY OF FAILURES ---")
        for f in failures:
            logging.error(f"- {f}")

            
if __name__ == "__main__":
    main()