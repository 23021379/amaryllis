import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import re
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys
from typing import List

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
NGED_SUBSTATION_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_SITES_INPUT = r"[REDACTED_BY_SCRIPT]"

# Output Artifact
L27_OUTPUT_PATH = r'[REDACTED_BY_SCRIPT]'

# Architectural Parameters
SOURCE_CRS = "EPSG:4326"  # WGS84 for raw lon/lat
TARGET_CRS = "EPSG:27700" # British National Grid
POWER_FACTOR = 1.0 # Assumed for MW to kVA conversion
KNN_NEIGHBORS = 5
RADIUS_METERS = 10000

def parse_max_voltage(voltage_string: str) -> float:
    """
    Mandate 7.1: Parses a string to find all numeric voltage values and returns the maximum.
    e.g., "132/33kV" -> 132.0
    """
    if not isinstance(voltage_string, str):
        return np.nan
    
    # Find all numbers (including decimals) in the string
    numeric_values = re.findall(r'(\d+\.?\d*)', voltage_string)
    
    if not numeric_values:
        return np.nan
        
    return max([float(v) for v in numeric_values])


def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    new_cols = []
    for col in df.columns:
        new_col = col.lower().strip()
        new_col = ''.join(c if c.isalnum() else '_' for c in new_col)
        while '__' in new_col:
            new_col = new_col.replace('__', '_')
        new_col = new_col.strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def ingest_and_prepare_substations(filepath: str) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(filepath)
    
    # CRITICAL MANDATE: Schema Normalization
    df = clean_col_names(df)

    # DECONTAMINATION GATE: Purge records with invalid coordinates before georeferencing.
    initial_count = len(df)
    df.dropna(subset=['longitude', 'latitude'], inplace=True)
    final_count = len(df)
    if initial_count > final_count:
        dropped_count = initial_count - final_count
        logging.warning(f"[REDACTED_BY_SCRIPT]")
    
    # CRITICAL MANDATE: Georeferencing & CRS Unification (Anti-Pattern 1 Gate)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs=SOURCE_CRS
    )
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gdf.to_crs(TARGET_CRS)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    return gdf

def synthesize_substation_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Paranoid Type Casting
    numeric_cols = [
        'demandtotalcapacity', 'demandmaximum', 'generationconnectedheadroommw',
        'demandconnectedheadroommw', 'demand50percentile'
    ]
    for col in numeric_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

    # Mandate 7.1: Robust Voltage Parsing
    if 'voltages' in gdf.columns:
        gdf['max_voltage_kv'] = gdf['voltages'].apply(parse_max_voltage)

    # Standardize Capacity Units (MW -> kVA)
    gdf['substation_total_kva'] = gdf['demandtotalcapacity'] * 1000 * POWER_FACTOR

    # Mandate 7.2: Decommission generic loading and synthesize stratified seasonal loading
    with np.errstate(divide='ignore', invalid='ignore'):
        # Winter loading based on absolute maximum demand
        gdf['[REDACTED_BY_SCRIPT]'] = np.where(
            gdf['demandtotalcapacity'] > 0,
            gdf['demandmaximum'] / gdf['demandtotalcapacity'],
            np.nan
        )
        # Summer loading based on 50th percentile as proxy for typical/off-peak load
        gdf['[REDACTED_BY_SCRIPT]'] = np.where(
            gdf['demandtotalcapacity'] > 0,
            gdf['demand50percentile'] / gdf['demandtotalcapacity'],
            np.nan
        )

    # Standardize Headroom Metric
    gdf['[REDACTED_BY_SCRIPT]'] = gdf['generationconnectedheadroommw']

    # One-Hot Encode RAG Status
    if 'demandconnectedrag' in gdf.columns:
        gdf['demandconnectedrag'] = gdf['demandconnectedrag'].str.lower()
        rag_dummies = pd.get_dummies(gdf['demandconnectedrag'], prefix='demandrag', dtype=int)
        # Ensure all potential RAG columns exist, filling with 0 if a color is not present
        for color in ['green', 'red', 'yellow', 'amber']: # Amber is a common synonym
             col_name = f'demandrag_{color}'
             if col_name not in rag_dummies.columns:
                 rag_dummies[col_name] = 0
        # Standardize to green, red, yellow
        if 'demandrag_amber' in rag_dummies.columns:
            rag_dummies['demandrag_yellow'] = rag_dummies['demandrag_yellow'] | rag_dummies['demandrag_amber']
            rag_dummies.drop(columns=['demandrag_amber'], inplace=True)

        gdf = pd.concat([gdf, rag_dummies[['demandrag_green', 'demandrag_red', 'demandrag_yellow']]], axis=1)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    return gdf

def calculate_aggregated_proximity_features(gdf_solar: gpd.GeoDataFrame, gdf_substations: gpd.GeoDataFrame) -> pd.DataFrame:
    """Phase 4: The 'New Density'[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # Build a high-performance spatial index on substation coordinates
    substation_coords = np.array(list(gdf_substations.geometry.apply(lambda p: (p.x, p.y))))
    spatial_index = cKDTree(substation_coords)
    
    solar_coords = np.array(list(gdf_solar.geometry.apply(lambda p: (p.x, p.y))))

    # --- K-Nearest Neighbors Aggregation (Vectorized) ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    distances_knn, indices_knn = spatial_index.query(solar_coords, k=KNN_NEIGHBORS)
    
    # Use NumPy indexing for high-performance feature extraction
    neighbor_headroom = gdf_substations['[REDACTED_BY_SCRIPT]'].to_numpy()[indices_knn]
    neighbor_kva = gdf_substations['substation_total_kva'].to_numpy()[indices_knn]
    
    # Calculate aggregate statistics
    agg_features = pd.DataFrame({
        f'[REDACTED_BY_SCRIPT]': np.mean(distances_knn, axis=1) / 1000,
        f'[REDACTED_BY_SCRIPT]': np.nanmean(neighbor_headroom, axis=1),
        f'[REDACTED_BY_SCRIPT]': np.nanstd(neighbor_headroom, axis=1),
        f'[REDACTED_BY_SCRIPT]': np.nanmean(neighbor_kva, axis=1)
    }, index=gdf_solar.index)

    # --- Radius-Based Aggregation ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    indices_radius = spatial_index.query_ball_point(solar_coords, r=RADIUS_METERS)
    
    # This part is less suited to vectorization, so we iterate efficiently
    total_kva_in_radius = [
        gdf_substations['substation_total_kva'].iloc[indices].sum()
        for indices in tqdm(indices_radius, desc="[REDACTED_BY_SCRIPT]")
    ]
    agg_features[f'[REDACTED_BY_SCRIPT]'] = total_kva_in_radius
    
    logging.info("[REDACTED_BY_SCRIPT]")
    return agg_features

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    try:
        # Phases 1 & 2: Prepare the master substation data artifact
        gdf_substations_raw = ingest_and_prepare_substations(NGED_SUBSTATION_INPUT)
        gdf_substations_l1 = synthesize_substation_features(gdf_substations_raw)

        # Load solar application data
        logging.info(f"[REDACTED_BY_SCRIPT]")
        df_solar = pd.read_csv(SOLAR_SITES_INPUT)
        gdf_solar = gpd.GeoDataFrame(
            df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS
        )
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)
    except KeyError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # Phase 3: Direct Feature Integration via Proximity Join
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_joined = gpd.sjoin_nearest(gdf_solar, gdf_substations_l1, distance_col="dist_to_nearest_substation_m")

    # DECONTAMINATION GATE: Enforce one-to-one join result.
    # The sjoin_nearest operation can, in rare edge cases (e.g., multiple equidistant points),
    # return multiple matches for a single left geometry. This guard rail de-duplicates the
    # result based on the solar site index, ensuring each site appears exactly once.
    if gdf_joined.index.has_duplicates:
        duplicate_count = gdf_joined.index.duplicated().sum()
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf_joined = gdf_joined[~gdf_joined.index.duplicated(keep='first')]
    
    # Rename direct features to match project schema
    rename_map = {
        "[REDACTED_BY_SCRIPT]": "nearest_[REDACTED_BY_SCRIPT]",
        "substation_total_kva": "[REDACTED_BY_SCRIPT]",
        "max_voltage_kv": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "demandrag_green": "demandrag_green",
        "demandrag_red": "demandrag_red",
        "demandrag_yellow": "demandrag_yellow",
    }
    # Filter map to only include columns that actually exist after the join
    rename_map_filtered = {k: v for k, v in rename_map.items() if k in gdf_joined.columns}
    gdf_joined.rename(columns=rename_map_filtered, inplace=True)
    gdf_joined['[REDACTED_BY_SCRIPT]'] = gdf_joined['dist_to_nearest_substation_m'] / 1000.0

    # Phase 4: The "New Density" Engine
    aggregated_features = calculate_aggregated_proximity_features(gdf_solar, gdf_substations_l1)

    # Final Integration
    logging.info("[REDACTED_BY_SCRIPT]")
    # Ensure indices align for a clean join
    gdf_final = gdf_joined.join(aggregated_features)

    # --- Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    cols_to_drop = [
        'index_right', 'geometry', 'dist_to_nearest_substation_m'
    ] + [c for c in gdf_final.columns if c not in rename_map_filtered.values() and c in gdf_substations_l1.columns]
    
    df_final = pd.DataFrame(gdf_final.drop(columns=cols_to_drop, errors='ignore'))
    
    return_ti_orig = {
        "demandrag_green": "[REDACTED_BY_SCRIPT]",
        "demandrag_red": "[REDACTED_BY_SCRIPT]",
        "demandrag_yellow": "[REDACTED_BY_SCRIPT]",
    }
    df_final.rename(columns=return_ti_orig, inplace=True)
    # Verify no data loss
    if len(df_final) != len(gdf_solar):
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    df_final.to_csv(L27_OUTPUT_PATH, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()