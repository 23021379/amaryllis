import os
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import re
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys
from typing import List

# --- Project Setup ---
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
NGED_SUBSTATION_INPUT = r"[REDACTED_BY_SCRIPT]"

# Architectural Parameters
SOURCE_CRS = "EPSG:4326"  # WGS84 for raw lon/lat
TARGET_CRS = "EPSG:27700" # British National Grid
POWER_FACTOR = 1.0 # Assumed for MW to kVA conversion
KNN_NEIGHBORS = 5
RADIUS_METERS = 10000

def parse_max_voltage(voltage_string: str) -> float:
    """[REDACTED_BY_SCRIPT]"""
    if not isinstance(voltage_string, str):
        return np.nan
    numeric_values = re.findall(r'(\d+\.?\d*)', voltage_string)
    if not numeric_values:
        return np.nan
    return max([float(v) for v in numeric_values])

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    new_cols = []
    for col in df.columns:
        new_col = col.lower().strip().replace(' ', '_')
        new_col = ''.join(c if c.isalnum() else '_' for c in new_col)
        while '__' in new_col:
            new_col = new_col.replace('__', '_')
        new_col = new_col.strip('_')
        new_cols.append(f"ng_sub_{new_col}") # Prefix with ng_sub_
    df.columns = new_cols
    return df

def ingest_and_prepare_substations(filepath: str) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(filepath)
    df = clean_col_names(df)

    initial_count = len(df)
    df.dropna(subset=['ng_sub_longitude', 'ng_sub_latitude'], inplace=True)
    if len(df) < initial_count:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
    
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.ng_sub_longitude, df.ng_sub_latitude),
        crs=SOURCE_CRS
    ).to_crs(TARGET_CRS)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    return gdf

def synthesize_substation_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    numeric_cols = [
        '[REDACTED_BY_SCRIPT]', 'ng_sub_demandmaximum', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
    ]
    for col in numeric_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

    if 'ng_sub_voltages' in gdf.columns:
        gdf['[REDACTED_BY_SCRIPT]'] = gdf['ng_sub_voltages'].apply(parse_max_voltage)

    gdf['ng_sub_total_kva'] = gdf['[REDACTED_BY_SCRIPT]'] * 1000 * POWER_FACTOR

    with np.errstate(divide='ignore', invalid='ignore'):
        gdf['[REDACTED_BY_SCRIPT]'] = np.where(
            gdf['[REDACTED_BY_SCRIPT]'] > 0,
            gdf['ng_sub_demandmaximum'] / gdf['[REDACTED_BY_SCRIPT]'], np.nan
        )
        gdf['[REDACTED_BY_SCRIPT]'] = np.where(
            gdf['[REDACTED_BY_SCRIPT]'] > 0,
            gdf['[REDACTED_BY_SCRIPT]'] / gdf['[REDACTED_BY_SCRIPT]'], np.nan
        )

    gdf['ng_sub_headroom_mw'] = gdf['[REDACTED_BY_SCRIPT]']

    if '[REDACTED_BY_SCRIPT]' in gdf.columns:
        gdf['[REDACTED_BY_SCRIPT]'] = gdf['[REDACTED_BY_SCRIPT]'].str.lower()
        rag_dummies = pd.get_dummies(gdf['[REDACTED_BY_SCRIPT]'], prefix='ng_sub_demandrag', dtype=int)
        for color in ['green', 'red', 'yellow', 'amber']:
             col_name = f'[REDACTED_BY_SCRIPT]'
             if col_name not in rag_dummies.columns:
                 rag_dummies[col_name] = 0
        if 'ng_sub_demandrag_amber' in rag_dummies.columns:
            rag_dummies['[REDACTED_BY_SCRIPT]'] = rag_dummies['[REDACTED_BY_SCRIPT]'] | rag_dummies['ng_sub_demandrag_amber']
            rag_dummies.drop(columns=['ng_sub_demandrag_amber'], inplace=True)
        gdf = pd.concat([gdf, rag_dummies[['ng_sub_demandrag_green', 'ng_sub_demandrag_red', '[REDACTED_BY_SCRIPT]']]], axis=1)
    
    return gdf

def calculate_aggregated_proximity_features(gdf_solar: gpd.GeoDataFrame, gdf_substations: gpd.GeoDataFrame) -> pd.DataFrame:
    """Calculates 'New Density'[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    substation_coords = np.array(list(gdf_substations.geometry.apply(lambda p: (p.x, p.y))))
    spatial_index = cKDTree(substation_coords)
    solar_coords = np.array(list(gdf_solar.geometry.apply(lambda p: (p.x, p.y))))

    logging.info(f"[REDACTED_BY_SCRIPT]")
    distances_knn, indices_knn = spatial_index.query(solar_coords, k=KNN_NEIGHBORS)
    
    neighbor_headroom = gdf_substations['ng_sub_headroom_mw'].to_numpy()[indices_knn]
    neighbor_kva = gdf_substations['ng_sub_total_kva'].to_numpy()[indices_knn]
    
    agg_features = pd.DataFrame({
        f'[REDACTED_BY_SCRIPT]': np.mean(distances_knn, axis=1) / 1000,
        f'[REDACTED_BY_SCRIPT]': np.nanmean(neighbor_headroom, axis=1),
        f'[REDACTED_BY_SCRIPT]': np.nanstd(neighbor_headroom, axis=1),
        f'[REDACTED_BY_SCRIPT]': np.nanmean(neighbor_kva, axis=1)
    }, index=gdf_solar.index)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    indices_radius = spatial_index.query_ball_point(solar_coords, r=RADIUS_METERS)
    
    total_kva_in_radius = [
        gdf_substations['ng_sub_total_kva'].iloc[indices].sum()
        for indices in tqdm(indices_radius, desc="[REDACTED_BY_SCRIPT]")
    ]
    agg_features[f'[REDACTED_BY_SCRIPT]'] = total_kva_in_radius
    
    return agg_features

def execute(master_gdf):
    """
    Executor entry point for calculating National Grid substation features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Critical CRS Check (Pattern 1 Defense)
    # We must enforce EPSG:27700 (Meters) before any distance calculations
    if master_gdf.crs is None:
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf.set_crs(TARGET_CRS, inplace=True)
    elif master_gdf.crs.to_string() != TARGET_CRS:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        master_gdf = master_gdf.to_crs(TARGET_CRS)

    try:
        gdf_substations_raw = ingest_and_prepare_substations(NGED_SUBSTATION_INPUT)
        gdf_substations_l1 = synthesize_substation_features(gdf_substations_raw)
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    # Phase 3: Direct Feature Integration via Proximity Join
    logging.info("[REDACTED_BY_SCRIPT]")
    # V6.4: Reset index before join to preserve it, as sjoin_nearest can create non-unique indices
    master_gdf_reset = master_gdf.reset_index()
    gdf_joined = gpd.sjoin_nearest(master_gdf_reset, gdf_substations_l1, distance_col="dist_to_nearest_substation_m", how="left")

    # The join can create duplicates if a point is equidistant to multiple substations.
    # We keep the first match for each unique grid cell ('hex_id').
    gdf_joined = gdf_joined.drop_duplicates(subset='hex_id', keep='first')
    
    # Rename direct features to match project schema
    rename_map = {
        "ng_sub_headroom_mw": "[REDACTED_BY_SCRIPT]",
        "ng_sub_total_kva": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "ng_sub_demandrag_green": "ng_sub_demandrag_green",
        "ng_sub_demandrag_red": "ng_sub_demandrag_red",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
    }
    rename_map_filtered = {k: v for k, v in rename_map.items() if k in gdf_joined.columns}
    gdf_joined.rename(columns=rename_map_filtered, inplace=True)
    gdf_joined['[REDACTED_BY_SCRIPT]'] = gdf_joined['dist_to_nearest_substation_m'] / 1000.0

    # Phase 4: The "New Density" Engine
    # We must calculate this on the original master_gdf to ensure correct alignment
    aggregated_features = calculate_aggregated_proximity_features(master_gdf, gdf_substations_l1)

    # Final Integration
    logging.info("[REDACTED_BY_SCRIPT]")

    # 1. Merge Direct Features (Join on ID)
    # Ensure we only merge attributes, not geometry, to preserve master_gdf's spatial integrity
    direct_cols_to_merge = ['hex_id', '[REDACTED_BY_SCRIPT]'] + list(rename_map_filtered.values())
    
    # Using strict left merge on hex_id
    master_gdf = master_gdf.merge(
        gdf_joined[direct_cols_to_merge],
        on='hex_id',
        how='left'
    )

    # 2. Merge Aggregated Features (Join on Index)
    # calculated_aggregated_proximity_features preserves the input index, so we can safely join
    master_gdf = master_gdf.join(aggregated_features, how='left')

    # 3. Final Artifact Validation
    if not isinstance(master_gdf, gpd.GeoDataFrame):
        logging.warning("[REDACTED_BY_SCRIPT]")
        master_gdf = gpd.GeoDataFrame(master_gdf, geometry='geometry', crs=TARGET_CRS)


    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf
