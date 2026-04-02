import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
import re
from thefuzz import process as fuzzy_process
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
FAULT_LEVEL_INPUT = r"[REDACTED_BY_SCRIPT]"
PRIMARY_SUBS_AUTH_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_SITES_INPUT = r"[REDACTED_BY_SCRIPT]"

# Output Artifacts
L1_FAULT_ENRICHED_OUTPUT = r"[REDACTED_BY_SCRIPT]"
L30_FINAL_OUTPUT = r"[REDACTED_BY_SCRIPT]"

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
FUZZY_MATCH_CONFIDENCE_THRESHOLD = 90
KNN_NEIGHBORS = 5

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    new_cols = []
    for col in df.columns:
        new_col = col.lower().strip().replace(' ', '_')
        new_col = ''.join(c if c.isalnum() else '_' for c in new_col)
        while '__' in new_col: new_col = new_col.replace('__', '_')
        new_col = new_col.strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def aggregate_fault_levels(fault_path: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    df = pd.read_csv(fault_path)
    df = clean_col_names(df)

    for col in ['3ph_rms_break', '1ph_rms_break']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # CRITICAL MANDATE: Aggregate using max() to model worst-case engineering constraint
    agg_rules = {
        '3ph_rms_break': 'max',
        '1ph_rms_break': 'max'
    }
    df_agg = df.groupby('substation_or_busbar_name').agg(agg_rules).reset_index()
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return df_agg

def sanitize_join_key(name: str) -> str:
    """[REDACTED_BY_SCRIPT]"""
    if not isinstance(name, str): return ''
    name = name.lower()
    # Remove common terms and punctuation
    stopwords = ['kv', 'substation', 'primary', 'bsp', 'grid']
    for word in stopwords:
        name = name.replace(word, '')
    name = re.sub(r'[^a-z0-9]', '', name)
    return name.strip()

def fuse_with_fault_data(gdf_primary: gpd.GeoDataFrame, df_fault: pd.DataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Create sanitized join keys in both dataframes
    gdf_primary['sanitized_join_key'] = gdf_primary['name'].apply(sanitize_join_key)
    df_fault['sanitized_join_key'] = df_fault['substation_or_busbar_name'].apply(sanitize_join_key)
    
    # Stage A: Direct Join
    gdf_enriched = pd.merge(
        gdf_primary, df_fault, on='sanitized_join_key', how='left'
    )
    
    # Isolate orphans for Stage B
    orphans = gdf_enriched[gdf_enriched['3ph_rms_break'].isna()].copy()
    successes = gdf_enriched[gdf_enriched['3ph_rms_break'].notna()]
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    if not orphans.empty:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        choices = df_fault['sanitized_join_key'].unique()
        
        matches = orphans['sanitized_join_key'].apply(
            lambda x: fuzzy_process.extractOne(x, choices, score_cutoff=FUZZY_MATCH_CONFIDENCE_THRESHOLD)
        )
        
        # Filter out low-confidence matches
        valid_matches = matches.dropna()
        if not valid_matches.empty:
            match_df = pd.DataFrame(valid_matches.tolist(), index=valid_matches.index, columns=['match', 'score'])
            match_df = pd.merge(match_df, df_fault, left_on='match', right_on='sanitized_join_key', how='left')
            
            # Update the orphan records with the matched fault data
            orphans.update(match_df)
            logging.info(f"[REDACTED_BY_SCRIPT]")
        
        # Recombine successes and high-confidence fuzzy matches
        gdf_enriched = pd.concat([successes, orphans], ignore_index=True)

    gdf_enriched.to_parquet(L1_FAULT_ENRICHED_OUTPUT)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_enriched

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        df_fault_agg = aggregate_fault_levels(FAULT_LEVEL_INPUT)
        gdf_primary_subs = gpd.read_parquet(PRIMARY_SUBS_AUTH_INPUT)
        df_solar = pd.read_csv(SOLAR_SITES_INPUT)
        gdf_solar = gpd.GeoDataFrame(
            df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS
        )
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]"); sys.exit(1)

    gdf_enriched_subs = fuse_with_fault_data(gdf_primary_subs, df_fault_agg)

    # --- Phase 3: Final Integration & Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # Direct Feature Integration
    gdf_solar_joined = gpd.sjoin_nearest(gdf_solar, gdf_enriched_subs, how="left")
    
    # MANDATED DE-DUPLICATION GUARD RAIL: Immediately cleanse join artifacts.
    if gdf_solar_joined.index.has_duplicates:
        duplicate_count = gdf_solar_joined.index.duplicated().sum()
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf_solar_joined = gdf_solar_joined[~gdf_solar_joined.index.duplicated(keep='first')]

    gdf_solar_joined.rename(columns={
        '3ph_rms_break': '[REDACTED_BY_SCRIPT]',
        '1ph_rms_break': '[REDACTED_BY_SCRIPT]'
    }, inplace=True)

    # Aggregate Feature Synthesis (k-NN)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    # Ensure we only use substations with valid fault data for aggregation
    gdf_agg_source = gdf_enriched_subs.dropna(subset=['3ph_rms_break', '1ph_rms_break'])
    
    agg_coords = np.array(list(gdf_agg_source.geometry.apply(lambda p: (p.x, p.y))))
    solar_coords = np.array(list(gdf_solar.geometry.apply(lambda p: (p.x, p.y))))
    spatial_index = cKDTree(agg_coords)
    
    _, indices_knn = spatial_index.query(solar_coords, k=KNN_NEIGHBORS)
    
    fault_3ph_values = gdf_agg_source['3ph_rms_break'].to_numpy()[indices_knn]
    fault_1ph_values = gdf_agg_source['1ph_rms_break'].to_numpy()[indices_knn]
    
    agg_features = pd.DataFrame({
        '[REDACTED_BY_SCRIPT]': np.nanmean(fault_3ph_values, axis=1),
        'avg_max_ef_fault_5nn_ka': np.nanmean(fault_1ph_values, axis=1)
    }, index=gdf_solar.index)
    
    # Final join of aggregated features
    df_final = gdf_solar_joined.join(agg_features)

    # --- Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    cols_to_drop = [col for col in df_final.columns if col in gdf_enriched_subs.columns and col != 'geometry']
    cols_to_drop += ['geometry', 'index_right']
    df_final = df_final.drop(columns=cols_to_drop, errors='ignore')
    
    df_final.to_csv(L30_FINAL_OUTPUT, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()