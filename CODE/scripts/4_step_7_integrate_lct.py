import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
from tqdm import tqdm
from scipy.spatial import cKDTree

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
pd.set_option('[REDACTED_BY_SCRIPT]', True)

# Input Artifacts
LCT_INPUT = r"[REDACTED_BY_SCRIPT]"
PRIMARY_SUBS_AUTH_INPUT = r"[REDACTED_BY_SCRIPT]"
SOLAR_SITES_INPUT = r"[REDACTED_BY_SCRIPT]"

# Output Artifacts
L2_LCT_SPATIOTEMPORAL_OUTPUT = r"[REDACTED_BY_SCRIPT]"
L30_FINAL_OUTPUT = r"[REDACTED_BY_SCRIPT]"

# Architectural Parameters
TARGET_CRS = "EPSG:27700"
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

def create_l2_lct_artifact(lct_path: str, primary_subs_path: str) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    
    df_lct = pd.read_csv(lct_path, dtype=str)
    df_lct = clean_col_names(df_lct)
    
    # Paranoid Type Casting & Validation
    lct_cols = ['ev', 'es', 'hp', 'pv']
    for col in lct_cols:
        df_lct[col] = pd.to_numeric(df_lct[col], errors='coerce')
    df_lct['connected_month'] = pd.to_datetime(df_lct['connected_month'], errors='coerce')
    
    # Quarantine poison records
    initial_count = len(df_lct)
    df_lct.dropna(subset=lct_cols + ['connected_month', '[REDACTED_BY_SCRIPT]'], inplace=True)
    if len(df_lct) < initial_count:
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    gdf_primary_subs = gpd.read_parquet(primary_subs_path)
    
    # DECONTAMINATION GATE: Enforce consistent key data types before merge.
    # Coerce both keys to a common, nullable integer format to prevent ValueError.
    df_lct['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(df_lct['[REDACTED_BY_SCRIPT]'], errors='coerce')
    df_lct.dropna(subset=['[REDACTED_BY_SCRIPT]'], inplace=True)
    df_lct['[REDACTED_BY_SCRIPT]'] = df_lct['[REDACTED_BY_SCRIPT]'].astype('Int64')

    gdf_primary_subs['substationnumber'] = pd.to_numeric(gdf_primary_subs['substationnumber'], errors='coerce')
    gdf_primary_subs.dropna(subset=['substationnumber'], inplace=True)
    gdf_primary_subs['substationnumber'] = gdf_primary_subs['substationnumber'].astype('Int64')
    
    # Geospatial Bridge Protocol
    gdf_merged = pd.merge(
        df_lct,
        gdf_primary_subs,
        left_on='[REDACTED_BY_SCRIPT]',
        right_on='substationnumber',
        how='left'
    )
    
    # Validate the bridge and quarantine orphans
    orphans = gdf_merged['geometry'].isna().sum()
    if orphans > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf_merged.dropna(subset=['geometry'], inplace=True)

    gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry='geometry', crs=TARGET_CRS)
    gdf_merged.to_parquet(L2_LCT_SPATIOTEMPORAL_OUTPUT)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return gdf_merged

def synthesize_point_in_time_lct_features(solar_farm_row: pd.Series, lct_gdf: gpd.GeoDataFrame, primary_subs_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    # MANDATE: Prevent state pollution across loop iterations.
    # The function must operate on a copy to avoid modifying the original gdf_primary_subs
    # dataframe in the main loop, which would cause column name conflicts (_x, _y) on subsequent merges.
    primary_subs_gdf = primary_subs_gdf.copy()
    
    submission_date = solar_farm_row['submission_date']
    
    # CRITICAL: The Temporal Guard
    lct_snapshot = lct_gdf[lct_gdf['connected_month'] <= submission_date]
    
    if lct_snapshot.empty:
        # Return the base substation data with zeroed-out features
        primary_subs_gdf['[REDACTED_BY_SCRIPT]'] = 0
        primary_subs_gdf['[REDACTED_BY_SCRIPT]'] = 0
        primary_subs_gdf['[REDACTED_BY_SCRIPT]'] = 0
        primary_subs_gdf['[REDACTED_BY_SCRIPT]'] = 0
        primary_subs_gdf['[REDACTED_BY_SCRIPT]'] = -1 # Sentinel for no data
        return primary_subs_gdf

    # Aggregate point-in-time intelligence
    agg_df = lct_snapshot.groupby('[REDACTED_BY_SCRIPT]')[['ev', 'es', 'hp', 'pv']].sum()
    
    # Synthesize strategic features
    agg_df['[REDACTED_BY_SCRIPT]'] = agg_df[['ev', 'es', 'hp', 'pv']].sum(axis=1)
    generation_lcts = agg_df['pv'] + agg_df['es']
    demand_lcts = agg_df['ev'] + agg_df['hp']
    
    # Mandated stability pattern (+1 in denominator)
    agg_df['[REDACTED_BY_SCRIPT]'] = generation_lcts / (demand_lcts + 1)
    agg_df['[REDACTED_BY_SCRIPT]'] = agg_df['pv'] / (agg_df['[REDACTED_BY_SCRIPT]'] + 1)
    agg_df['[REDACTED_BY_SCRIPT]'] = demand_lcts / (agg_df['[REDACTED_BY_SCRIPT]'] + 1)

    # Synthesize data vintage feature
    latest_lct_date = lct_snapshot['connected_month'].max()
    vintage_days = (submission_date - latest_lct_date).days
    agg_df['[REDACTED_BY_SCRIPT]'] = round(vintage_days / 30.44)

    # Re-spatialize the synthesized features
    enriched_subs = primary_subs_gdf.merge(
        agg_df, left_on='substationnumber', right_on='[REDACTED_BY_SCRIPT]', how='left'
    ).fillna(0) # Fill substations with no LCT data with 0
    
    return enriched_subs

def main():
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        gdf_lct_spatiotemporal = create_l2_lct_artifact(LCT_INPUT, PRIMARY_SUBS_AUTH_INPUT)
        gdf_primary_subs = gpd.read_parquet(PRIMARY_SUBS_AUTH_INPUT)
        df_solar = pd.read_csv(SOLAR_SITES_INPUT)
        
        # Reconstruct the submission_date from its integer components.
        date_cols = {
            'year': df_solar['submission_year'],
            'month': df_solar['submission_month'],
            'day': df_solar['submission_day']
        }
        df_solar['submission_date'] = pd.to_datetime(date_cols, errors='coerce')

        df_solar.dropna(subset=['submission_date'], inplace=True)
        gdf_solar = gpd.GeoDataFrame(
            df_solar, geometry=gpd.points_from_xy(df_solar.easting, df_solar.northing), crs=TARGET_CRS
        )
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]"); sys.exit(1)

    # --- Pre-computation for k-NN Search ---
    # Build a spatial index on the authoritative substation locations for high performance.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    primary_sub_coords = np.array(list(gdf_primary_subs.geometry.apply(lambda p: (p.x, p.y))))
    spatial_index = cKDTree(primary_sub_coords)
    
    # Query the k-nearest neighbors for ALL solar sites at once.
    solar_coords = np.array(list(gdf_solar.geometry.apply(lambda p: (p.x, p.y))))
    logging.info(f"[REDACTED_BY_SCRIPT]")
    _, nearest_indices_for_all_sites = spatial_index.query(solar_coords, k=KNN_NEIGHBORS)

    gdf_solar['application_id'] = gdf_solar.index.astype(str)
    # gdf_solar.set_index('application_id', inplace=True, drop=False) 
    gdf_solar = gdf_solar.drop_duplicates(subset=['application_id'], keep='first').reset_index(drop=True)

    all_solar_features = []
    
    # Use gdf_solar.index to correctly align with pre-calculated indices
    for i, (index, solar_farm_row) in enumerate(tqdm(gdf_solar.iterrows(), total=len(gdf_solar), desc="[REDACTED_BY_SCRIPT]")):
        # Phase 2: Generate a unique, point-in-time feature set for this solar farm
        gdf_enriched_subs = synthesize_point_in_time_lct_features(solar_farm_row, gdf_lct_spatiotemporal, gdf_primary_subs)
        
        # Phase 3: Final Integration using pre-calculated indices
        # Get the indices of the 5 nearest substations for this specific solar farm
        nearest_sub_indices = nearest_indices_for_all_sites[i]
        
        # Select the cohort of nearest substations from the temporally-enriched data
        gdf_nearest_subs = gdf_enriched_subs.iloc[nearest_sub_indices]

        # Extract features for the single nearest substation (n=1)
        nearest_sub_features = gdf_nearest_subs.iloc[0]
        
        # Calculate aggregate features for the k-nearest cohort
        feature_cols = [
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]'
        ]
        agg_calcs = gdf_nearest_subs[feature_cols].agg(['mean', 'std'])

        # ENFORCE AMARYLLIS NAMING PROTOCOL
        site_features = {
            # Data Provenance
            '[REDACTED_BY_SCRIPT]': nearest_sub_features['[REDACTED_BY_SCRIPT]'],
            
            # Nearest Neighbor Features (n=1)
            'ng_lct_nearest_total_connections': nearest_sub_features['[REDACTED_BY_SCRIPT]'],
            '[REDACTED_BY_SCRIPT]': nearest_sub_features['[REDACTED_BY_SCRIPT]'],
            '[REDACTED_BY_SCRIPT]': nearest_sub_features['[REDACTED_BY_SCRIPT]'],
            '[REDACTED_BY_SCRIPT]': nearest_sub_features['[REDACTED_BY_SCRIPT]'],

            # Local Cluster Features (k=5) - Averages
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['mean', '[REDACTED_BY_SCRIPT]'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['mean', '[REDACTED_BY_SCRIPT]'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['mean', '[REDACTED_BY_SCRIPT]'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['mean', '[REDACTED_BY_SCRIPT]'],

            # Local Cluster Features (k=5) - Standard Deviations
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['std', '[REDACTED_BY_SCRIPT]'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['std', '[REDACTED_BY_SCRIPT]'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['std', '[REDACTED_BY_SCRIPT]'],
            f'[REDACTED_BY_SCRIPT]': agg_calcs.loc['std', '[REDACTED_BY_SCRIPT]'],
        }
        
        site_features['application_id'] = solar_farm_row['application_id'] # Use a unique ID
        all_solar_features.append(site_features)

    # Final merge of all synthesized features
    df_lct_features = pd.DataFrame(all_solar_features)
    df_final = pd.merge(gdf_solar, df_lct_features, on='application_id', how='left')

    # --- Finalization ---
    logging.info("[REDACTED_BY_SCRIPT]")
    df_final = df_final.drop(columns='geometry', errors='ignore')
    df_final.to_csv(L30_FINAL_OUTPUT, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()