import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import warnings
import os
# Suppress pandas SettingWithCopyWarning, as the operations are deliberate.
# The location of this warning class was moved in newer pandas versions.
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# Define project constants for file paths and CRS to ensure consistency.
L26_GRID_LPA_PATH = r"[REDACTED_BY_SCRIPT]"
L41_ENV_PATH = r"[REDACTED_BY_SCRIPT]"
CS_SOURCE_PATH = r"[REDACTED_BY_SCRIPT]"
CS_L1_SANITIZED_PATH = r"[REDACTED_BY_SCRIPT]"
L44_FINAL_ARTIFACT_PATH = r"[REDACTED_BY_SCRIPT]"

PROJECT_CRS = "EPSG:27700"
RISK_GRADIENT_RADII = {'2km': 2000, '5km': 5000, '10km': 10000, '20km': 20000}


def phase_1_prepare_temporal_base_artifact(l26_path, l41_path, crs):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Load L26 and isolate the true temporal key components.
    # As per user, 'application_id' is not reliable. We will use index-based merging.
    date_cols = ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']
    df_l26 = pd.read_csv(l26_path, usecols=date_cols)
    
    # Data Integrity: Drop rows where any date component is missing.
    df_l26.dropna(subset=date_cols, inplace=True)

    # Construct a valid datetime object from the component columns.
    date_components = {
        'year': df_l26['[REDACTED_BY_SCRIPT]'].astype(int),
        'month': df_l26['[REDACTED_BY_SCRIPT]'].astype(int),
        'day': df_l26['[REDACTED_BY_SCRIPT]'].astype(int)
    }
    df_l26['date_validated'] = pd.to_datetime(date_components, errors='coerce')
    
    # Create the final temporal key dataframe, dropping rows where date construction failed.
    df_temporal_key = df_l26[['date_validated']].dropna()

    # Load L41 environmental data
    df_l41 = pd.read_csv(l41_path)

    # Create the Temporal Bridge by merging on index.
    df_l41_temporal = pd.merge(df_l41, df_temporal_key, left_index=True, right_index=True, how='left')
    
    # Create 'application_id' from the index for downstream compatibility.
    df_l41_temporal['application_id'] = df_l41_temporal.index
    
    ## Validate the merge - critical for temporal integrity
    num_null = df_l41_temporal['date_validated'].isnull().sum()
    if num_null > 0:
        pct_null = (num_null / len(df_l41_temporal)) * 100
        print(f"[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")

    # Convert to GeoDataFrame
    gdf_base_temporal = gpd.GeoDataFrame(
        df_l41_temporal,
        geometry=gpd.points_from_xy(df_l41_temporal.easting_x, df_l41_temporal.northing_x),
        crs=crs
    )
    
    print("[REDACTED_BY_SCRIPT]")
    return gdf_base_temporal


def phase_2_ingest_and_sanitize_cs_data(source_path, output_path, crs):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # High-performance ingestion using pyogrio engine
    cs_gdf = gpd.read_file(source_path, engine='pyogrio')
    
    # IMMEDIATE CRS Unification - non-negotiable first step.
    cs_gdf = cs_gdf.to_crs(crs)

    # Proactive Schema Normalization
    cs_gdf.columns = cs_gdf.columns.str.lower().str.replace(' ', '_')

    # Temporal & Categorical Sanitization
    cs_gdf['startdate'] = pd.to_datetime(cs_gdf['startdate'], format='%Y%m%d', errors='coerce')
    cs_gdf['enddate'] = pd.to_datetime(cs_gdf['enddate'], format='%Y%m%d', errors='coerce')
    
    cs_gdf['mag_cs_typ'] = cs_gdf['mag_cs_typ'].str.strip()
    # Mapping to a clean, standardized tier system.
    tier_map = {
        '[REDACTED_BY_SCRIPT]': 'Higher Tier',
        '[REDACTED_BY_SCRIPT]': 'Mid Tier',
        'Capital Grants': 'Capital Grants'
    }
    cs_gdf['cs_tier'] = cs_gdf['mag_cs_typ'].map(tier_map).fillna('Other')

    # Data Integrity Protocol
    cs_gdf.dropna(subset=['startdate', 'enddate', 'geometry'], inplace=True)
    cs_gdf = cs_gdf[cs_gdf.is_valid]

    # Performance Optimization: Create spatial index
    cs_gdf.sindex

    # Ensure the output directory exists before writing the artifact.
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Persist the sanitized artifact
    cs_gdf.to_file(output_path, driver='GPKG', engine='pyogrio')
    print(f"[REDACTED_BY_SCRIPT]")
    return cs_gdf


def phase_3_4_temporal_feature_engineering(gdf_base_temporal, cs_gdf):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    results_list = []
    
    for _, solar_farm in tqdm(gdf_base_temporal.iterrows(), total=gdf_base_temporal.shape[0]):
        solar_geom = solar_farm.geometry
        solar_date = solar_farm.date_validated
        solar_id = solar_farm.application_id
        
        # The Temporal Guard: Create a point-in-time snapshot.
        # CRITICAL FIX: Handle missing dates gracefully without dropping the row.
        if pd.isnull(solar_date):
             # If no date exists, we cannot determine overlap. Assume zero risk but KEEP the row.
             active_cs_gdf = gpd.GeoDataFrame(columns=cs_gdf.columns, crs=cs_gdf.crs)
        else:
             active_cs_gdf = cs_gdf[(cs_gdf['startdate'] <= solar_date) & (cs_gdf['enddate'] > solar_date)]
        
        features = {'application_id': solar_id}

        # Initialize all possible features with default "no impact" values.
        direct_impact_keys = ['cs_on_site_bool', 'cs_on_site_area_ha', 'cs_on_site_pct_area', 'cs_on_site_total_value']
        for key in direct_impact_keys: features[key] = 0
        features['cs_on_site_highest_tier'] = 'None'
        
        for name, radius in RISK_GRADIENT_RADII.items():
            features[f'cs_count_{name}'] = 0
            features[f'[REDACTED_BY_SCRIPT]'] = 0
            features[f'[REDACTED_BY_SCRIPT]'] = 0
            features[f'cs_total_area_ha_{name}'] = 0
            features[f'cs_density_{name}'] = 0.0
            features[f'cs_avg_value_{name}'] = 0.0

        # Edge Case Handling: If no agreements were active, append defaults and continue.
        if active_cs_gdf.empty:
            results_list.append(features)
            continue
            
        # Phase 4: Strategic Feature Engineering (Direct Impact)
        intersecting_cs = active_cs_gdf[active_cs_gdf.intersects(solar_geom)]
        if not intersecting_cs.empty:
            features['cs_on_site_bool'] = 1
            intersection_area = gpd.overlay(
                gpd.GeoDataFrame(geometry=[solar_geom], crs=PROJECT_CRS), 
                intersecting_cs, 
                how='intersection'
            ).area.sum()
            features['cs_on_site_area_ha'] = intersection_area / 10000
            features['cs_on_site_pct_area'] = (features['cs_on_site_area_ha'] * 10000) / solar_geom.area
            features['cs_on_site_total_value'] = intersecting_cs['totval_no'].sum()
            
            tier_order = pd.CategoricalDtype(['None', 'Other', 'Capital Grants', 'Mid Tier', 'Higher Tier'], ordered=True)
            intersecting_cs['cs_tier_ordered'] = intersecting_cs['cs_tier'].astype(tier_order)
            features['cs_on_site_highest_tier'] = intersecting_cs['cs_tier_ordered'].max()

        # Phase 4: Strategic Feature Engineering (Risk Gradient)
        for name, radius_m in RISK_GRADIENT_RADII.items():
            buffer_geom = solar_geom.buffer(radius_m)
            cs_in_buffer = active_cs_gdf[active_cs_gdf.intersects(buffer_geom)]
            
            if not cs_in_buffer.empty:
                features[f'cs_count_{name}'] = len(cs_in_buffer)
                features[f'[REDACTED_BY_SCRIPT]'] = (cs_in_buffer['cs_tier'] == 'Higher Tier').sum()
                features[f'[REDACTED_BY_SCRIPT]'] = (cs_in_buffer['cs_tier'] == 'Mid Tier').sum()
                features[f'cs_total_area_ha_{name}'] = cs_in_buffer.area.sum() / 10000
                features[f'cs_density_{name}'] = (cs_in_buffer.area.sum()) / buffer_geom.area
                features[f'cs_avg_value_{name}'] = cs_in_buffer['totval_no'].mean()

        results_list.append(features)
        
    print("[REDACTED_BY_SCRIPT]")
    return pd.DataFrame(results_list)


def phase_5_finalize_artifact(gdf_base_temporal, df_features, output_path):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Set index for a clean merge
    df_features.set_index('application_id', inplace=True)
    
    # Drop geometry from base GDF for final CSV output
    df_base = gdf_base_temporal.drop(columns=['geometry'])
    
    # Final Merge
    df_final = df_base.merge(df_features, on='application_id', how='left')
    
    # Verification: Ensure row count integrity
    if len(df_final) != len(gdf_base_temporal):
        raise RuntimeError("[REDACTED_BY_SCRIPT]")
        
    # Fill any NaNs that may have resulted from the merge (though 'how=left' should prevent this)
    # For feature columns, NaN should be filled with 0/default values.
    feature_cols = [col for col in df_features.columns if col != 'application_id']
    for col in feature_cols:
        if df_final[col].dtype in ['int64', 'float64']:
            df_final[col].fillna(0, inplace=True)
        else:
            df_final[col].fillna('None', inplace=True)

    # Ensure the output directory exists before writing the final artifact.
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Persist Final Artifact
    df_final.to_csv(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")
    return df_final


if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    
    # Phase 1: Create the temporally-aware base GeoDataFrame
    gdf_base_temporal = phase_1_prepare_temporal_base_artifact(
        L26_GRID_LPA_PATH, L41_ENV_PATH, PROJECT_CRS
    )
    
    # Phase 2: Ingest and sanitize the hostile source data, creating a clean L1 artifact.
    # This is an expensive operation; for subsequent runs, we could load the sanitized file directly.
    cs_gdf = phase_2_ingest_and_sanitize_cs_data(
        CS_SOURCE_PATH, CS_L1_SANITIZED_PATH, PROJECT_CRS
    )
    
    # Phases 3 & 4: Execute the core temporal logic and feature engineering.
    df_features = phase_3_4_temporal_feature_engineering(gdf_base_temporal, cs_gdf)
    
    # Phase 5: Assemble the final results and persist the L44 artifact.
    df_final_l44 = phase_5_finalize_artifact(gdf_base_temporal, df_features, L44_FINAL_ARTIFACT_PATH)
    
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]", df_final_l44.shape)
    print("[REDACTED_BY_SCRIPT]")
    print(df_final_l44[['application_id', 'cs_on_site_bool', 'cs_density_2km', 'cs_count_10km']].head())