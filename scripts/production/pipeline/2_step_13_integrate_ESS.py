import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import os
import warnings

# Suppress pandas SettingWithCopyWarning, as the operations are deliberate.
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# Define project constants for file paths and CRS.
L44_ENV_CS_PATH = r"[REDACTED_BY_SCRIPT]"
ES_SOURCE_PATH = r"[REDACTED_BY_SCRIPT]"
ES_L1_SANITIZED_PATH = r"[REDACTED_BY_SCRIPT]"
L45_FINAL_ARTIFACT_PATH = r"[REDACTED_BY_SCRIPT]"

PROJECT_CRS = "EPSG:27700"
RISK_GRADIENT_RADII = {'2km': 2000, '5km': 5000, '10km': 10000, '20km': 20000}


def phase_1_prepare_base_artifact(l44_path, crs):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    df_base = pd.read_csv(l44_path)
    
    # Create 'application_id' from the index for downstream compatibility.
    df_base['application_id'] = df_base.index
    
    # CRITICAL MANDATE: VERIFY TEMPORAL ANCHOR.
    df_base['date_validated'] = pd.to_datetime(df_base['date_validated'], errors='coerce')
    
    # Data Integrity: Check for rows where the temporal anchor is invalid, but DO NOT drop them.
    num_null = df_base['date_validated'].isnull().sum()
    if num_null > 0:
        pct_null = (num_null / len(df_base)) * 100
        print(f"[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")

    # Mandated Implementation: Restore Geospatial State.
    gdf_base = gpd.GeoDataFrame(
        df_base,
        geometry=gpd.points_from_xy(df_base.easting_x, df_base.northing_x),
        crs=crs
    )
    
    print("[REDACTED_BY_SCRIPT]")
    return gdf_base


def phase_2_ingest_and_sanitize_es_data(source_path, output_path, crs):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # High-Performance Ingestion
    es_gdf = gpd.read_file(source_path, engine='pyogrio')
    
    # IMMEDIATE CRS Unification
    es_gdf = es_gdf.to_crs(crs)

    # Proactive Schema Normalization
    es_gdf.columns = es_gdf.columns.str.lower().str.replace(' ', '_')

    # Temporal, Numeric, and Categorical Sanitization
    es_gdf['startdat'] = pd.to_datetime(es_gdf['startdat'], format='%Y%m%d', errors='coerce')
    es_gdf['enddate'] = pd.to_datetime(es_gdf['enddate'], format='%Y%m%d', errors='coerce')
    
    es_gdf['totcost'] = pd.to_numeric(es_gdf['totcost'], errors='coerce')
    es_gdf['amtpaid'] = pd.to_numeric(es_gdf['amtpaid'], errors='coerce')

    # Strategic Triage: Create the critical 'es_tier' feature.
    es_gdf['es_tier'] = np.where(
        es_gdf['scheme'].str.contains("Higher Level", case=False, na=False), 
        'HLS', 
        'ELS'
    )

    # Data Integrity Protocol
    es_gdf.dropna(subset=['startdat', 'enddate', 'geometry'], inplace=True)
    es_gdf = es_gdf[es_gdf.is_valid]

    # Performance Optimization & Artifact Integrity
    es_gdf.sindex
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    es_gdf.to_file(output_path, driver='GPKG', engine='pyogrio')
    
    print(f"[REDACTED_BY_SCRIPT]")
    return es_gdf


def phase_3_4_temporal_feature_engineering(gdf_base, es_gdf):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    results_list = []
    
    for _, solar_farm in tqdm(gdf_base.iterrows(), total=gdf_base.shape[0]):
        solar_geom = solar_farm.geometry
        solar_date = solar_farm.date_validated
        solar_id = solar_farm.application_id
        
        features = {'application_id': solar_id}
        
        # The Temporal Guard
        # CRITICAL FIX: Handle missing dates gracefully without dropping the row.
        if pd.isnull(solar_date):
            # If no date exists, we cannot determine overlap. Assume zero risk but KEEP the row.
            active_es_gdf = gpd.GeoDataFrame(columns=es_gdf.columns, crs=es_gdf.crs)
        else:
            active_es_gdf = es_gdf[(es_gdf['startdat'] <= solar_date) & (es_gdf['enddate'] > solar_date)]
        
        if active_es_gdf.empty:
            # Populate with default "no impact" values and skip to the next farm
            features.update({
                'es_on_site_bool': 0, 'es_hls_on_site_bool': 0, 'es_hls_on_site_pct_area': 0.0,
                'es_on_site_highest_tier': 'None'
            })
            for name in RISK_GRADIENT_RADII.keys():
                features.update({
                    f'es_count_{name}': 0, f'es_hls_count_{name}': 0, f'es_total_area_ha_{name}': 0.0,
                    f'[REDACTED_BY_SCRIPT]': 0.0, f'[REDACTED_BY_SCRIPT]': 0.0
                })
            results_list.append(features)
            continue
            
        # The HLS Triage
        active_hls_gdf = active_es_gdf[active_es_gdf['es_tier'] == 'HLS']
        
        # Phase 4: Direct Impact Features
        intersecting_es = active_es_gdf[active_es_gdf.intersects(solar_geom)]
        intersecting_hls = active_hls_gdf[active_hls_gdf.intersects(solar_geom)]
        
        features['es_on_site_bool'] = 1 if not intersecting_es.empty else 0
        features['es_hls_on_site_bool'] = 1 if not intersecting_hls.empty else 0
        
        if features['es_hls_on_site_bool']:
            hls_intersection_area = gpd.overlay(
                gpd.GeoDataFrame(geometry=[solar_geom], crs=PROJECT_CRS), 
                intersecting_hls, how='intersection'
            ).area.sum()
            features['es_hls_on_site_pct_area'] = hls_intersection_area / solar_geom.area
            features['es_on_site_highest_tier'] = 'HLS'
        elif features['es_on_site_bool']:
            features['es_hls_on_site_pct_area'] = 0.0
            features['es_on_site_highest_tier'] = 'ELS'
        else:
            features['es_hls_on_site_pct_area'] = 0.0
            features['es_on_site_highest_tier'] = 'None'
            
        # Phase 4: Risk Gradient Features
        for name, radius_m in RISK_GRADIENT_RADII.items():
            buffer_geom = solar_geom.buffer(radius_m)
            es_in_buffer = active_es_gdf[active_es_gdf.intersects(buffer_geom)]
            hls_in_buffer = active_hls_gdf[active_hls_gdf.intersects(buffer_geom)]
            
            features[f'es_count_{name}'] = len(es_in_buffer)
            features[f'es_hls_count_{name}'] = len(hls_in_buffer)
            features[f'es_total_area_ha_{name}'] = es_in_buffer.area.sum() / 10000
            
            if not hls_in_buffer.empty:
                features[f'[REDACTED_BY_SCRIPT]'] = hls_in_buffer.area.sum() / buffer_geom.area
                features[f'[REDACTED_BY_SCRIPT]'] = hls_in_buffer['totcost'].mean()
            else:
                features[f'[REDACTED_BY_SCRIPT]'] = 0.0
                features[f'[REDACTED_BY_SCRIPT]'] = 0.0
                
        results_list.append(features)
        
    print("[REDACTED_BY_SCRIPT]")
    return pd.DataFrame(results_list)

def phase_5_finalize_artifact(df_base, df_features, output_path):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # Set index for a clean merge
    df_features.set_index('application_id', inplace=True)
    
    # Final Merge
    df_final = df_base.merge(df_features, on='application_id', how='left')
    
    # Verification and Cleanup
    if len(df_final) != len(df_base):
        raise RuntimeError("[REDACTED_BY_SCRIPT]")
    
    # Fill NaNs for farms that had no matching features, ensuring numeric columns are 0.
    feature_cols = [col for col in df_features.columns if col != 'application_id']
    for col in feature_cols:
        if 'tier' in col:
            df_final[col].fillna('None', inplace=True)
        elif pd.api.types.is_numeric_dtype(df_final[col]):
            df_final[col].fillna(0, inplace=True)

    # Persist Final Artifact
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    print(f"[REDACTED_BY_SCRIPT]")
    return df_final


if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    
    # Phase 1: Prepare the base artifact from L44 output.
    gdf_base = phase_1_prepare_base_artifact(L44_ENV_CS_PATH, PROJECT_CRS)
    
    # Phase 2: Ingest and sanitize the hostile ES data.
    # For efficiency, check if the sanitized L1 artifact already exists.
    if not os.path.exists(ES_L1_SANITIZED_PATH):
        es_gdf = phase_2_ingest_and_sanitize_es_data(
            ES_SOURCE_PATH, ES_L1_SANITIZED_PATH, PROJECT_CRS
        )
    else:
        print(f"[REDACTED_BY_SCRIPT]")
        es_gdf = gpd.read_file(ES_L1_SANITIZED_PATH, engine='pyogrio')

    # Phases 3 & 4: Execute the core temporal logic and feature engineering.
    df_features = phase_3_4_temporal_feature_engineering(gdf_base, es_gdf)
    
    # Phase 5: Assemble the final results and persist the L45 artifact.
    # We pass the original df_base (without geometry) for the final merge.
    df_final_l45 = phase_5_finalize_artifact(gdf_base.drop(columns=['geometry']), df_features, L45_FINAL_ARTIFACT_PATH)
    
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]", df_final_l45.shape)
    print("[REDACTED_BY_SCRIPT]")
    print(df_final_l45[['application_id', 'es_hls_on_site_bool', 'es_hls_density_5km', 'es_hls_count_20km']].head())