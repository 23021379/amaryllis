import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import numpy as np
import os
import warnings

# Suppress pandas SettingWithCopyWarning, as the operations are deliberate.
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)

# --- Artifact & Project Constants ---
L45_ENV_CS_ES_PATH = r"[REDACTED_BY_SCRIPT]"
NHLC_SOURCE_PATH = r"[REDACTED_BY_SCRIPT]"
NHLC_L1_SANITIZED_PATH = r"[REDACTED_BY_SCRIPT]"
L46_FINAL_ARTIFACT_PATH = r"[REDACTED_BY_SCRIPT]"

PROJECT_CRS = "EPSG:27700"
RISK_GRADIENT_RADII = {'2km': 2000, '5km': 5000, '10km': 10000}

# Mandated list for Surgical Triage of NHLC data
URBAN_TYPES = ['SETTLEMENT', 'COMMERCE', 'INDUSTRY', 'CIVIC PROVISION']


def phase_1_prepare_base_artifact(l45_path, crs):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    df_base = pd.read_csv(l45_path)

    # Create 'application_id' from the index for downstream compatibility.
    df_base['application_id'] = df_base.index
    
    # Mandated Implementation: Restore Geospatial State.
    gdf_base = gpd.GeoDataFrame(
        df_base,
        geometry=gpd.points_from_xy(df_base.easting_x, df_base.northing_x),
        crs=crs
    )
    
    print("[REDACTED_BY_SCRIPT]")
    return gdf_base


def phase_2_ingest_sanitize_and_triage_nhlc(source_path, sanitized_output_path, crs):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # High-Performance Ingestion
    nhlc_gdf = gpd.read_file(source_path, engine='pyogrio')
    
    # IMMEDIATE CRS Unification
    nhlc_gdf = nhlc_gdf.to_crs(crs)

    # Proactive Schema Normalization
    nhlc_gdf.columns = nhlc_gdf.columns.str.lower()

    # Data Integrity Protocol
    nhlc_gdf.dropna(subset=['geometry'], inplace=True)
    nhlc_gdf = nhlc_gdf[nhlc_gdf.is_valid]

    # The Surgical Triage: Isolate Urban Proxies
    print(f"[REDACTED_BY_SCRIPT]")
    # The column name is 'dominantbroadtype' after lowercasing
    urban_gdf = nhlc_gdf[nhlc_gdf['dominantbroadtype'].isin(URBAN_TYPES)].copy()
    
    # Mandated Implementation: Memory Management.
    print(f"[REDACTED_BY_SCRIPT]")
    del nhlc_gdf
    
    # Performance Optimization & Artifact Integrity
    urban_gdf.sindex
    output_dir = os.path.dirname(sanitized_output_path)
    os.makedirs(output_dir, exist_ok=True)
    urban_gdf.to_file(sanitized_output_path, driver='GPKG', engine='pyogrio')
    
    print(f"[REDACTED_BY_SCRIPT]")
    return urban_gdf


def phase_3_strategic_feature_engineering(gdf_base, urban_gdf):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    # --- Direct Impact Features ---
    print("[REDACTED_BY_SCRIPT]")
    # Use sjoin for efficient boolean check. This can create duplicates if one application intersects multiple urban polygons.
    intersecting_join = gpd.sjoin(gdf_base, urban_gdf, how='left', predicate='intersects')

    # Create a boolean series indicating an intersection, then drop duplicates to match gdf_base's length.
    # This correctly handles the one-to-many join problem.
    on_site_bool = intersecting_join.index_right.notna()
    on_site_bool = on_site_bool[~on_site_bool.index.duplicated(keep='first')]
    
    # Map the boolean results back to the original gdf_base.
    gdf_base['[REDACTED_BY_SCRIPT]'] = gdf_base.index.map(on_site_bool).fillna(False).astype(int)
    
    # Use overlay for precise area calculation
    overlay = gpd.overlay(gdf_base, urban_gdf, how='intersection')
    intersection_area = overlay.groupby('application_id')['geometry'].apply(lambda x: x.union_all().area)
    gdf_base['[REDACTED_BY_SCRIPT]'] = gdf_base['application_id'].map(intersection_area).fillna(0)
    gdf_base['[REDACTED_BY_SCRIPT]'] = gdf_base['[REDACTED_BY_SCRIPT]'] / gdf_base.geometry.area

    # --- Risk Gradient Features ---
    print("[REDACTED_BY_SCRIPT]")
    for name, radius_m in tqdm(RISK_GRADIENT_RADII.items()):
        buffers = gdf_base.copy()
        buffers['geometry'] = buffers.geometry.buffer(radius_m)
        buffer_area = buffers.geometry.area.iloc[0] # All buffers are the same size
        
        # Stage 1: Use a memory-efficient `sjoin` to identify all intersecting pairs without
        # creating a massive intermediate overlay artifact. This is the key to preventing the MemoryError.
        #
        # ARCHITECTURAL NOTE (Category 3 Pattern): The following `sjoin` is a deliberate
        # one-to-many join. It will create duplicate rows for each solar farm buffer that
        # intersects multiple urban polygons. This is intentional. The duplicated rows are
        # essential for the subsequent `groupby` operations which correctly calculate
        # cell counts and aggregate intersection areas. DO NOT de-duplicate this join.
        sjoined = gpd.sjoin(buffers, urban_gdf, how='inner', predicate='intersects')

        if not sjoined.empty:
            # Calculate cell count directly from the efficient sjoin result.
            cell_counts = sjoined.groupby('application_id').size()
            gdf_base[f'[REDACTED_BY_SCRIPT]'] = gdf_base['application_id'].map(cell_counts).fillna(0)
    
            # Stage 2: Perform targeted, vectorized intersection only on the pre-filtered pairs.
            # Look up the corresponding urban geometries using the `index_right` from the sjoin.
            urban_geoms_sjoined = urban_gdf.geometry.loc[sjoined.index_right]
            
            # MANDATORY FIX: Align the index of the target geometries to the sjoined result's index.
            # This ensures a correct, element-wise vectorized operation without index conflicts.
            urban_geoms_sjoined.index = sjoined.index
            
            # Perform the intersection using the correctly typed and indexed GeoSeries.
            intersection_geoms = sjoined.geometry.intersection(urban_geoms_sjoined, align=False)
            
            # Add the calculated intersection area and source urban type to the sjoined dataframe.
            sjoined['intersection_area'] = intersection_geoms.area
            
            # Calculate urban density (High-Value Feature)
            urban_area_in_buffer = sjoined.groupby('application_id')['intersection_area'].sum()
            gdf_base[f'[REDACTED_BY_SCRIPT]'] = gdf_base['application_id'].map(urban_area_in_buffer / buffer_area).fillna(0)
            
            # Calculate settlement-only density
            settlement_sjoined = sjoined[sjoined['dominantbroadtype'] == 'SETTLEMENT']
            settlement_area_in_buffer = settlement_sjoined.groupby('application_id')['intersection_area'].sum()
            gdf_base[f'[REDACTED_BY_SCRIPT]'] = gdf_base['application_id'].map(settlement_area_in_buffer / buffer_area).fillna(0)
        else:
            # If no intersections are found for this radius, create empty columns to maintain schema integrity.
            gdf_base[f'[REDACTED_BY_SCRIPT]'] = 0
            gdf_base[f'[REDACTED_BY_SCRIPT]'] = 0
            gdf_base[f'[REDACTED_BY_SCRIPT]'] = 0

    # Clean up intermediate columns
    gdf_base.drop(columns=['[REDACTED_BY_SCRIPT]'], inplace=True)
    
    print("[REDACTED_BY_SCRIPT]")
    return gdf_base


def phase_4_finalize_artifact(gdf_final, output_path):
    """[REDACTED_BY_SCRIPT]"""
    print("[REDACTED_BY_SCRIPT]")
    
    df_final = gdf_final.drop(columns=['geometry'])
    
    # Persist Final Artifact
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    
    print(f"[REDACTED_BY_SCRIPT]")
    return df_final


if __name__ == '__main__':
    print("[REDACTED_BY_SCRIPT]")
    
    # Phase 1: Prepare the base artifact from L45 output.
    gdf_base = phase_1_prepare_base_artifact(L45_ENV_CS_ES_PATH, PROJECT_CRS)
    
    # Phase 2: Ingest, sanitize, and triage the hostile NHLC data.
    # For efficiency, check if the sanitized & triaged L1 artifact already exists.
    if not os.path.exists(NHLC_L1_SANITIZED_PATH):
        urban_gdf = phase_2_ingest_sanitize_and_triage_nhlc(
            NHLC_SOURCE_PATH, NHLC_L1_SANITIZED_PATH, PROJECT_CRS
        )
    else:
        print(f"[REDACTED_BY_SCRIPT]")
        urban_gdf = gpd.read_file(NHLC_L1_SANITIZED_PATH, engine='pyogrio')

    # Phase 3: Execute the strategic feature engineering.
    gdf_final = phase_3_strategic_feature_engineering(gdf_base, urban_gdf)
    
    # Phase 4: Assemble the final results and persist the L46 artifact.
    df_final_l46 = phase_4_finalize_artifact(gdf_final, L46_FINAL_ARTIFACT_PATH)
    
    print("[REDACTED_BY_SCRIPT]")
    print("[REDACTED_BY_SCRIPT]", df_final_l46.shape)
    print("[REDACTED_BY_SCRIPT]")
    print(df_final_l46[['application_id', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']].head())