import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import sys
from shapely.geometry import LineString

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input Artifacts
SEC_SUB_AREA_INPUT = r"[REDACTED_BY_SCRIPT]"
L25_DATA_INPUT = '[REDACTED_BY_SCRIPT]'
PRIMARY_SUB_L1_ARTIFACT = '[REDACTED_BY_SCRIPT]' # For friction feature upgrade

# Output Artifact
L26_DATA_OUTPUT = '[REDACTED_BY_SCRIPT]'

# Architectural Parameters
TARGET_CRS = "EPSG:27700"


def parse_and_clean_sec_sub_areas(gdf_raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Executes the robust parsing and decontamination protocol for the raw service area data.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf_raw.columns = [col.lower().strip() for col in gdf_raw.columns]

    # MANDATE: Geospatial Decontamination Gate (Pattern 1 Mitigation)
    initial_count = len(gdf_raw)
    invalid_geom_mask = gdf_raw.geometry.is_empty | gdf_raw.geometry.isna()
    if invalid_geom_mask.any():
        gdf_raw = gdf_raw[~invalid_geom_mask].copy()
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    # MANDATE: Reuse robust parsing logic from Directive 027
    gdf_raw['onan_rating_kva'] = pd.to_numeric(gdf_raw['onanrating'], errors='coerce')
    bands = gdf_raw['utilisation_band'].str.extract(r'(\d+)-(\d+)%')
    gdf_raw['utilisation_midpoint_pct'] = ((pd.to_numeric(bands[0], errors='coerce') + pd.to_numeric(bands[1], errors='coerce')) / 2) / 100.0
    
    # MANDATE: Create new geometric feature
    gdf_raw['sec_sub_area_sqm'] = gdf_raw.geometry.area
    
    # Schema Pruning
    essential_cols = [
        'geometry', 'sitefunctionallocation', 'customer_count',
        'onan_rating_kva', 'utilisation_midpoint_pct', 'sec_sub_area_sqm'
    ]
    return gdf_raw[[col for col in essential_cols if col in gdf_raw.columns]]


def main():
    """
    Main function to execute the deterministic linkage protocol for secondary sub areas.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # --- Phase 1: Ingest, Parse, and Prepare L1 Artifact ---
    try:
        gdf_sec_sub_areas_raw = gpd.read_file(SEC_SUB_AREA_INPUT)
        gdf_sec_sub_areas_raw = gdf_sec_sub_areas_raw.to_crs(TARGET_CRS)
        gdf_sec_sub_areas_l1 = parse_and_clean_sec_sub_areas(gdf_sec_sub_areas_raw)
        
        df_l25 = pd.read_csv(L25_DATA_INPUT)
        gdf_solar = gpd.GeoDataFrame(
            df_l25, geometry=gpd.points_from_xy(df_l25.easting_x, df_l25.northing_x), crs=TARGET_CRS
        )
        gdf_solar['solar_farm_id'] = gdf_solar.index # Ensure stable key
        
        gdf_primary_subs = gpd.read_file(PRIMARY_SUB_L1_ARTIFACT)
    except FileNotFoundError as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        sys.exit(1)

    # --- Phase 2: The Two-Stage, Fault-Tolerant Spatial Join ---
    logging.info("[REDACTED_BY_SCRIPT]'within')...")
    gdf_joined_raw = gpd.sjoin(gdf_solar, gdf_sec_sub_areas_l1, how='left', predicate='within')
    
    # MANDATE: Install duplicate resolution gate. In cases of overlapping polygons, a site
    # can be joined to multiple areas. We deterministically select the first match.
    gdf_joined = gdf_joined_raw.drop_duplicates(subset=['solar_farm_id'], keep='first')
    
    success_mask = gdf_joined['index_right'].notna()
    orphan_mask = ~success_mask
    gdf_joined['[REDACTED_BY_SCRIPT]'] = np.where(success_mask, 'within', 'nearest')
    
    if orphan_mask.any():
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        cols_to_drop = [col for col in gdf_sec_sub_areas_l1.columns if col != 'geometry'] + ['index_right']
        orphans = gdf_joined[orphan_mask].drop(columns=cols_to_drop)
        fallback_join_raw = gpd.sjoin_nearest(orphans, gdf_sec_sub_areas_l1, how='left')
        
        # MANDATE: Install duplicate resolution gate. In cases of equidistance, sjoin_nearest
        # can return multiple rows per orphan. We deterministically select the first match.
        fallback_join = fallback_join_raw[~fallback_join_raw.index.duplicated(keep='first')]
        
        gdf_joined.update(fallback_join)

    assert len(gdf_joined) == len(gdf_solar), "[REDACTED_BY_SCRIPT]"

    # --- Phase 3: Feature Synthesis and Deprecation ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # 1. Deprecate and Remove Obsolete k-NN Features
    knn_cols_to_drop = [col for col in gdf_joined.columns if '_knn50' in col]
    gdf_joined.drop(columns=knn_cols_to_drop, inplace=True)
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # 2. Direct Inheritance & Land Character Features
    gdf_joined.rename(columns={
        'utilisation_midpoint_pct': '[REDACTED_BY_SCRIPT]',
        'customer_count': '[REDACTED_BY_SCRIPT]',
        'onan_rating_kva': '[REDACTED_BY_SCRIPT]',
        'sec_sub_area_sqm': 'sec_sub_gov_area_sqm'
    }, inplace=True)
    
    gdf_joined['[REDACTED_BY_SCRIPT]'] = (
        gdf_joined['[REDACTED_BY_SCRIPT]'] / (gdf_joined['sec_sub_gov_area_sqm'] / 1_000_000)
    ).fillna(0)

    # 3. Sophisticated Friction Feature Upgrade
    logging.info("[REDACTED_BY_SCRIPT]")
    # Recreate connection paths (solar site to primary sub)
    gdf_nearest_primary = gpd.sjoin_nearest(gdf_solar, gdf_primary_subs, how='left').drop_duplicates(subset=['solar_farm_id'])
    nearest_primary_geoms = gdf_nearest_primary['index_right'].map(gdf_primary_subs.geometry)
    paths = [LineString([site, sub]) if pd.notna(sub) else None for site, sub in zip(gdf_solar.geometry, nearest_primary_geoms)]
    gdf_paths = gpd.GeoDataFrame({'solar_farm_id': gdf_solar.index}, geometry=paths, crs=TARGET_CRS).dropna()

    # Count intersections with the new service area polygons
    path_intersections = gpd.sjoin(gdf_paths, gdf_sec_sub_areas_l1, how='inner', predicate='intersects')
    intersection_counts = path_intersections.groupby('solar_farm_id').size().reset_index(name='[REDACTED_BY_SCRIPT]')
    
    # Merge the upgraded feature, replacing the old one
    gdf_joined = gdf_joined.merge(intersection_counts, on='solar_farm_id', how='left')
    gdf_joined.drop(columns=['[REDACTED_BY_SCRIPT]'], inplace=True, errors='ignore')
    gdf_joined.rename(columns={'[REDACTED_BY_SCRIPT]': '[REDACTED_BY_SCRIPT]'}, inplace=True)
    gdf_joined['[REDACTED_BY_SCRIPT]'].fillna(0, inplace=True)

    # --- Finalization ---
    df_final = gdf_joined.drop(columns=['geometry', 'index_right', 'sitefunctionallocation'], errors='ignore')
    
    df_final.to_csv(L26_DATA_OUTPUT, index=False)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()