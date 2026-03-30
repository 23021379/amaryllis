import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
import fiona
from shapely.ops import nearest_points
from tqdm import tqdm
from functools import partial

# --- Project Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

# Input paths
INFRA_DIR = Path(r"[REDACTED_BY_SCRIPT]")
OSM_DIR = Path(r"[REDACTED_BY_SCRIPT]")

BUILDINGS_SOURCE = [OSM_DIR / "[REDACTED_BY_SCRIPT]"]
LAND_COVER_SOURCE = [INFRA_DIR / "clc2018_uk.shp"]
ROADS_SOURCE = [OSM_DIR / "[REDACTED_BY_SCRIPT]"]
PLACES_SOURCE = [OSM_DIR / "[REDACTED_BY_SCRIPT]"]

# Artifact paths
ARTIFACT_DIR = INFRA_DIR / 'artifacts'
L1_BUILDINGS_CACHE = ARTIFACT_DIR / "[REDACTED_BY_SCRIPT]"
L1_LAND_COVER_CACHE = ARTIFACT_DIR / "[REDACTED_BY_SCRIPT]"
L1_ROADS_CACHE = ARTIFACT_DIR / "[REDACTED_BY_SCRIPT]"
L1_PLACES_CACHE = ARTIFACT_DIR / "[REDACTED_BY_SCRIPT]"

# Geospatial constants
TARGET_CRS = "EPSG:27700"
NULL_SENTINEL = 0

# --- RECLASSIFICATION MAPS ---
RECLASS_MAPS = {
    'buildings': {
        'source_col': 'fclass',
        'map': {
            'apartments': 'residential', 'house': 'residential', 'residential': 'residential',
            'detached': 'residential', 'semidetached_house': 'residential',
            'commercial': '[REDACTED_BY_SCRIPT]', 'industrial': '[REDACTED_BY_SCRIPT]',
            'office': '[REDACTED_BY_SCRIPT]', 'retail': '[REDACTED_BY_SCRIPT]',
            'supermarket': '[REDACTED_BY_SCRIPT]', 'warehouse': '[REDACTED_BY_SCRIPT]',
        }
    },
    'land_cover': {
        'source_col': 'code_18',
        'map': {
            '111': 'urban_fabric', '112': 'urban_fabric', '121': '[REDACTED_BY_SCRIPT]',
            '211': 'agricultural', '221': 'agricultural', '231': 'agricultural',
            '311': 'natural_protected', '312': 'natural_protected', '321': 'natural_protected',
            '324': 'natural_protected', '332': 'natural_protected', '333': 'natural_protected',
        }
    },
    'roads': {
        'source_col': 'fclass',
        'map': {
            'motorway': 'major_road', 'primary': 'major_road', 'secondary': 'major_road',
            'tertiary': 'major_road', 'trunk': 'major_road',
            'residential': 'minor_road', 'service': 'minor_road', 'unclassified': 'minor_road',
        }
    },
}

# --- ARTIFACT CREATION ---

def _load_gpkg_in_chunks(gpkg_path, chunk_size=50000):
    """[REDACTED_BY_SCRIPT]"""
    with fiona.open(gpkg_path, 'r') as source:
        total_features = len(source)
    for start in range(0, total_features, chunk_size):
        yield gpd.read_file(gpkg_path, rows=slice(start, min(start + chunk_size, total_features)))

def get_or_create_l1_infra_artifact(source_files, cache_path, reclass_key):
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    reclass_config = RECLASS_MAPS.get(reclass_key)
    if not reclass_config:
        raise ValueError(f"[REDACTED_BY_SCRIPT]'{reclass_key}' not found.")

    source_col, reclass_map = reclass_config['source_col'], reclass_config['map']
    
    all_chunks = []
    for file_path in source_files:
        for gdf_chunk in _load_gpkg_in_chunks(file_path):
            gdf_chunk = gdf_chunk.to_crs(TARGET_CRS)
            gdf_chunk.columns = [col.lower() for col in gdf_chunk.columns]
            if source_col in gdf_chunk.columns:
                gdf_chunk['super_class'] = gdf_chunk[source_col].astype(str).map(reclass_map)
                gdf_chunk.dropna(subset=['super_class', 'geometry'], inplace=True)
                if not gdf_chunk.empty:
                    gdf_chunk.geometry = gdf_chunk.geometry.buffer(0)
                    all_chunks.append(gdf_chunk[['super_class', 'geometry']])
    
    if not all_chunks:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        return gpd.GeoDataFrame(columns=['super_class', 'geometry'], crs=TARGET_CRS)

    master_gdf = pd.concat(all_chunks, ignore_index=True)
    master_gdf = gpd.GeoDataFrame(master_gdf, crs=TARGET_CRS)
    
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    master_gdf.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf

def get_or_create_l1_places_artifact(source_files, cache_path):
    if os.path.exists(cache_path):
        logging.info(f"[REDACTED_BY_SCRIPT]")
        return gpd.read_parquet(cache_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")
    all_chunks = []
    for file_path in source_files:
        # Use chunked loading for potentially large shapefiles
        for gdf_chunk in _load_gpkg_in_chunks(file_path):
            gdf_chunk = gdf_chunk.to_crs(TARGET_CRS)
            gdf_chunk.columns = [col.lower() for col in gdf_chunk.columns]
            all_chunks.append(gdf_chunk)
    
    master_gdf = pd.concat(all_chunks, ignore_index=True)
    master_gdf = gpd.GeoDataFrame(master_gdf, crs=TARGET_CRS)
    master_gdf.to_parquet(cache_path)
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf

# --- FEATURE CALCULATION HELPERS ---

def _calculate_polygon_coverage(buffer_gdf, features_gdf, super_class_filter):
    filtered = features_gdf[features_gdf['super_class'].isin(super_class_filter)]
    if filtered.empty: return 0.0
    intersected = gpd.overlay(buffer_gdf, filtered, how='intersection')
    return intersected.area.sum() / buffer_gdf.area.iloc[0] if not intersected.empty else 0.0

def _calculate_line_length(buffer_gdf, features_gdf, super_class_filter):
    filtered = features_gdf[features_gdf['super_class'].isin(super_class_filter)]
    if filtered.empty: return 0.0
    clipped = gpd.clip(filtered, buffer_gdf)
    return clipped.length.sum() / 1000 if not clipped.empty else 0.0 # Return km

# --- FEATURE CALCULATION ---

def calculate_proxies(site, buildings, land_cover, roads, places, radii_meters):
    """
    Calculates all infrastructure proxy features for a single site.
    The site is a pandas Series from master_gdf.iterrows().
    """
    # V6.4 FIX: The site ID is now the index of the series, not a column.
    site_id = site.name
    site_geom = site.geometry
    
    all_features = {'hex_id': site_id}

    # 1. Proximity Features (Distance to nearest)
    if not buildings.empty:
        nearest_building = nearest_points(site_geom, buildings.unary_union)[1]
        all_features['[REDACTED_BY_SCRIPT]'] = site_geom.distance(nearest_building)
    else:
        all_features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL

    if not places.empty:
        nearest_place = nearest_points(site_geom, places.unary_union)[1]
        all_features['[REDACTED_BY_SCRIPT]'] = site_geom.distance(nearest_place)
    else:
        all_features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL

    # 2. Area Proxies
    for r_m in radii_meters:
        r_km = r_m // 1000
        buffer = gpd.GeoDataFrame(geometry=[site_geom.buffer(r_m)], crs=TARGET_CRS)
        
        # Human Settlement & Demand
        all_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, land_cover, ['urban_fabric'])
        all_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, land_cover, ['agricultural'])
        
        # Economic & Industrial Activity
        all_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, land_cover, ['[REDACTED_BY_SCRIPT]'])
        
        # Linear Infrastructure
        all_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_line_length(buffer, roads, ['major_road'])
        
        # Natural Barriers
        all_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, land_cover, ['natural_protected'])

    return all_features

# --- MAIN EXECUTOR FUNCTION ---

def execute(master_gdf):
    """
    Executor entry point for integrating infrastructure proxy features.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # V14 ID Management
    id_col = 'hex_id'
    if master_gdf.index.name != id_col:
        if id_col in master_gdf.columns:
            master_gdf.set_index(id_col, inplace=True)
        else:
            logging.error(f"FATAL: '{id_col}'[REDACTED_BY_SCRIPT]")
            return master_gdf

    try:
        buildings = get_or_create_l1_infra_artifact(BUILDINGS_SOURCE, L1_BUILDINGS_CACHE, 'buildings')
        land_cover = get_or_create_l1_infra_artifact(LAND_COVER_SOURCE, L1_LAND_COVER_CACHE, 'land_cover')
        roads = get_or_create_l1_infra_artifact(ROADS_SOURCE, L1_ROADS_CACHE, 'roads')
        places = get_or_create_l1_places_artifact(PLACES_SOURCE, L1_PLACES_CACHE)
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf

    # This dataframe will accumulate all new features, indexed by hex_id.
    features_df = pd.DataFrame(index=master_gdf.index)

    # --- Phase 1: Proximity Features (Vectorized) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    if not buildings.empty:
        buildings_union = buildings.unary_union
        features_df['[REDACTED_BY_SCRIPT]'] = master_gdf.geometry.apply(lambda g: g.distance(nearest_points(g, buildings_union)[1]))
    if not places.empty:
        places_union = places.unary_union
        features_df['[REDACTED_BY_SCRIPT]'] = master_gdf.geometry.apply(lambda g: g.distance(nearest_points(g, places_union)[1]))

    # --- Phase 2: Area & Length Proxies (Vectorized) ---
    logging.info("[REDACTED_BY_SCRIPT]")
    land_cover_sindex = land_cover.sindex
    roads_sindex = roads.sindex
    radii_meters = [1000, 2000, 5000, 10000]

    for r_m in tqdm(radii_meters, desc="Processing Radii"):
        r_km = r_m // 1000
        logging.info(f"[REDACTED_BY_SCRIPT]")
        
        buffers_gdf = gpd.GeoDataFrame(geometry=master_gdf.geometry.buffer(r_m), index=master_gdf.index)
        buffer_area = buffers_gdf.geometry.iloc[0].area

        # --- Land Cover ---
        possible_matches_idx = list(land_cover_sindex.intersection(buffers_gdf.total_bounds))
        if possible_matches_idx:
            lc_subset = land_cover.iloc[possible_matches_idx]
            intersected_lc = gpd.overlay(buffers_gdf.reset_index(), lc_subset, how='intersection')
            intersected_lc['area'] = intersected_lc.geometry.area
            coverage = intersected_lc.groupby([id_col, 'super_class'])['area'].sum() / buffer_area
            coverage = coverage.unstack(level='super_class').fillna(0)
            coverage.columns = [f'[REDACTED_BY_SCRIPT]' for col in coverage.columns]
            features_df = features_df.join(coverage, how='left')

        # --- Roads ---
        possible_matches_idx_roads = list(roads_sindex.intersection(buffers_gdf.total_bounds))
        if possible_matches_idx_roads:
            roads_subset = roads.iloc[possible_matches_idx_roads]
            intersected_roads = gpd.overlay(buffers_gdf.reset_index(), roads_subset, how='intersection', keep_geom_type=True)
            intersected_roads['length_km'] = intersected_roads.geometry.length / 1000
            road_lengths = intersected_roads.groupby([id_col, 'super_class'])['length_km'].sum()
            road_lengths = road_lengths.unstack(level='super_class').fillna(0)
            road_lengths.columns = [f'[REDACTED_BY_SCRIPT]' for col in road_lengths.columns]
            features_df = features_df.join(road_lengths, how='left')

    # --- Synthesize SICs ---
    logging.info("[REDACTED_BY_SCRIPT]")
    terrain_gradient = master_gdf.get('terrain_gradient_5km', 0)
    features_df['[REDACTED_BY_SCRIPT]'] = features_df.get('[REDACTED_BY_SCRIPT]', 0) * terrain_gradient
    features_df['[REDACTED_BY_SCRIPT]'] = features_df.get('[REDACTED_BY_SCRIPT]', 0) * features_df.get('[REDACTED_BY_SCRIPT]', 0)
    
    # --- Merge & Finalize ---
    logging.info("[REDACTED_BY_SCRIPT]")
    final_gdf = master_gdf.join(features_df)
    
    new_cols = [col for col in features_df.columns if col not in master_gdf.columns]
    final_gdf[new_cols] = final_gdf[new_cols].fillna(NULL_SENTINEL)

    logging.info(f"[REDACTED_BY_SCRIPT]")
    return final_gdf
