import fiona
import geopandas as gpd
import pandas as pd
from pathlib import Path

# MANDATED RECLASSIFICATION MAPS
# This implements the "[REDACTED_BY_SCRIPT]" mandate.
# These maps are the single source of truth for reducing noise from raw source data.
RECLASS_MAPS = {
    'osm_buildings': {
        'source_col': 'fclass',
        'map': {
            'apartments': '[REDACTED_BY_SCRIPT]',
            'house': '[REDACTED_BY_SCRIPT]',
            'residential': '[REDACTED_BY_SCRIPT]',
            'detached': '[REDACTED_BY_SCRIPT]',
            'semidetached_house': '[REDACTED_BY_SCRIPT]',
            'commercial': '[REDACTED_BY_SCRIPT]',
            'industrial': '[REDACTED_BY_SCRIPT]',
            'office': '[REDACTED_BY_SCRIPT]',
            'retail': '[REDACTED_BY_SCRIPT]',
            'supermarket': '[REDACTED_BY_SCRIPT]',
            'warehouse': '[REDACTED_BY_SCRIPT]',
        }
    },
    'clc_land_cover': {
        'source_col': 'code_18',
        'map': {
            '111': 'urban_fabric',
            '112': 'urban_fabric',
            '121': '[REDACTED_BY_SCRIPT]',
            '211': 'agricultural',
            '221': 'agricultural',
            '231': 'agricultural',
            '311': 'natural_protected', # Broadleaf forest
            '312': 'natural_protected', # Coniferous forest
            '321': 'natural_protected', # Natural grasslands
            '324': 'natural_protected', # Transitional woodland-shrub
            '332': 'natural_protected', # Bare rocks
            '333': 'natural_protected', # Sparsely vegetated areas
        }
    },
    'osm_roads': {
        'source_col': 'fclass',
        'map': {
            'motorway': 'major_road',
            'primary': 'major_road',
            'secondary': 'major_road',
            'tertiary': 'major_road',
            'trunk': 'major_road',
            'residential': 'minor_road',
            'service': 'minor_road',
            'unclassified': 'minor_road',
        }
    },
    # Add other maps for railways, landuse, natural, etc. as needed
}

TARGET_CRS = "EPSG:27700"

def _load_gpkg_in_chunks(gpkg_path, chunk_size=50000):
    """
    Generator to load large GeoPackage files in chunks to prevent memory overload.
    This directly mitigates failure Pattern 4: Resource Overload.
    """
    with fiona.open(gpkg_path, 'r') as source:
        total_features = len(source)
    
    for start in range(0, total_features, chunk_size):
        stop = min(start + chunk_size, total_features)
        # Use a slice object for robust row selection
        yield gpd.read_file(gpkg_path, rows=slice(start, stop))

def forge_l1_artifact(source_files: list, output_path: Path, reclass_key: str):
    """
    Executes the full ingestion, unification, and reclassification protocol.
    This function is the heart of "The Forge".
    """
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    reclass_config = RECLASS_MAPS.get(reclass_key)
    if not reclass_config:
        raise ValueError(f"[REDACTED_BY_SCRIPT]'{reclass_key}'[REDACTED_BY_SCRIPT]")

    source_col = reclass_config['source_col']
    reclass_map = reclass_config['map']
    
    is_first_chunk = True
    for file_path in source_files:
        print(f"[REDACTED_BY_SCRIPT]")
        for i, gdf_chunk in enumerate(_load_gpkg_in_chunks(file_path)):
            # --- MANDATE 1: Uncompromising CRS Unification ---
            # Mitigates Pattern 1: Geospatial Corruption.
            if gdf_chunk.crs != TARGET_CRS:
                gdf_chunk = gdf_chunk.to_crs(TARGET_CRS)

            # --- MANDATE 2: Schema Normalization & Reclassification ---
            # Mitigates Pattern 2: Data Contamination.
            gdf_chunk.columns = [col.lower() for col in gdf_chunk.columns]
            if source_col not in gdf_chunk.columns:
                print(f"[REDACTED_BY_SCRIPT]'{source_col}'[REDACTED_BY_SCRIPT]")
                continue

            gdf_chunk['super_class'] = gdf_chunk[source_col].astype(str).map(reclass_map)
            
            # --- MANDATE 3: Paranoid Filtering ---
            # Reduces artifact size and complexity.
            gdf_chunk.dropna(subset=['super_class', 'geometry'], inplace=True)
            if gdf_chunk.empty:
                continue

            # --- MANDATE 4: Geometric Integrity Check ---
            gdf_chunk.geometry = gdf_chunk.geometry.buffer(0)

            # Persist chunk to unified L1 artifact
            if is_first_chunk:
                gdf_chunk.to_file(output_path, driver='GPKG', layer=output_path.stem)
                is_first_chunk = False
            else:
                gdf_chunk.to_file(output_path, driver='GPKG', layer=output_path.stem, mode='a')
    
    # Final step from blueprint: Create spatial index for performance
    # This is typically handled by GeoPackage driver but can be done explicitly if needed.
    print(f"[REDACTED_BY_SCRIPT]")