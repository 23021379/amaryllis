import fiona
import geopandas as gpd
import pandas as pd
from pathlib import Path

# MANDATED RECLASSIFICATION MAPS (RE-ARCHITECTED)
RECLASS_MAPS = {
    'osm_buildings': {
        # NEW: Define a list of source columns in order of priority.
        'source_cols_priority': ['type', 'fclass'], 
        '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {
            # This map now contains keys from BOTH 'type' and 'fclass'
            'apartments': '[REDACTED_BY_SCRIPT]', 'house': '[REDACTED_BY_SCRIPT]',
            'residential': '[REDACTED_BY_SCRIPT]', 'detached': '[REDACTED_BY_SCRIPT]',
            'semidetached_house': '[REDACTED_BY_SCRIPT]', 'commercial': '[REDACTED_BY_SCRIPT]',
            'industrial': '[REDACTED_BY_SCRIPT]', 'office': '[REDACTED_BY_SCRIPT]',
            'retail': '[REDACTED_BY_SCRIPT]', 'supermarket': '[REDACTED_BY_SCRIPT]',
            'warehouse': '[REDACTED_BY_SCRIPT]',
            # CRITICAL: The 'building' key is now removed from the map. Generic tags that
            # are not explicitly mapped will result in a null super_class and be dropped,
            # but only after the primary 'type' column has been checked.
        }
    },
    'clc_land_cover': {
        'source_cols_priority': ['code_18'], 
        '[REDACTED_BY_SCRIPT]': 'EPSG:27700',
        'map': {
            '111': 'urban_fabric', '112': 'urban_fabric', '121': '[REDACTED_BY_SCRIPT]',
            '211': 'agricultural', '221': 'agricultural', '231': 'agricultural',
            '311': 'natural_protected_clc', '312': 'natural_protected_clc', '321': 'natural_protected_clc',
            '324': 'natural_protected_clc', '332': 'natural_protected_clc', '333': 'natural_protected_clc',
        }
    },
    'osm_roads': {
        'source_cols_priority': ['fclass'], 
        '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {
            'motorway': 'major_road', 'primary': 'major_road', 'secondary': 'major_road',
            'tertiary': 'major_road', 'trunk': 'major_road', 'residential': 'minor_road',
            'service': 'minor_road', 'unclassified': 'minor_road',
        }
    },
    'osm_landuse': {
        'source_cols_priority': ['fclass'], '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {
            'commercial': '[REDACTED_BY_SCRIPT]', 'industrial': '[REDACTED_BY_SCRIPT]',
            'retail': '[REDACTED_BY_SCRIPT]', 'quarry': '[REDACTED_BY_SCRIPT]',
        }
    },
    'osm_natural': {
        'source_cols_priority': ['code'], '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {
            '4101': 'spring', '4141': 'beach', '4112': 'cliff'
        }
    },
    'osm_railways': {
        'source_cols_priority': ['fclass'], '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': { 'rail': 'rail', 'light_rail': 'light_rail', 'subway': 'subway', 'narrow_gauge': 'narrow_gauge' }
    },
    'osm_places': {
        'source_cols_priority': ['fclass'], '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {
            'island': 'island', 'city': 'urban_area', 'town': 'urban_area',
            'village': 'urban_area', 'hamlet': 'urban_area', 'national_capital': 'urban_area',
            'farm': 'rural_area', 'locality': 'rural_area', 'suburb': 'rural_area'
        }
    },
}
TARGET_CRS = "EPSG:27700"

def _load_geospatial_file_in_chunks(file_path, chunk_size=50000):
    # This helper function is architecturally sound and requires no changes.
    with fiona.Env(SHAPE_RESTORE_SHX='YES'):
        try:
            str_path = str(file_path)
            with fiona.open(str_path, 'r') as source:
                total_features = len(source)
        except fiona.errors.DriverError as e:
            if str_path.lower().endswith('.shp'):
                raise IOError(f"[REDACTED_BY_SCRIPT]'{file_path}'. Restore failed.") from e
            else:
                raise e
        for start in range(0, total_features, chunk_size):
            stop = min(start + chunk_size, total_features)
            yield gpd.read_file(str_path, rows=slice(start, stop))

def forge_l1_artifact(source_files: list, output_path: Path, reclass_key: str):
    """
    Fortified V3.0 implementation, incorporating the Hierarchical Reclassification Protocol.
    """
    print(f"[REDACTED_BY_SCRIPT]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reclass_config = RECLASS_MAPS.get(reclass_key)
    if not reclass_config: raise ValueError(f"Reclass key '{reclass_key}' not found.")
    
    # --- MANDATE AD-PROXY-01.2: HIERARCHICAL RECLASSIFICATION ---
    source_cols = reclass_config['source_cols_priority']
    reclass_map = reclass_config['map']
    
    is_first_chunk = True
    for file_path in source_files:
        print(f"[REDACTED_BY_SCRIPT]")
        for gdf_chunk in _load_geospatial_file_in_chunks(file_path):
            # CRS & Schema logic remains sound.
            # CRS Unification and Schema Normalization (unchanged, remains correct)
            if gdf_chunk.crs is None:
                default_crs = reclass_config.get('[REDACTED_BY_SCRIPT]')
                if not default_crs: raise ValueError(f"Config Error for '{reclass_key}'[REDACTED_BY_SCRIPT]")
                print(f"[REDACTED_BY_SCRIPT]")
                gdf_chunk.set_crs(default_crs, inplace=True)
            if gdf_chunk.crs != TARGET_CRS:
                gdf_chunk = gdf_chunk.to_crs(TARGET_CRS)
            gdf_chunk.columns = [col.lower() for col in gdf_chunk.columns]

            # --- NEW HIERARCHICAL LOGIC ---
            # Initialize super_class column as empty (object type to hold Nones)
            gdf_chunk['super_class'] = None 

            for col in source_cols:
                if col not in gdf_chunk.columns:
                    print(f"[REDACTED_BY_SCRIPT]'{col}'[REDACTED_BY_SCRIPT]")
                    continue
                
                # Create a map for the current column's values
                current_map = gdf_chunk[col].astype(str).map(reclass_map)
                
                # Use .fillna() to coalesce. This is the core of the hierarchical logic.
                # It fills the nulls in 'super_class' with values from the current map,
                # but leaves existing, already-mapped values untouched.
                gdf_chunk['super_class'] = gdf_chunk['super_class'].fillna(current_map)
            # --- END OF NEW LOGIC ---

            gdf_chunk.dropna(subset=['super_class', 'geometry'], inplace=True)
            if gdf_chunk.empty:
                continue

            # Geometry integrity protocol remains sound.
            is_polygon_mask = gdf_chunk.geom_type.isin(['Polygon', 'MultiPolygon'])
            gdf_chunk.loc[is_polygon_mask, 'geometry'] = gdf_chunk.loc[is_polygon_mask, 'geometry'].buffer(0)

            # Persistence logic remains sound.
            if is_first_chunk:
                if output_path.exists(): output_path.unlink()
                gdf_chunk.to_file(output_path, driver='GPKG', layer=output_path.stem)
                is_first_chunk = False
            else:
                gdf_chunk.to_file(output_path, driver='GPKG', layer=output_path.stem, mode='a')

    if is_first_chunk:
        raise ValueError(f"Forge failed for '{output_path.name}'[REDACTED_BY_SCRIPT]")
    
    print(f"[REDACTED_BY_SCRIPT]")



def main():
    # This orchestrator function is architecturally sound and requires no changes.
    raw_data_dir = Path(r"[REDACTED_BY_SCRIPT]")
    l1_artifact_dir = Path(r"[REDACTED_BY_SCRIPT]")
    
    all_forge_jobs = [
        ('osm_buildings', ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']),
        ('clc_land_cover', ['clc2018_uk.shp']),
        ('osm_roads', ['[REDACTED_BY_SCRIPT]']),
        ('osm_landuse', ['[REDACTED_BY_SCRIPT]']),
        ('osm_places', ['[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]']),
        ('osm_railways', ['[REDACTED_BY_SCRIPT]']),
    ]
    
    for reclass_key, files in all_forge_jobs:
        forge_l1_artifact(
            source_files=[raw_data_dir / f for f in files],
            output_path=l1_artifact_dir / f'[REDACTED_BY_SCRIPT]',
            reclass_key=reclass_key
        )
    print("[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    main()