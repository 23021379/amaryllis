from pathlib import Path
import geopandas as gpd
import fiona
import geopandas as gpd
import pandas as pd
from pathlib import Path


# MANDATED RECLASSIFICATION MAPS (EXPANDED)
# This implements the "[REDACTED_BY_SCRIPT]" mandate for all cohorts.
RECLASS_MAPS = {
    'osm_buildings': {
        'source_col': 'fclass',
        '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {
            'apartments': '[REDACTED_BY_SCRIPT]', 'house': '[REDACTED_BY_SCRIPT]',
            'residential': '[REDACTED_BY_SCRIPT]', 'detached': '[REDACTED_BY_SCRIPT]',
            'semidetached_house': '[REDACTED_BY_SCRIPT]', 'commercial': '[REDACTED_BY_SCRIPT]',
            'industrial': '[REDACTED_BY_SCRIPT]', 'office': '[REDACTED_BY_SCRIPT]',
            'retail': '[REDACTED_BY_SCRIPT]', 'supermarket': '[REDACTED_BY_SCRIPT]',
            'warehouse': '[REDACTED_BY_SCRIPT]', 'building': None, # Keep quarantine for generic tag
        }
    },
    'clc_land_cover': {
        'source_col': 'code_18',
        '[REDACTED_BY_SCRIPT]': 'EPSG:27700',
        'map': {
            '111': 'urban_fabric', '112': 'urban_fabric', '121': '[REDACTED_BY_SCRIPT]',
            '211': 'agricultural', '221': 'agricultural', '231': 'agricultural',
            '311': 'natural_protected_clc', '312': 'natural_protected_clc', '321': 'natural_protected_clc',
            '324': 'natural_protected_clc', '332': 'natural_protected_clc', '333': 'natural_protected_clc',
        }
    },
    'osm_roads': {
        'source_col': 'fclass',
        '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {
            'motorway': 'major_road', 'primary': 'major_road', 'secondary': 'major_road',
            'tertiary': 'major_road', 'trunk': 'major_road', 'residential': 'minor_road',
            'service': 'minor_road', 'unclassified': 'minor_road',
        }
    },
    'osm_landuse': {
        'source_col': 'fclass',
        '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {
            'commercial': '[REDACTED_BY_SCRIPT]', 'industrial': '[REDACTED_BY_SCRIPT]',
            'retail': '[REDACTED_BY_SCRIPT]', 'quarry': '[REDACTED_BY_SCRIPT]',
        }
    },
    'osm_natural': {
        'source_col': 'fclass',
        '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {} # Pass-through: Data is unsuitable for area feature, will use CLC instead.
    },
    'osm_railways': {
        'source_col': 'fclass',
        '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': { 'rail': 'rail' }
    },
    'osm_places': {
        'source_col': 'fclass', # Column is used for filtering, not remapping.
        '[REDACTED_BY_SCRIPT]': 'EPSG:4326',
        'map': {} # An empty map signals no reclassification is needed.
    },
}

TARGET_CRS = "EPSG:27700"

def _load_geospatial_file_in_chunks(file_path, chunk_size=50000):
    """
    Generator to load large, fiona-compatible geospatial files (e.g., SHP, GPKG)
    in chunks to prevent memory overload. This directly mitigates failure
    Pattern 4: Resource Overload.
    """
    # --- MANDATE: On-the-fly repair of incomplete Shapefiles (Pattern 2) ---
    # Use fiona.Env to set the GDAL config option that restores the missing .shx file.
    with fiona.Env(SHAPE_RESTORE_SHX='YES'):
        try:
            # Hardening: Explicitly cast Path object to string for fiona compatibility.
            str_path = str(file_path)
            with fiona.open(str_path, 'r') as source:
                total_features = len(source)
        except fiona.errors.DriverError as e:
            # The previous error handler remains as a defense-in-depth measure
            # for unrecoverable errors beyond a missing .shx file.
            if str_path.lower().endswith('.shp'):
                raise IOError(
                    f"[REDACTED_BY_SCRIPT]'{file_path}'. "
                    "[REDACTED_BY_SCRIPT]"
                    "[REDACTED_BY_SCRIPT]"
                ) from e
            else:
                raise e

        for start in range(0, total_features, chunk_size):
            stop = min(start + chunk_size, total_features)
            # Use a slice object for robust row selection
            yield gpd.read_file(str_path, rows=slice(start, stop))

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
        for i, gdf_chunk in enumerate(_load_geospatial_file_in_chunks(file_path)):
            # --- MANDATE 1: Uncompromising CRS Unification (Architecturally Sound) ---
            # Mitigates Pattern 1 by using a configurable default CRS, preventing incorrect assumptions.
            if gdf_chunk.crs is None:
                default_crs = reclass_config.get('[REDACTED_BY_SCRIPT]')
                if not default_crs:
                    raise ValueError(
                        f"[REDACTED_BY_SCRIPT]'{reclass_key}': "
                        f"Chunk from '{file_path}' has no CRS, and no 'default_crs_if_missing' is defined."
                    )
                
                print(f"[REDACTED_BY_SCRIPT]")
                gdf_chunk.set_crs(default_crs, inplace=True)

            # Now that the CRS is guaranteed to be set, we can safely transform.
            if gdf_chunk.crs != TARGET_CRS:
                gdf_chunk = gdf_chunk.to_crs(TARGET_CRS)

            # --- MANDATE 2: Schema Normalization & Reclassification (Hardened) ---
            # Mitigates Pattern 2: Data Contamination by enforcing a strict schema contract.
            gdf_chunk.columns = [col.lower() for col in gdf_chunk.columns]
            
            if source_col not in gdf_chunk.columns:
                # This is a fatal error. The source data is unusable for reclassification.
                # Fail fast and provide a clear, actionable error message.
                raise ValueError(
                    f"[REDACTED_BY_SCRIPT]'{file_path}'. "
                    f"[REDACTED_BY_SCRIPT]'{source_col}'[REDACTED_BY_SCRIPT]"
                    f"[REDACTED_BY_SCRIPT]"
                )

            if reclass_map:
                # Standard case: Reclassify using the provided map.
                gdf_chunk['super_class'] = gdf_chunk[source_col].astype(str).map(reclass_map)
            else:
                # Pass-through case: Preserve original values for later filtering.
                gdf_chunk['super_class'] = gdf_chunk[source_col]

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
    
    # --- MANDATE: Enforce the Artifact Contract ---
    # If is_first_chunk is still True, it means no valid, classifiable data
    # was ever found and the output file was never created. This is a fatal error.
    if is_first_chunk:
        raise ValueError(
            f"[REDACTED_BY_SCRIPT]'{output_path.name}'[REDACTED_BY_SCRIPT]"
            f"[REDACTED_BY_SCRIPT]'s '{source_col}' values "
            f"[REDACTED_BY_SCRIPT]'{reclass_key}'[REDACTED_BY_SCRIPT]"
        )

    # Final step from blueprint: Create spatial index for performance
    # This is typically handled by GeoPackage driver but can be done explicitly if needed.
    print(f"[REDACTED_BY_SCRIPT]")

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points

# This is a placeholder for a complex GIS operation. In a real pipeline,
# this would be a pre-generated, project-wide asset.
# from rasterstats import zonal_stats

TARGET_CRS = "EPSG:27700"

def _calculate_polygon_coverage(buffer_gdf, features_gdf, super_class_filter):
    """[REDACTED_BY_SCRIPT]"""
    filtered_features = features_gdf[features_gdf['super_class'].isin(super_class_filter)]
    if filtered_features.empty:
        return 0.0
    
    # Use spatial index for performance
    intersected = gpd.overlay(buffer_gdf, filtered_features, how='intersection')
    if intersected.empty:
        return 0.0
        
    return intersected.area.sum() / buffer_gdf.area.iloc[0]

def _calculate_line_length(buffer_gdf, features_gdf, super_class_filter):
    """[REDACTED_BY_SCRIPT]"""
    filtered_features = features_gdf[features_gdf['super_class'].isin(super_class_filter)]
    if filtered_features.empty:
        return 0.0
    
    # Clip is more appropriate than intersection for lines
    clipped = gpd.clip(filtered_features, buffer_gdf)
    if clipped.empty:
        return 0.0
        
    return clipped.length.sum() / 1000 # Convert meters to km

def generate_proxy_features(solar_sites_gdf, l1_artifacts: dict):
    """
    Main engine to generate all proxy features for a given set of solar sites.
    Implements Phase 2 of the directive.
    """
    if solar_sites_gdf.crs != TARGET_CRS:
        raise ValueError(f"[REDACTED_BY_SCRIPT]")

    radii_meters = [1000, 2000, 5000, 10000]
    results = []

    # --- Pre-computation for Non-Buffered Features ---
    # Filter for major population centers once, outside the loop, for efficiency.
    major_places = l1_artifacts['places'][
        l1_artifacts['places']['fclass'].isin(['city', 'town', 'national_capital'])
    ].copy()
    
    # Use sjoin_nearest to efficiently find the index of the nearest place for each site
    joined = gpd.sjoin_nearest(solar_sites_gdf, major_places, how='left', distance_col='distance')
    # The result may have duplicate sites if a site is equidistant to multiple places.
    # We only care about the first nearest one.
    distance_map = joined.drop_duplicates(subset='application_reference').set_index('application_reference')['distance']

    for _, site in solar_sites_gdf.iterrows():
        site_features = {'application_reference': site['application_reference']}
        site_geom = site.geometry
        # --- Non-buffered features (calculated once per site) ---
        # Look up the pre-calculated distance and convert to km.
        distance_m = distance_map.get(site['application_reference'], np.nan)
        site_features['[REDACTED_BY_SCRIPT]'] = distance_m / 1000 if pd.notna(distance_m) else np.nan

        for r_m in radii_meters:
            r_km = int(r_m / 1000)
            buffer = gpd.GeoDataFrame([1], geometry=[site_geom.buffer(r_m)], crs=TARGET_CRS)
            buffer_area_sqkm = buffer.area.iloc[0] / 1e6

            # COHORT A: Human Settlement & Demand (De-quarantined)
            if 'buildings' in l1_artifacts and not l1_artifacts['buildings'].empty:
                res_buildings = l1_artifacts['buildings'][l1_artifacts['buildings']['super_class'] == '[REDACTED_BY_SCRIPT]']
                intersected_buildings = gpd.overlay(buffer, res_buildings, how='intersection')
                site_features[f'[REDACTED_BY_SCRIPT]'] = intersected_buildings.area.sum() / buffer_area_sqkm if not intersected_buildings.empty else 0.0
            else:
                site_features[f'[REDACTED_BY_SCRIPT]'] = 0.0 # No buildings means zero density
            
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, l1_artifacts['land_cover'], ['urban_fabric'])
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, l1_artifacts['land_cover'], ['agricultural'])

            # COHORT B: Economic & Industrial Activity (Using correct OSM source)
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, l1_artifacts['landuse'], ['[REDACTED_BY_SCRIPT]'])

            # COHORT C: Linear Infrastructure (Full implementation)
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_line_length(buffer, l1_artifacts['roads'], ['major_road'])
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_line_length(buffer, l1_artifacts['railways'], ['rail'])

            # COHORT D: Topographical & Natural Barriers (Using correct OSM source)
            # DEM features remain placeholders as per directive.
            site_features[f'[REDACTED_BY_SCRIPT]'] = np.nan
            site_features[f'[REDACTED_BY_SCRIPT]'] = np.nan

            # Re-routed to use superior Corine Land Cover data as per architectural directive.
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, l1_artifacts['land_cover'], ['natural_protected_clc'])

        results.append(site_features)

    feature_df = pd.DataFrame(results)
    return solar_sites_gdf.merge(feature_df, on='application_reference')


def synthesize_sics(feature_df: pd.DataFrame):
    """
    Synthesizes the Surgical Interaction Constructs for the imputation model.
    Implements Phase 3 of the directive.
    """
    # Fill NaNs from placeholder features to avoid calculation errors.
    # In a real run, these would be populated from the DEM.
    feature_df['[REDACTED_BY_SCRIPT]'] = feature_df['[REDACTED_BY_SCRIPT]'].fillna(0)
    # Assume a placeholder for railway length if not calculated
    if 'railway_length_5km' not in feature_df.columns:
        feature_df['railway_length_5km'] = 0

    # SIC-IMPUTE-01: "[REDACTED_BY_SCRIPT]"
    feature_df['[REDACTED_BY_SCRIPT]'] = (feature_df['[REDACTED_BY_SCRIPT]'] + feature_df['railway_length_5km']) * feature_df['[REDACTED_BY_SCRIPT]']

    # SIC-IMPUTE-02: "[REDACTED_BY_SCRIPT]"
    feature_df['[REDACTED_BY_SCRIPT]'] = feature_df['[REDACTED_BY_SCRIPT]'] * feature_df['[REDACTED_BY_SCRIPT]']
    
    return feature_df

def main():
    """
    Main pipeline orchestrator for the National Proxy Fabric.
    """
    # Define file paths
    raw_data_dir = Path(r"[REDACTED_BY_SCRIPT]")
    l1_artifact_dir = Path(r"[REDACTED_BY_SCRIPT]")
    output_dir = Path(r"[REDACTED_BY_SCRIPT]")
    
    # --- PHASE 1: THE FORGE ---
    # Execute the forging process for each data cohort, using the correct source file formats.
    
    # --- PHASE 1: THE FORGE ---
    # De-quarantined. Processing the newly acquired complete dataset.
    forge_l1_artifact(
        source_files=[
            raw_data_dir / '[REDACTED_BY_SCRIPT]',
            raw_data_dir / '[REDACTED_BY_SCRIPT]'
        ],
        output_path=l1_artifact_dir / '[REDACTED_BY_SCRIPT]',
        reclass_key='osm_buildings'
    )
    forge_l1_artifact(
        source_files=[raw_data_dir / 'clc2018_uk.shp'],
        output_path=l1_artifact_dir / '[REDACTED_BY_SCRIPT]',
        reclass_key='clc_land_cover'
    )
    forge_l1_artifact(
        source_files=[raw_data_dir / '[REDACTED_BY_SCRIPT]'],
        output_path=l1_artifact_dir / 'L1_osm_roads.gpkg',
        reclass_key='osm_roads'
    )
    forge_l1_artifact(
        source_files=[raw_data_dir / '[REDACTED_BY_SCRIPT]'],
        output_path=l1_artifact_dir / 'L1_osm_landuse.gpkg',
        reclass_key='osm_landuse'
    )
    forge_l1_artifact(
        source_files=[
            raw_data_dir / '[REDACTED_BY_SCRIPT]',
            raw_data_dir / '[REDACTED_BY_SCRIPT]'
        ],
        output_path=l1_artifact_dir / 'L1_osm_natural.gpkg',
        reclass_key='osm_natural'
    )
    forge_l1_artifact(
        source_files=[
            raw_data_dir / '[REDACTED_BY_SCRIPT]',
            raw_data_dir / '[REDACTED_BY_SCRIPT]'
        ],
        output_path=l1_artifact_dir / 'L1_osm_places.gpkg',
        reclass_key='osm_places' # Will pass through without reclassification
    )
    forge_l1_artifact(
        source_files=[raw_data_dir / '[REDACTED_BY_SCRIPT]'],
        output_path=l1_artifact_dir / '[REDACTED_BY_SCRIPT]',
        reclass_key='osm_railways'
    )

    # --- PHASE 2 & 3: THE ENGINE & SIC SYNTHESIS ---
    print("[REDACTED_BY_SCRIPT]")
    
    # Load all clean L1 artifacts
    l1_artifacts = {
        'land_cover': gpd.read_file(l1_artifact_dir / '[REDACTED_BY_SCRIPT]'),
        'roads': gpd.read_file(l1_artifact_dir / 'L1_osm_roads.gpkg'),
        'landuse': gpd.read_file(l1_artifact_dir / 'L1_osm_landuse.gpkg'),
        'places': gpd.read_file(l1_artifact_dir / 'L1_osm_places.gpkg'),
        'railways': gpd.read_file(l1_artifact_dir / '[REDACTED_BY_SCRIPT]'),
    }


    # Load the primary solar applications file from its tabular source (e.g., CSV)
    solar_sites_df = pd.read_csv(r"[REDACTED_BY_SCRIPT]")

    # --- MANDATE: Create and validate geometry from source coordinates ---
    # This is a critical control point to prevent Pattern 1 (Geospatial Corruption).
    # We create a geometry column and immediately assign the correct, project-wide CRS.
    solar_sites_gdf = gpd.GeoDataFrame(
        solar_sites_df,
        geometry=gpd.points_from_xy(solar_sites_df.easting, solar_sites_df.northing),
        crs="EPSG:27700"
    )
    
    # The feature engine requires a unique identifier. The source data has no obvious key,
    # so we create a robust one.
    if 'application_reference' not in solar_sites_gdf.columns:
        solar_sites_gdf['application_reference'] = solar_sites_gdf.index

    # Generate the core proxy features
    # The engine now receives a correctly formatted and projected GeoDataFrame.
    features_df = generate_proxy_features(solar_sites_gdf, l1_artifacts)

    # Synthesize the final SICs
    final_imputation_input_df = synthesize_sics(features_df)

    # Save the final output
    output_path = output_dir / '[REDACTED_BY_SCRIPT]'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_imputation_input_df.to_csv(output_path, index=False)

    print(f"[REDACTED_BY_SCRIPT]")
    print(final_imputation_input_df.head())


if __name__ == '__main__':
    main()