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

    for _, site in solar_sites_gdf.iterrows():
        site_features = {'application_reference': site['application_reference']}
        site_geom = site.geometry
        
        # Non-buffered features (calculated once per site)
        major_places = l1_artifacts['places'][l1_artifacts['places']['fclass'].isin(['city', 'town', 'national_capital'])]
        nearest_geom = nearest_points(site_geom, major_places.unary_union)[1]
        site_features['[REDACTED_BY_SCRIPT]'] = site_geom.distance(nearest_geom) / 1000

        for r_m in radii_meters:
            r_km = int(r_m / 1000)
            buffer = gpd.GeoDataFrame([1], geometry=[site_geom.buffer(r_m)], crs=TARGET_CRS)
            buffer_area_sqkm = buffer.area.iloc[0] / 1e6

            # COHORT A: Human Settlement & Demand
            res_buildings = l1_artifacts['buildings'][l1_artifacts['buildings']['super_class'] == '[REDACTED_BY_SCRIPT]']
            intersected_buildings = gpd.overlay(buffer, res_buildings, how='intersection')
            site_features[f'[REDACTED_BY_SCRIPT]'] = intersected_buildings.area.sum() / buffer_area_sqkm if not intersected_buildings.empty else 0.0
            
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, l1_artifacts['land_cover'], ['urban_fabric'])
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, l1_artifacts['land_cover'], ['agricultural'])

            # COHORT B: Economic & Industrial Activity
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, l1_artifacts['land_cover'], ['[REDACTED_BY_SCRIPT]'])

            # COHORT C: Linear Infrastructure
            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_line_length(buffer, l1_artifacts['roads'], ['major_road'])
            # Assuming a 'railways' artifact exists and is loaded
            # site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_line_length(buffer, l1_artifacts['railways'], ['rail'])

            # COHORT D: Topographical & Natural Barriers
            # This is a placeholder. A real implementation requires a pre-generated DEM raster.
            # dem_stats = zonal_stats(buffer, '[REDACTED_BY_SCRIPT]', stats=['mean', 'std'])
            # site_features[f'[REDACTED_BY_SCRIPT]'] = dem_stats[0]['mean']
            # site_features[f'[REDACTED_BY_SCRIPT]'] = dem_stats[0]['std']
            site_features[f'[REDACTED_BY_SCRIPT]'] = np.nan # Placeholder
            site_features[f'[REDACTED_BY_SCRIPT]'] = np.nan # Placeholder

            site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, l1_artifacts['land_cover'], ['natural_protected'])

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