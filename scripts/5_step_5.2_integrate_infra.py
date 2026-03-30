import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import multiprocessing
from functools import partial
from tqdm import tqdm
from shapely.geometry import box

# --- CONFIGURATION ---
TARGET_CRS = "EPSG:27700"
L1_ARTIFACT_DIR = Path(r"[REDACTED_BY_SCRIPT]")
SOLAR_SITES_INPUT = Path(r"[REDACTED_BY_SCRIPT]")
FINAL_OUTPUT = Path(r"[REDACTED_BY_SCRIPT]")

# Architectural Parameters
RADII_METERS = [1000, 2000, 5000, 10000]
MAX_RADIUS = max(RADII_METERS)
NUM_WORKERS = multiprocessing.cpu_count() - 1

def _calculate_polygon_coverage(buffer_gdf, features_gdf, super_class_filter):
    if features_gdf is None or features_gdf.empty: return 0.0
    filtered = features_gdf[features_gdf['super_class'].isin(super_class_filter)]
    if filtered.empty: return 0.0
    intersected = gpd.overlay(buffer_gdf, filtered, how='intersection', keep_geom_type=True)
    if intersected.empty: return 0.0
    return intersected.area.sum() / buffer_gdf.area.iloc[0]

def _calculate_line_length(buffer_gdf, features_gdf, super_class_filter):
    if features_gdf is None or features_gdf.empty: return 0.0
    filtered = features_gdf[features_gdf['super_class'].isin(super_class_filter)]
    if filtered.empty: return 0.0
    clipped = gpd.clip(filtered, buffer_gdf)
    if clipped.empty: return 0.0
    return clipped.length.sum() / 1000

def process_site(site_tuple: tuple, l1_artifact_dir: Path):
    app_ref, site_geom = site_tuple
    site_features = {'application_reference': app_ref}
    bbox = box(
        site_geom.x - MAX_RADIUS, site_geom.y - MAX_RADIUS,
        site_geom.x + MAX_RADIUS, site_geom.y + MAX_RADIUS
    )
    try:
        local_assets = {
            'land_cover': gpd.read_file(l1_artifact_dir / '[REDACTED_BY_SCRIPT]', bbox=bbox),
            'roads': gpd.read_file(l1_artifact_dir / 'L1_osm_roads.gpkg', bbox=bbox),
            'landuse': gpd.read_file(l1_artifact_dir / 'L1_osm_landuse.gpkg', bbox=bbox),
            'railways': gpd.read_file(l1_artifact_dir / '[REDACTED_BY_SCRIPT]', bbox=bbox),
            #'buildings': gpd.read_file(l1_artifact_dir / '[REDACTED_BY_SCRIPT]', bbox=bbox),
        }
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return site_features

    for r_m in RADII_METERS:
        r_km = int(r_m / 1000)
        buffer = gpd.GeoDataFrame([1], geometry=[site_geom.buffer(r_m)], crs=TARGET_CRS)
        buffer_area_sqkm = buffer.area.iloc[0] / 1e6

        # res_buildings = local_assets['buildings'][local_assets['buildings']['super_class'] == '[REDACTED_BY_SCRIPT]']
        # intersected = gpd.overlay(buffer, res_buildings, how='intersection', keep_geom_type=True)
        # site_features[f'[REDACTED_BY_SCRIPT]'] = intersected.area.sum() / buffer_area_sqkm if not intersected.empty else 0.0
        
        site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, local_assets['land_cover'], ['urban_fabric'])
        site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, local_assets['land_cover'], ['agricultural'])
        site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, local_assets['landuse'], ['[REDACTED_BY_SCRIPT]'])
        site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_line_length(buffer, local_assets['roads'], ['major_road'])
        site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_line_length(buffer, local_assets['railways'], ['rail'])
        site_features[f'[REDACTED_BY_SCRIPT]'] = np.nan
        site_features[f'[REDACTED_BY_SCRIPT]'] = np.nan
        site_features[f'[REDACTED_BY_SCRIPT]'] = _calculate_polygon_coverage(buffer, local_assets['land_cover'], ['natural_protected_clc'])
    return site_features

def synthesize_sics(df: pd.DataFrame):
    # MANDATE: Eradicate ambiguous zeros. A gradient of 0 is valid; missing data must be a sentinel.
    df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'].fillna(-1.0)
    if 'railway_length_5km' not in df.columns: df['railway_length_5km'] = 0
    df['[REDACTED_BY_SCRIPT]'] = (df['[REDACTED_BY_SCRIPT]'] + df['railway_length_5km']) * df['[REDACTED_BY_SCRIPT]']
    df['[REDACTED_BY_SCRIPT]'] = df['[REDACTED_BY_SCRIPT]'] * df['[REDACTED_BY_SCRIPT]']
    return df

def main():
    print("[REDACTED_BY_SCRIPT]")
    solar_sites_df = pd.read_csv(SOLAR_SITES_INPUT)
    solar_sites_gdf = gpd.GeoDataFrame(
        solar_sites_df,
        geometry=gpd.points_from_xy(solar_sites_df.easting, solar_sites_df.northing),
        crs=TARGET_CRS
    )
    if 'application_reference' not in solar_sites_gdf.columns:
        solar_sites_gdf['application_reference'] = solar_sites_gdf.index

    print("[REDACTED_BY_SCRIPT]")
    # ARCHITECTURAL LOGGING: Acknowledge simplification in nearest city calculation.
    print("[REDACTED_BY_SCRIPT]")
    places_gdf = gpd.read_file(L1_ARTIFACT_DIR / 'L1_osm_places.gpkg')
    major_places = places_gdf[places_gdf['super_class'].isin(['city', 'town', 'national_capital'])].copy()
    joined = gpd.sjoin_nearest(solar_sites_gdf, major_places, how='left', distance_col='distance_m')
    distance_map = joined.drop_duplicates(subset='application_reference').set_index('application_reference')['distance_m']
    solar_sites_gdf['[REDACTED_BY_SCRIPT]'] = solar_sites_gdf['application_reference'].map(distance_map) / 1000

    site_tuples = list(zip(solar_sites_gdf.application_reference, solar_sites_gdf.geometry))
    print(f"[REDACTED_BY_SCRIPT]")
    worker_func = partial(process_site, l1_artifact_dir=L1_ARTIFACT_DIR)
    
    all_results = []
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(site_tuples), desc="Processing Sites") as pbar:
            for result in pool.imap_unordered(worker_func, site_tuples):
                if result: all_results.append(result)
                pbar.update()

    print("[REDACTED_BY_SCRIPT]")
    features_from_workers = pd.DataFrame(all_results)
    final_df = pd.merge(solar_sites_gdf.drop(columns='geometry'), features_from_workers, on='application_reference', how='left')

    print("[REDACTED_BY_SCRIPT]")
    final_df = synthesize_sics(final_df)
    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT, index=False)
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(final_df.head())

if __name__ == '__main__':
    main()