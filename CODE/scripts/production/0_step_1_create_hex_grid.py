import geopandas as gpd
import pandas as pd
import h3pandas
import warnings

# Suppress FutureWarning from h3-pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Parameters from Architectural Directive AD-SURFACE-01 ---
SOURCE_GEOJSON = r"[REDACTED_BY_SCRIPT]"
TARGET_DNOS = ['UKPN', 'NGED']
PROJECT_CRS = "EPSG:27700"
H3_RESOLUTION = 7
OUTPUT_ARTIFACT_GPKG = r"[REDACTED_BY_SCRIPT]"
OUTPUT_ARTIFACT_GEOJSON = r"[REDACTED_BY_SCRIPT]"

print("[REDACTED_BY_SCRIPT]")


# --- Stage 1: Ingestion & Zonal Isolation ---
print("[REDACTED_BY_SCRIPT]")
dno_gdf = gpd.read_file(SOURCE_GEOJSON)

# PRE-OPERATIVE CHECK: Enforce CRS Mandate (Pattern 1)
assert dno_gdf.crs == PROJECT_CRS, f"[REDACTED_BY_SCRIPT]"

# Isolate target zones and create a unified boundary for efficiency
target_zones_gdf = dno_gdf[dno_gdf['DNO'].isin(TARGET_DNOS)].copy()
unified_boundary = target_zones_gdf.unary_union

print(f"[REDACTED_BY_SCRIPT]")
print("[REDACTED_BY_SCRIPT]")


# --- Stage 2: Hexagonal Tessellation ---
print("[REDACTED_BY_SCRIPT]")
# Create a temporary GeoDataFrame for the unified boundary to leverage h3-pandas
boundary_gdf = gpd.GeoDataFrame([1], geometry=[unified_boundary], crs=PROJECT_CRS)

# PRE-OPERATIVE INTERVENTION: The h3-pandas library requires EPSG:4326 for its internal
# polyfill algorithm. We must re-project the boundary *before* the call.
boundary_gdf_wgs84 = boundary_gdf.to_crs("EPSG:4326")

# Generate hex grid. The library will correctly process the WGS84 data and output in WGS84.
hex_grid_gdf_wgs84 = boundary_gdf_wgs84.h3.polyfill_resample(H3_RESOLUTION)

# POST-OPERATIVE CORRECTION: Immediately re-project the resulting grid back to the
# mandated project CRS to restore pipeline integrity.
hex_grid_gdf = hex_grid_gdf_wgs84.to_crs(PROJECT_CRS)

hex_grid_gdf.reset_index(inplace=True) # h3 index to column
hex_grid_gdf.rename(columns={'h3_polyfill': 'hex_id'}, inplace=True)

print(f"[REDACTED_BY_SCRIPT]")


# --- Stage 3: Zonal Attribution & Pruning ---
print("[REDACTED_BY_SCRIPT]")
# Perform the spatial join to prune the grid and attribute DNO information
pruned_grid_gdf = gpd.sjoin(hex_grid_gdf, target_zones_gdf, how="inner", predicate="intersects")

# Drop duplicate hex_ids that may arise if a hexagon spans two DNO polygons (rare but possible)
# The choice of keeping 'first' is arbitrary but ensures a unique hex_id primary key.
pruned_grid_gdf.drop_duplicates(subset='hex_id', keep='first', inplace=True)

# Select only necessary columns from the join result
pruned_grid_gdf = pruned_grid_gdf[['hex_id', 'DNO', 'geometry']]

print(f"[REDACTED_BY_SCRIPT]")


# --- Stage 4: Centroid Extraction & Finalization ---
print("[REDACTED_BY_SCRIPT]")
# Calculate centroids for the final point grid
centroids_geom = pruned_grid_gdf.geometry.centroid

# Construct the final artifact according to the specified schema
final_grid_centroids_gdf = gpd.GeoDataFrame({
    'hex_id': pruned_grid_gdf['hex_id'],
    'dno': pruned_grid_gdf['DNO'],
    'geometry': centroids_geom
}, crs=PROJECT_CRS)

print("[REDACTED_BY_SCRIPT]")


# --- Deliverable Artifact Generation ---
print("[REDACTED_BY_SCRIPT]")
final_grid_centroids_gdf.to_file(OUTPUT_ARTIFACT_GPKG, driver='GPKG')
print("[REDACTED_BY_SCRIPT]")
final_grid_centroids_gdf.to_file(OUTPUT_ARTIFACT_GEOJSON, driver='GeoJSON')

print("[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]'dno'[REDACTED_BY_SCRIPT]")
print("--- END DIRECTIVE ---")