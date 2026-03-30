import os
import logging
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, box
from sklearn.neighbors import BallTree

# --- CONFIGURATION ---
BASE_DATA_DIR = r"[REDACTED_BY_SCRIPT]"
L1_MASTER_ASSET_ARTIFACT = os.path.join(BASE_DATA_DIR, r"[REDACTED_BY_SCRIPT]")
SUBSTATION_L1_ARTIFACT = os.path.join(BASE_DATA_DIR, "[REDACTED_BY_SCRIPT]")

TARGET_CRS = "EPSG:27700"
DENSITY_RADII_M = [2000, 5000, 10000]
MAX_RADIUS_M = max(DENSITY_RADII_M)

# --- MODULE-LEVEL STATE ---
gdf_subs, tree_subs = None, None

def _initialize_module_state():
    """[REDACTED_BY_SCRIPT]"""
    global gdf_subs, tree_subs
    if tree_subs is not None: return

    try:
        logging.info("[REDACTED_BY_SCRIPT]")
        gdf_subs = gpd.read_file(SUBSTATION_L1_ARTIFACT).to_crs(TARGET_CRS)
        assert gdf_subs.crs.to_string() == TARGET_CRS, "[REDACTED_BY_SCRIPT]"
        coords_subs = np.array(list(zip(gdf_subs.geometry.x, gdf_subs.geometry.y)))
        tree_subs = BallTree(coords_subs, metric='euclidean')
        logging.info("[REDACTED_BY_SCRIPT]")
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        tree_subs = "INIT_FAILED"

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    if tree_subs is None: _initialize_module_state()
    if tree_subs == "INIT_FAILED":
        state['errors'].append("[REDACTED_BY_SCRIPT]")
        return state

    try:
        site_geom = state['input_geom']
        
        # --- SURGICAL QUERY PROTOCOL ---
        bbox = box(
            site_geom.x - MAX_RADIUS_M, site_geom.y - MAX_RADIUS_M,
            site_geom.x + MAX_RADIUS_M, site_geom.y + MAX_RADIUS_M
        )
        local_points = gpd.read_file(L1_MASTER_ASSET_ARTIFACT, layer='points', bbox=bbox)
        local_lines = gpd.read_file(L1_MASTER_ASSET_ARTIFACT, layer='lines', bbox=bbox)

        # --- FEATURE ENGINEERING (POINTS) ---
        if not local_points.empty:
            local_points['distance_m'] = local_points.distance(site_geom)
            for _, row in local_points.loc[local_points.groupby(['voltage', 'asset_tag'])['distance_m'].idxmin()].iterrows():
                state['features'][f"[REDACTED_BY_SCRIPT]"] = row.distance_m

            for radius in DENSITY_RADII_M:
                radius_km_str = f"[REDACTED_BY_SCRIPT]"
                area_km2 = np.pi * ((radius / 1000) ** 2)
                points_in_buffer = local_points[local_points['distance_m'] <= radius]
                if not points_in_buffer.empty:
                    for (v, t), count in points_in_buffer.groupby(['voltage', 'asset_tag']).size().items():
                        state['features'][f"[REDACTED_BY_SCRIPT]"] = count
                        state['features'][f"[REDACTED_BY_SCRIPT]"] = count / area_km2

        # --- FEATURE ENGINEERING (LINES) ---
        if not local_lines.empty:
            local_lines['corridor_geom'] = local_lines.apply(lambda r: r.geometry.buffer(r.buffer_m), axis=1)
            local_lines_gdf = local_lines.set_geometry('corridor_geom')
            
            local_lines_gdf['distance_to_corridor_m'] = local_lines_gdf.distance(site_geom)
            for _, row in local_lines_gdf.loc[local_lines_gdf.groupby(['voltage', 'asset_tag'])['distance_to_corridor_m'].idxmin()].iterrows():
                state['features'][f"[REDACTED_BY_SCRIPT]"] = row.distance_to_corridor_m
            
            for _, row in local_lines_gdf[local_lines_gdf.intersects(site_geom)].iterrows():
                state['features'][f"[REDACTED_BY_SCRIPT]"] = 1

            _, idx = tree_subs.query([[site_geom.x, site_geom.y]], k=1)
            nearest_sub_geom = gdf_subs.geometry.iloc[idx[0][0]]
            connection_path = LineString([site_geom, nearest_sub_geom])
            for (v, t), count in local_lines_gdf[local_lines_gdf.intersects(connection_path)].groupby(['voltage', 'asset_tag']).size().items():
                state['features'][f"[REDACTED_BY_SCRIPT]"] = count

            for radius in DENSITY_RADII_M:
                site_buffer = site_geom.buffer(radius)
                clipped_lines = gpd.overlay(local_lines.set_geometry('geometry'), gpd.GeoDataFrame(geometry=[site_buffer], crs=TARGET_CRS), how='intersection')
                if not clipped_lines.empty:
                    clipped_lines['length_km'] = clipped_lines.geometry.length / 1000
                    for (v, t), length in clipped_lines.groupby(['voltage', 'asset_tag'])['length_km'].sum().items():
                        radius_km_str = f"[REDACTED_BY_SCRIPT]"
                        area_km2 = np.pi * ((radius/1000)**2)
                        state['features'][f"[REDACTED_BY_SCRIPT]"] = length / area_km2

        # --- CAPSTONE SYNTHESIS ---
        intersection_cols = [v for k, v in state['features'].items() if '_connection_path_intersection_count' in k]
        if intersection_cols:
            state['features']['[REDACTED_BY_SCRIPT]'] = sum(intersection_cols)

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)

    return state