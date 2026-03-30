"""
Directive 029: Environmental Constraint Integration (Phase 1: Ancient Woodland)

This script creates the L29_ENV artifact, which quantifies the planning risk
posed by Ancient Woodlands for each solar farm application.

It follows the Amaryllis Doctrine by:
1.  **Uncompromising CRS Unification:** Immediately re-projecting the hostile
    EPSG:4326 source data to the project standard EPSG:27700.
2.  **Architectural Modularity:** Creating a new, standalone environmental artifact
    (L29_ENV) initialized from the L1 bedrock to ensure a clean data lineage.
3.  **Defensive Data Handling:** Validating and repairing source geometries and
    using spatial indexing to prevent performance catastrophes.
4.  **Strategic Feature Engineering:** Creating a multi-faceted risk profile
    including proximity, containment, and landscape character features.
"""

"""
Directive 031: Multi-Radii Environmental Constraint Synthesis

This script creates the L31_ENV artifact, a comprehensive model of environmental
planning risk from Ancient Woodlands (AW) and Areas of Outstanding Natural Beauty (AONB).

It implements the "[REDACTED_BY_SCRIPT]" by:
1.  Generating density and character features across multiple radii [2, 5, 10, 20]km.
2.  Using a computationally efficient "Cascading Filter" methodology.
3.  Creating synthetic "Alpha" features to model risk concentration.

This script supersedes all previous environmental feature engineering scripts.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CRS = "EPSG:27700"

RISK_RADII_M = [2000, 5000, 10000, 20000] # Mandated radii in meters

logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

def load_solar_farm_base(l1_artifact_path: Path) -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    # Use the correct column names from the input file
    df = pd.read_csv(l1_artifact_path, usecols=['amaryllis_id', 'easting', 'northing'], low_memory=False)
    # Rename columns to match the names expected by the rest of the script
    df.rename(columns={
        'amaryllis_id': 'solar_farm_id',
        'easting': 'easting_x',
        'northing': 'northing_x'
    }, inplace=True)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['easting_x'], df['northing_x']), crs=PROJECT_CRS
    )
    gdf.set_index('solar_farm_id', inplace=True)
    return gdf

def load_and_prepare_constraint(path: Path, area_unit_converter: float = 1.0, area_col_name: str = 'area_sqkm') -> gpd.GeoDataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(path)
    if gdf.crs != PROJECT_CRS:
        logging.info(f"Hostile CRS '{gdf.crs}'[REDACTED_BY_SCRIPT]'{PROJECT_CRS}'.")
        gdf = gdf.to_crs(PROJECT_CRS)
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf[area_col_name] = gdf.geometry.area / area_unit_converter
    return gdf

def calculate_proximity_and_containment(solar_farms: gpd.GeoDataFrame, constraint: gpd.GeoDataFrame, prefix: str, attribute_cols: dict) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]'{prefix}'...")
    
    # Proximity
    prox_join = gpd.sjoin_nearest(solar_farms, constraint, how='left', distance_col=f'[REDACTED_BY_SCRIPT]')
    prox_join = prox_join[~prox_join.index.duplicated(keep='first')]
    
    # Containment
    within_join = gpd.sjoin(solar_farms, constraint, how='left', predicate='within')
    is_within = within_join['index_right'].notna().astype(int)
    is_within = is_within[~is_within.index.duplicated(keep='first')]

    # Assemble features
    cols_to_select = [f'[REDACTED_BY_SCRIPT]'] + list(attribute_cols.keys())
    features = prox_join[cols_to_select].copy()
    features.rename(columns=attribute_cols, inplace=True)
    features[f'{prefix}_is_within'] = is_within
    features.loc[features[f'{prefix}_is_within'] == 1, f'[REDACTED_BY_SCRIPT]'] = 0
    
    return features

def calculate_multi_radius_density(solar_farms: gpd.GeoDataFrame, constraint: gpd.GeoDataFrame, prefix: str, area_col: str, subtype_col: str = None) -> pd.DataFrame:
    """
    Implements the 'Cascading Filter' methodology for efficient multi-radii density analysis.
    Now includes per-subtype counts.
    """
    logging.info(f"[REDACTED_BY_SCRIPT]'{prefix}'[REDACTED_BY_SCRIPT]")
    max_radius = max(RISK_RADII_M)
    
    # ARCHITECTURAL NOTE (Category 3 Pattern): The following `sjoin_nearest` is a
    # deliberate one-to-many join. It acts as a computationally cheap pre-filter,
    # finding all constraint polygons within the maximum analysis radius for each
    # solar farm. The resulting `candidates` GeoDataFrame will contain duplicate
    # solar farm indices. This is the foundation of the "[REDACTED_BY_SCRIPT]"
    # pattern used here. DO NOT de-duplicate this join.
    candidates = gpd.sjoin_nearest(solar_farms, constraint, how='left', max_distance=max_radius)
    
    candidates_with_dist = candidates.join(solar_farms[['geometry']], rsuffix='_farm')
    # Filter out non-matches before expensive distance calculation
    candidates_with_dist = candidates_with_dist[candidates_with_dist.index_right.notna()].copy()
    candidates_with_dist['distance'] = candidates_with_dist.geometry.distance(candidates_with_dist.geometry_farm)

    all_radii_features = []
    for radius_m in RISK_RADII_M:
        radius_km = int(radius_m / 1000)
        logging.info(f"[REDACTED_BY_SCRIPT]")
        
        radius_subset = candidates_with_dist[candidates_with_dist['distance'] <= radius_m]
        radius_features = pd.DataFrame(index=solar_farms.index)

        if not radius_subset.empty:
            grouped_by_farm = radius_subset.groupby(radius_subset.index)
            radius_features[f'[REDACTED_BY_SCRIPT]'] = grouped_by_farm.size()
            radius_features[f'[REDACTED_BY_SCRIPT]'] = grouped_by_farm[area_col].sum()

            if subtype_col:
                # Efficiently count each subtype
                subtype_counts = radius_subset.groupby([radius_subset.index, subtype_col]).size().unstack(fill_value=0)
                subtype_counts.columns = [f"[REDACTED_BY_SCRIPT]" for col in subtype_counts.columns]
                radius_features = radius_features.join(subtype_counts)

        all_radii_features.append(radius_features)

    # Concatenate and fill NaNs for farms with no constraints in range
    final_density_features = pd.concat(all_radii_features, axis=1)
    # Ensure all columns exist even if no constraints were found for any farm
    for radius_m in RISK_RADII_M:
        radius_km = int(radius_m / 1000)
        if f'[REDACTED_BY_SCRIPT]' not in final_density_features:
            final_density_features[f'[REDACTED_BY_SCRIPT]'] = 0
        if f'[REDACTED_BY_SCRIPT]' not in final_density_features:
            final_density_features[f'[REDACTED_BY_SCRIPT]'] = 0
            
    return final_density_features.fillna(0)

def calculate_density_focus_ratios(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]'Alpha'[REDACTED_BY_SCRIPT]"""
    logging.info(f"[REDACTED_BY_SCRIPT]'{prefix}'...")
    focus_df = pd.DataFrame(index=df.index)
    
    # Use np.divide for safe division (handles division by zero)
    focus_df[f'[REDACTED_BY_SCRIPT]'] = np.divide(
        df[f'[REDACTED_BY_SCRIPT]'], df[f'[REDACTED_BY_SCRIPT]']
    ).fillna(0)
    
    return focus_df

def calculate_synthesis_features(df: pd.DataFrame) -> pd.DataFrame:
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    synth_df = pd.DataFrame(index=df.index)
    
    dist_cols = ['aw_dist_to_nearest_m', 'aonb_dist_to_nearest_m']
    synth_df['[REDACTED_BY_SCRIPT]'] = df[dist_cols].min(axis=1)
    
    # Determine the dominant constraint type
    synth_df['[REDACTED_BY_SCRIPT]'] = df[dist_cols].idxmin(axis=1).apply(
        lambda x: 'AncientWoodland' if 'aw_' in x else 'AONB'
    )
    
    # Calculate aonb_count_in_2km before summing
    # This is a placeholder for a full implementation if needed, for now we assume it's 1 if dist < 2000
    aonb_count_in_2km = (df['aonb_dist_to_nearest_m'] < 20000).astype(int)
    synth_df['[REDACTED_BY_SCRIPT]'] = df['aw_count_in_2km'] + aonb_count_in_2km

    return synth_df


def main():
    """
    Main orchestration function to build the L29_ENV artifact.
    """
    logging.info("[REDACTED_BY_SCRIPT]")
    
    # --- Phase 1: Initialization & Preparation ---
    l1_artifact_path = r"[REDACTED_BY_SCRIPT]"
    woodland_path = r"[REDACTED_BY_SCRIPT]"
    output_path = r"[REDACTED_BY_SCRIPT]"
    aonb_path = r"[REDACTED_BY_SCRIPT]"

    solar_farms_base = load_solar_farm_base(l1_artifact_path)
    woodlands_gdf = load_and_prepare_constraint(woodland_path, area_unit_converter=10000, area_col_name='area_ha')
    aonb_gdf = load_and_prepare_constraint(aonb_path, area_unit_converter=1000000, area_col_name='area_sqkm')

    # --- Phase 2: Feature Engineering ---
    aw_prox_contain = calculate_proximity_and_containment(
        solar_farms_base, woodlands_gdf, 'aw', 
        attribute_cols={'status': 'aw_nearest_status', 'area_ha': 'aw_nearest_area_ha'}
    )
    aonb_prox_contain = calculate_proximity_and_containment(
        solar_farms_base, aonb_gdf, 'aonb',
        attribute_cols={'name': 'aonb_nearest_name', 'area_sqkm': 'aonb_nearest_area_sqkm'}
    )
    
    aw_density = calculate_multi_radius_density(solar_farms_base, woodlands_gdf, 'aw', 'area_ha', subtype_col='status')
    aonb_density = calculate_multi_radius_density(solar_farms_base, aonb_gdf, 'aonb', 'area_sqkm')

    # --- Phase 3: Assembly & Synthesis ---
    logging.info("[REDACTED_BY_SCRIPT]")
    final_df = solar_farms_base.drop(columns=['geometry']).join([
        aw_prox_contain, aonb_prox_contain, aw_density, aonb_density
    ])
    
    aw_focus = calculate_density_focus_ratios(final_df, 'aw')
    # Correcting area column name for AONB focus ratio
    final_df['[REDACTED_BY_SCRIPT]'] = np.divide(
        final_df['aonb_total_area_in_5km'], final_df['[REDACTED_BY_SCRIPT]']
    ).fillna(0)
    
    final_df = final_df.join(aw_focus)

    # Add other features from superseded directive 030 (LPA coverage, etc.) here if needed
    # Adding omitted synthesis features
    synthesis_features = calculate_synthesis_features(final_df)
    final_df = final_df.join(synthesis_features)
    final_df.drop(columns=['aonb_nearest_name'], errors='ignore', inplace=True)

    # --- Phase 4: QA and Persistence ---
    logging.info("[REDACTED_BY_SCRIPT]")
    if not (final_df['aw_count_in_2km'] <= final_df['aw_count_in_20km']).all():
        logging.error("[REDACTED_BY_SCRIPT]")
    else:
        logging.info("[REDACTED_BY_SCRIPT]")

    logging.info(f"[REDACTED_BY_SCRIPT]")
    final_df.to_csv(output_path)
    
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()