import logging
import geopandas as gpd
import pandas as pd
from pathlib import Path

# --- Configuration ---
PROJECT_CRS = "EPSG:27700"
NON_PRIORITY_CODES = ['FHEAT', 'GMOOR', 'GQSIG', 'NMHAB']
HAB_GROUP_MAP = {
    # Woodland
    'DWOOD': 'Woodland', 'UPOWD': 'Woodland', 'LBYWD': 'Woodland', 'UMAWD': 'Woodland',
    'WETWD': 'Woodland', 'LMDWD': 'Woodland', 'UPBWD': 'Woodland', 'ASNWD': 'Woodland', 'PAWDS': 'Woodland',
    # Grassland
    'CALAM': 'Grassland', 'LCGRA': 'Grassland', 'LDAGR': 'Grassland', 'UCGRA': 'Grassland',
    'UHMEA': 'Grassland', 'WAXCP': 'Grassland', 'PMGRP': 'Grassland', 'GQSIG': 'Grassland',
    # Heathland
    'LHEAT': 'Heathland', 'UHEAT': 'Heathland', 'MHWSC': 'Heathland', 'FHEAT': 'Heathland',
    'DRYHL': 'Heathland', 'WETHL': 'Heathland',
    # Wetland/Bog
    'BLBOG': 'Wetland', 'LFENS': 'Wetland', 'LRBOG': 'Wetland', 'UFFSW': 'Wetland', 'RBEDS': 'Wetland',
    # Coastal/Marine
    'CFPGM': 'Coastal', 'SALTM': 'Coastal', 'CSDUN': 'Coastal', 'CVSHI': 'Coastal',
    'MCSLP': 'Coastal', 'MUDFL': 'Coastal', 'SLAGO': 'Coastal',
    # Other
    'LPAVE': 'Other', 'TORCH': 'Other', 'LAKES': 'Other', 'PONDS': 'Other'
}
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')

def main():
    """
    Orchestrates the creation of the PHI L1 and L2 artifacts per Directive 043.
    """
    logging.info("[REDACTED_BY_SCRIPT]")

    # Define paths
    raw_data_path = Path(r"[REDACTED_BY_SCRIPT]")
    l1_output_path = Path(r"[REDACTED_BY_SCRIPT]")
    l2_output_path = Path(r"[REDACTED_BY_SCRIPT]")

    # --- Stage 1: Create Sanitized L1 Artifact ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf = gpd.read_file(raw_data_path)

    # MANDATE 1: CRS Verification.
    if str(gdf.crs) != PROJECT_CRS:
        logging.error(f"[REDACTED_BY_SCRIPT]'{PROJECT_CRS}', found '{gdf.crs}'[REDACTED_BY_SCRIPT]")
        raise ValueError("[REDACTED_BY_SCRIPT]")
    logging.info(f"[REDACTED_BY_SCRIPT]")

    # MANDATE 2: Schema Normalization.
    gdf.columns = [col.lower().strip() for col in gdf.columns]
    
    # MANDATE 3: Geometric Validation.
    invalid_geom_count = len(gdf) - gdf.geometry.is_valid.sum()
    if invalid_geom_count > 0:
        logging.warning(f"[REDACTED_BY_SCRIPT]")
        gdf['geometry'] = gdf.geometry.buffer(0)
    
    # MANDATE 4: Performance Optimization.
    logging.info("[REDACTED_BY_SCRIPT]")
    gdf.sindex

    # MANDATE 5: Persist L1 Artifact.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    gdf.to_file(l1_output_path, driver='GPKG', layer='phi_england_l1')
    logging.info("--- Stage 1 Complete ---")

    # --- Stage 2: Create Enriched L2 Artifact ---
    logging.info(f"[REDACTED_BY_SCRIPT]")
    
    # MANDATE 6: Create `is_priority` binary feature.
    gdf['is_priority'] = (~gdf['habcodes'].isin(NON_PRIORITY_CODES)).astype(int)
    logging.info(f"Created 'is_priority'[REDACTED_BY_SCRIPT]'is_priority'[REDACTED_BY_SCRIPT]")

    # MANDATE 7: Create `hab_group` categorical feature.
    # Use `habcodes` first, then check `addhabs` as a fallback for non-priority main habitats.
    gdf['hab_group'] = gdf['habcodes'].map(HAB_GROUP_MAP)
    fallback_mask = gdf['hab_group'].isna()
    gdf.loc[fallback_mask, 'hab_group'] = gdf.loc[fallback_mask, 'addhabs'].map(HAB_GROUP_MAP)
    gdf['hab_group'].fillna('Unknown', inplace=True)
    logging.info("Created 'hab_group' feature.")

    # Distill to only necessary columns for the L2 artifact
    l2_gdf = gdf[['is_priority', 'hab_group', 'geometry']].copy()

    # MANDATE 8: Persist L2 Artifact.
    logging.info(f"[REDACTED_BY_SCRIPT]")
    l2_gdf.to_file(l2_output_path, driver='GPKG', layer='[REDACTED_BY_SCRIPT]')
    logging.info("--- Stage 2 Complete ---")
    logging.info("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()