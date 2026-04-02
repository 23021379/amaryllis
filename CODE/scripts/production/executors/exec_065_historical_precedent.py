import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import re
import os
from sklearn.neighbors import KDTree
from shapely.geometry import Point

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[REDACTED_BY_SCRIPT]')
PROJECT_CRS = "EPSG:27700"
K_NEIGHBORS = 15
MAX_DISTANCE_KM = 999.0
NULL_SENTINEL = -1.0
SOLAR_TECH_ID = 21  # From categorical_mappings.json

# Paths
LEGACY_SOLAR_PATH = r"[REDACTED_BY_SCRIPT]"
PS1_PATH = r'[REDACTED_BY_SCRIPT]'
PS2_PATH = r'[REDACTED_BY_SCRIPT]'
CPS1_PATH = r"[REDACTED_BY_SCRIPT]"
CPS2_PATH = r'[REDACTED_BY_SCRIPT]'
LPA_BOUNDARIES_PATH = r"[REDACTED_BY_SCRIPT]"

# --- HELPER FUNCTIONS: LPA CLEANING & MATCHING ---

def clean_lpa_name(series: pd.Series) -> pd.Series:
    """[REDACTED_BY_SCRIPT]"""
    return series.astype(str).str.lower().str.strip().str.replace(r'\s*\(.*\)\s*', '', regex=True).str.strip()

def find_column_by_pattern(df: pd.DataFrame, pattern: str) -> str:
    """[REDACTED_BY_SCRIPT]"""
    for col in df.columns:
        normalized_col = re.sub(r'[^a-z0-9]+', '_', col.lower()).strip('_')
        if re.search(pattern, normalized_col):
            return col
    return None

def safe_division(numerator, denominator):
    """[REDACTED_BY_SCRIPT]"""
    return np.divide(numerator.values, denominator.values, out=np.full_like(numerator.values, np.nan, dtype=float), where=(denominator.values!=0))

def calculate_trend(group):
    """[REDACTED_BY_SCRIPT]"""
    if len(group) < 3:
        return np.nan
    # Drop NaNs
    group = group.dropna(subset=['year', 'workload'])
    if len(group) < 3:
        return np.nan
    
    x = group['year'].values
    y = group['workload'].values
    # Simple linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope
    except:
        return np.nan

# --- MODULE 1: LPA GOVERNMENT DATA PROCESSING ---

def process_ps1_cps1(ps1_path, cps1_path):
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    try:
        dfs = []
        # 1. Process PS1
        if os.path.exists(ps1_path):
            df_ps1 = pd.read_csv(ps1_path, encoding='latin-1', dtype=str)
            df_ps1.columns = df_ps1.columns.str.lower().str.replace(' ', '_', regex=False)
            df_ps1.replace('..', np.nan, inplace=True)
            
            lpa_col = find_column_by_pattern(df_ps1, r'^lpanm$')
            rec_col = find_column_by_pattern(df_ps1, r'[REDACTED_BY_SCRIPT]')
            wd_col = find_column_by_pattern(df_ps1, r'[REDACTED_BY_SCRIPT]')
            year_col = find_column_by_pattern(df_ps1, r'^f_year$')
            
            if lpa_col and rec_col and wd_col and year_col:
                df_ps1['lpanm_clean'] = clean_lpa_name(df_ps1[lpa_col])
                df_ps1['received'] = pd.to_numeric(df_ps1[rec_col], errors='coerce').fillna(0)
                df_ps1['withdrawn'] = pd.to_numeric(df_ps1[wd_col], errors='coerce').fillna(0)
                df_ps1['year'] = pd.to_numeric(df_ps1[year_col], errors='coerce') # Assuming year is numeric-ish
                dfs.append(df_ps1[['lpanm_clean', 'year', 'received', 'withdrawn']])

        # 2. Process CPS1
        if os.path.exists(cps1_path):
            df_cps1 = pd.read_csv(cps1_path, encoding='latin-1', dtype=str)
            df_cps1.columns = df_cps1.columns.str.lower().str.replace(' ', '_', regex=False)
            df_cps1.replace('..', np.nan, inplace=True)
            
            lpa_col = find_column_by_pattern(df_cps1, r'^lpanm$')
            rec_col = find_column_by_pattern(df_cps1, r'[REDACTED_BY_SCRIPT]')
            wd_col = find_column_by_pattern(df_cps1, r'[REDACTED_BY_SCRIPT]')
            quarter_col = find_column_by_pattern(df_cps1, r'^quarter$')

            if lpa_col and rec_col and wd_col and quarter_col:
                df_cps1['lpanm_clean'] = clean_lpa_name(df_cps1[lpa_col])
                df_cps1['received'] = pd.to_numeric(df_cps1[rec_col], errors='coerce').fillna(0)
                df_cps1['withdrawn'] = pd.to_numeric(df_cps1[wd_col], errors='coerce').fillna(0)
                # Derive year from quarter string (e.g. "2022 Q1" -> 2022)
                df_cps1['year'] = pd.to_numeric(df_cps1[quarter_col].astype(str).str.split(' ').str[0], errors='coerce')
                dfs.append(df_cps1[['lpanm_clean', 'year', 'received', 'withdrawn']])
        
        if not dfs:
            return pd.DataFrame()

        full_df = pd.concat(dfs, ignore_index=True)
        
        # --- Feature 1: Withdrawal Rate (Aggregated) ---
        agg_base = full_df.groupby('lpanm_clean').agg({'received': 'sum', 'withdrawn': 'sum'})
        agg_base['lpa_withdrawal_rate'] = safe_division(agg_base['withdrawn'], agg_base['received'])
        
        # --- Feature 2: Workload Dynamics (Yearly) ---
        yearly_workload = full_df.groupby(['lpanm_clean', 'year'])['received'].sum().reset_index()
        yearly_workload.rename(columns={'received': 'workload'}, inplace=True)
        
        agg_dynamics = yearly_workload.groupby('lpanm_clean')['workload'].agg(['mean', 'sum'])
        agg_dynamics.rename(columns={'mean': '[REDACTED_BY_SCRIPT]', 'sum': 'lpa_total_experience'}, inplace=True)
        
        # --- Feature 3: Workload Trend ---
        trend_series = yearly_workload.groupby('lpanm_clean').apply(calculate_trend).rename('lpa_workload_trend')
        
        # Merge all
        final_agg = agg_base[['lpa_withdrawal_rate']].join([agg_dynamics, trend_series], how='outer')
        
        return final_agg
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return pd.DataFrame()

def process_ps2(ps2_path):
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    if not os.path.exists(ps2_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(ps2_path, encoding='latin-1', dtype=str)
        df.replace('..', np.nan, inplace=True)
        
        lpa_col = find_column_by_pattern(df, r'^lpanm$')
        major_in_time_col = find_column_by_pattern(df, r'[REDACTED_BY_SCRIPT]')
        total_major_col = find_column_by_pattern(df, r'[REDACTED_BY_SCRIPT]')
        
        if not (lpa_col and major_in_time_col and total_major_col):
            return pd.DataFrame()
            
        df['lpanm_clean'] = clean_lpa_name(df[lpa_col])
        df['in_time'] = pd.to_numeric(df[major_in_time_col], errors='coerce').fillna(0)
        df['total'] = pd.to_numeric(df[total_major_col], errors='coerce').fillna(0)
        
        agg = df.groupby('lpanm_clean').agg({'in_time': 'sum', 'total': 'sum'})
        agg['[REDACTED_BY_SCRIPT]'] = safe_division(agg['in_time'], agg['total'])
        
        # --- Feature: Major Commercial Approval Rate ---
        # We need specific columns for this
        major_comm_granted_cols = [c for c in df.columns if re.search(r'[REDACTED_BY_SCRIPT]', c)]
        major_comm_dec_cols = [c for c in df.columns if re.search(r'[REDACTED_BY_SCRIPT]', c)]
        
        if major_comm_granted_cols and major_comm_dec_cols:
            df['comm_granted'] = df[major_comm_granted_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            df['comm_decisions'] = df[major_comm_dec_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
            
            comm_agg = df.groupby('lpanm_clean').agg({'comm_granted': 'sum', 'comm_decisions': 'sum'})
            agg['lpa_major_commercial_approval_rate'] = safe_division(comm_agg['comm_granted'], comm_agg['comm_decisions'])
        else:
            agg['lpa_major_commercial_approval_rate'] = np.nan

        return agg[['[REDACTED_BY_SCRIPT]', 'lpa_major_commercial_approval_rate']]
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return pd.DataFrame()

def process_cps2(cps2_path):
    """[REDACTED_BY_SCRIPT]"""
    logging.info("[REDACTED_BY_SCRIPT]")
    if not os.path.exists(cps2_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(cps2_path, encoding='latin-1', dtype=str)
        df.replace('..', np.nan, inplace=True)
        
        lpa_col = find_column_by_pattern(df, r'^lpanm$')
        granted_col = find_column_by_pattern(df, r'[REDACTED_BY_SCRIPT]')
        rec_col = find_column_by_pattern(df, r'^date_received$')
        disp_col = find_column_by_pattern(df, r'^date_dispatched$')
        scheme_col = find_column_by_pattern(df, r'^type_of_scheme$')
        
        if not (lpa_col and granted_col and rec_col and disp_col):
            return pd.DataFrame()

        df['lpanm_clean'] = clean_lpa_name(df[lpa_col])
        df['is_granted'] = df[granted_col].map({'Granted': 1, 'Refused': 0})
        df['rec_date'] = pd.to_datetime(df[rec_col], errors='coerce', dayfirst=True)
        df['disp_date'] = pd.to_datetime(df[disp_col], errors='coerce', dayfirst=True)
        df['decision_days'] = (df['disp_date'] - df['rec_date']).dt.days
        
        # Filter valid rows
        df = df[(df['decision_days'] >= 0) & (df['is_granted'].notna())]
        
        # General Metrics
        agg = df.groupby('lpanm_clean').agg(
            lpa_approval_rate_cps2=('is_granted', 'mean'),
            lpa_avg_decision_days=('decision_days', 'mean'),
            lpa_decision_speed_variance=('decision_days', 'std'),
            lpa_p90_decision_days=('decision_days', lambda x: x.quantile(0.9))
        )
        
        # --- Feature: Grant vs Refuse Speed Ratio ---
        speed_by_outcome = df.groupby(['lpanm_clean', 'is_granted'])['decision_days'].mean().unstack()
        # Columns will be 0.0 (Refused) and 1.0 (Granted) if present
        if 0 in speed_by_outcome.columns and 1 in speed_by_outcome.columns:
            agg['[REDACTED_BY_SCRIPT]'] = speed_by_outcome[1] / speed_by_outcome[0]
        else:
            agg['[REDACTED_BY_SCRIPT]'] = np.nan
        
        # Industrial Specific
        if scheme_col:
            ind_df = df[df[scheme_col].str.contains('[REDACTED_BY_SCRIPT]', case=False, na=False)]
            if not ind_df.empty:
                ind_agg = ind_df.groupby('lpanm_clean')['is_granted'].mean().rename('[REDACTED_BY_SCRIPT]')
                agg = agg.join(ind_agg)
        
        return agg
    except Exception as e:
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return pd.DataFrame()

# --- MODULE 2: LEGACY SOLAR & KNN PROCESSING ---

def process_legacy_spatial(master_gdf, legacy_path):
    """[REDACTED_BY_SCRIPT]"""
    if not os.path.exists(legacy_path):
        logging.error(f"[REDACTED_BY_SCRIPT]")
        return master_gdf, None

    # Load Legacy Data
    df_legacy = pd.read_csv(legacy_path, encoding='latin-1')
    
    # 1. Parse Geometry (WKT or Easting/Northing)
    if 'geometry' in df_legacy.columns and df_legacy['geometry'].dtype == 'object':
        try:
            geoms = gpd.GeoSeries.from_wkt(df_legacy['geometry'])
            gdf_legacy = gpd.GeoDataFrame(df_legacy, geometry=geoms, crs=PROJECT_CRS)
        except:
            # Fallback check for coordinate columns
            if 'easting' in df_legacy.columns and 'northing' in df_legacy.columns:
                gdf_legacy = gpd.GeoDataFrame(df_legacy, geometry=gpd.points_from_xy(df_legacy['easting'], df_legacy['northing']), crs=PROJECT_CRS)
            elif 'x' in df_legacy.columns and 'y' in df_legacy.columns:
                gdf_legacy = gpd.GeoDataFrame(df_legacy, geometry=gpd.points_from_xy(df_legacy['x'], df_legacy['y']), crs=PROJECT_CRS)
            else:
                logging.error(f"[REDACTED_BY_SCRIPT]")
                return master_gdf, None
    else:
        # Direct check for coordinate columns
        if 'easting' in df_legacy.columns and 'northing' in df_legacy.columns:
            gdf_legacy = gpd.GeoDataFrame(df_legacy, geometry=gpd.points_from_xy(df_legacy['easting'], df_legacy['northing']), crs=PROJECT_CRS)
        elif 'x' in df_legacy.columns and 'y' in df_legacy.columns:
            gdf_legacy = gpd.GeoDataFrame(df_legacy, geometry=gpd.points_from_xy(df_legacy['x'], df_legacy['y']), crs=PROJECT_CRS)
        else:
            logging.error(f"[REDACTED_BY_SCRIPT]")
            return master_gdf, None
    
    # --- FIX: Filter Invalid Geometries ---
    # Remove points at (0,0) or clearly outside UK bounds (e.g. < 1000m)
    initial_len = len(gdf_legacy)
    gdf_legacy = gdf_legacy[
        (gdf_legacy.geometry.x > 1000) & 
        (gdf_legacy.geometry.y > 1000)
    ].copy()
    if len(gdf_legacy) < initial_len:
        logging.warning(f"[REDACTED_BY_SCRIPT]")

    # 2. Clean Legacy Attributes
    # Schema alignment: Use 'permission_granted' (0/1) and 'planning_authority' directly
    
    # Outcome: Ensure it's numeric (0 or 1)
    if 'permission_granted' in gdf_legacy.columns:
        gdf_legacy['outcome_binary'] = pd.to_numeric(gdf_legacy['permission_granted'], errors='coerce').fillna(0)
    else:
        logging.warning("Column 'permission_granted'[REDACTED_BY_SCRIPT]'outcome' search.")
        gdf_legacy['outcome_binary'] = 0 # Default sentinel
    
    # Duration: Ensure numeric
    if '[REDACTED_BY_SCRIPT]' in gdf_legacy.columns:
        gdf_legacy['duration_clean'] = pd.to_numeric(gdf_legacy['[REDACTED_BY_SCRIPT]'], errors='coerce')
    else:
        gdf_legacy['duration_clean'] = np.nan

    # LPA Name Cleaning
    target_lpa_col = 'planning_authority' if 'planning_authority' in gdf_legacy.columns else 'lpa_raw'
    
    if target_lpa_col in gdf_legacy.columns:
        gdf_legacy['lpa_clean'] = clean_lpa_name(gdf_legacy[target_lpa_col])
        
        # --- Feature: LPA Legacy Stats ---
        lpa_legacy_stats = gdf_legacy.groupby('lpa_clean').agg(
            lpa_legacy_approval_rate=('outcome_binary', 'mean'),
            lpa_legacy_application_count=('outcome_binary', 'count'),
            lpa_legacy_avg_duration=('duration_clean', 'mean')
        )
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")
        lpa_legacy_stats = pd.DataFrame()

    # --- Feature: Spatial KNN ---
    logging.info("[REDACTED_BY_SCRIPT]")
    
    master_coords = np.vstack([master_gdf.geometry.x, master_gdf.geometry.y]).T
    legacy_coords = np.vstack([gdf_legacy.geometry.x, gdf_legacy.geometry.y]).T
    
    # Split legacy by outcome
    approved_mask = gdf_legacy['outcome_binary'] == 1
    refused_mask = gdf_legacy['outcome_binary'] == 0
    
    # 3. KNN Features (Sklearn)
    knn_features = pd.DataFrame(index=master_gdf.index)
    
    # Tree: ALL Legacy
    if len(legacy_coords) > 0:
        tree_all = KDTree(legacy_coords)
        
        # Nearest Dist + Index
        dists, indices = tree_all.query(master_coords, k=min(K_NEIGHBORS, len(legacy_coords)))
        
        knn_features['[REDACTED_BY_SCRIPT]'] = dists[:, 0] / 1000.0
        knn_features['knn_dist_to_nearest_km'] = dists[:, 0] / 1000.0 # Duplicate as per requirements
        knn_features['knn_std_distance_km'] = dists.std(axis=1) / 1000.0
        
        # --- FIX: Count Solar Neighbors ---
        if 'technology_type' in gdf_legacy.columns:
            neighbor_techs = gdf_legacy['technology_type'].values[indices]
            # Count how many are Solar (ID 21)
            # Ensure technology_type is numeric
            try:
                neighbor_techs = neighbor_techs.astype(int)
                knn_features['knn_count_solar'] = (neighbor_techs == SOLAR_TECH_ID).sum(axis=1)
            except:
                knn_features['knn_count_solar'] = 0
        else:
            knn_features['knn_count_solar'] = 0
        
        # Average Approval of Neighbors
        neighbor_outcomes = gdf_legacy['outcome_binary'].values[indices]
        knn_features['knn_approval_rate'] = neighbor_outcomes.mean(axis=1)
    else:
        for col in ['[REDACTED_BY_SCRIPT]', 'knn_dist_to_nearest_km', 'knn_std_distance_km', 'knn_approval_rate']:
            knn_features[col] = NULL_SENTINEL
        knn_features['knn_count_solar'] = 0

    # Tree: Approved
    if approved_mask.sum() > 0:
        tree_app = KDTree(legacy_coords[approved_mask])
        dists_app, _ = tree_app.query(master_coords, k=1)
        knn_features['[REDACTED_BY_SCRIPT]'] = dists_app.flatten() / 1000.0
    else:
        knn_features['[REDACTED_BY_SCRIPT]'] = MAX_DISTANCE_KM

    # Tree: Refused
    if refused_mask.sum() > 0:
        tree_ref = KDTree(legacy_coords[refused_mask])
        dists_ref, _ = tree_ref.query(master_coords, k=1)
        knn_features['[REDACTED_BY_SCRIPT]'] = dists_ref.flatten() / 1000.0
    else:
        knn_features['[REDACTED_BY_SCRIPT]'] = MAX_DISTANCE_KM

    # 4. Radius Features (Nearby Legacy) - 10km
    logging.info("[REDACTED_BY_SCRIPT]")
    if len(legacy_coords) > 0:
        # query_radius returns array of arrays of indices
        indices_list = tree_all.query_radius(master_coords, r=10000.0) # 10km in meters
        
        nearby_counts = [len(idxs) for idxs in indices_list]
        nearby_approval_rates = []
        
        for idxs in indices_list:
            if len(idxs) > 0:
                outcomes = gdf_legacy['outcome_binary'].values[idxs]
                nearby_approval_rates.append(outcomes.mean())
            else:
                nearby_approval_rates.append(NULL_SENTINEL)
                
        knn_features['nearby_legacy_count'] = nearby_counts
        knn_features['[REDACTED_BY_SCRIPT]'] = nearby_approval_rates
        
        # 5. Advanced KNN Features
        # knn_avg_distance_km
        knn_features['knn_avg_distance_km'] = dists.mean(axis=1) / 1000.0
        
        # knn_inverse_dist_weighted_approval
        # Weights = 1 / (dist + epsilon)
        epsilon = 1e-6
        weights = 1.0 / (dists + epsilon)
        
        # Weighted average of outcomes
        # neighbor_outcomes shape: (n_samples, k)
        weighted_approval = (neighbor_outcomes * weights).sum(axis=1) / weights.sum(axis=1)
        knn_features['[REDACTED_BY_SCRIPT]'] = weighted_approval
        
    else:
        knn_features['nearby_legacy_count'] = 0
        knn_features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL
        knn_features['knn_avg_distance_km'] = NULL_SENTINEL
        knn_features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL

    return lpa_legacy_stats, knn_features

# --- MAIN EXECUTOR ---

def execute(master_gdf):
    logging.info("[REDACTED_BY_SCRIPT]")
    
    id_col = 'hex_id'
    if master_gdf.index.name != id_col:
        master_gdf = master_gdf.set_index(id_col) if id_col in master_gdf.columns else master_gdf

    # 1. Process Government Data (LPA Profiling)
    # Prioritized search for the exact LPA Identifier column.
    # We use anchored regexes to prevent partial matching against features like 'lpa_growth_pct'.
    target_patterns = [
        r'[REDACTED_BY_SCRIPT]',  # Primary target
        r'^local_authority$',     # Secondary
        r'^lpa_name$',            # Tertiary
        r'^lpa$'                  # Fallback
    ]
    
    lpa_target_col = None
    for pattern in target_patterns:
        lpa_target_col = find_column_by_pattern(master_gdf, pattern)
        if lpa_target_col:
            break
            
    if lpa_target_col:
        logging.info(f"[REDACTED_BY_SCRIPT]")
        master_gdf['lpa_join_key'] = clean_lpa_name(master_gdf[lpa_target_col])
    else:
        logging.info("[REDACTED_BY_SCRIPT]")
        if os.path.exists(LPA_BOUNDARIES_PATH):
            try:
                # Load Boundaries
                lpa_gdf = gpd.read_file(LPA_BOUNDARIES_PATH)
                if lpa_gdf.crs.to_string() != PROJECT_CRS:
                    lpa_gdf = lpa_gdf.to_crs(PROJECT_CRS)
                
                # Spatial Join
                # We only need the name column
                lpa_gdf = lpa_gdf[['LPA23NM', 'geometry']]
                
                # Ensure master is a GeoDataFrame
                if not isinstance(master_gdf, gpd.GeoDataFrame):
                     logging.error("[REDACTED_BY_SCRIPT]")
                else:
                    # Use sjoin to find which LPA polygon contains each point
                    # 'inner' join would drop points outside LPAs (e.g. offshore), 'left' keeps them
                    joined = gpd.sjoin(master_gdf, lpa_gdf, how='left', predicate='within')
                    
                    # Extract the LPA name and clean it
                    if 'LPA23NM' in joined.columns:
                        master_gdf['lpa_join_key'] = clean_lpa_name(joined['LPA23NM'])
                        logging.info(f"[REDACTED_BY_SCRIPT]'lpa_join_key'[REDACTED_BY_SCRIPT]")
                    else:
                        logging.warning("[REDACTED_BY_SCRIPT]'LPA23NM' column.")
            except Exception as e:
                logging.error(f"[REDACTED_BY_SCRIPT]")
        else:
            logging.warning(f"[REDACTED_BY_SCRIPT]")

    if 'lpa_join_key' in master_gdf.columns:
        df_withdrawal = process_ps1_cps1(PS1_PATH, CPS1_PATH)
        df_compliance = process_ps2(PS2_PATH)
        df_cps2 = process_cps2(CPS2_PATH)
        
        # Merge LPA stats
        lpa_stats = df_withdrawal.join([df_compliance, df_cps2], how='outer')
        
        master_gdf = master_gdf.join(lpa_stats, on='lpa_join_key', how='left')
    else:
        logging.warning("[REDACTED_BY_SCRIPT]")

    # 2. Process Legacy Data (Stats + Spatial)
    lpa_legacy_stats, knn_features = process_legacy_spatial(master_gdf, LEGACY_SOLAR_PATH)
    
    # Merge Legacy LPA stats
    if 'lpa_join_key' in master_gdf.columns and not lpa_legacy_stats.empty:
        master_gdf = master_gdf.join(lpa_legacy_stats, on='lpa_join_key', how='left')
        
    # Merge KNN Spatial Features
    master_gdf = master_gdf.join(knn_features)
    
    # Cleanup
    if 'lpa_join_key' in master_gdf.columns:
        master_gdf.drop(columns=['lpa_join_key'], inplace=True)

    # Final Imputation for new features
    new_features = [
        'lpa_withdrawal_rate', '[REDACTED_BY_SCRIPT]', 'lpa_approval_rate_cps2',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'lpa_legacy_approval_rate', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        '[REDACTED_BY_SCRIPT]', 'knn_std_distance_km', 
        'knn_dist_to_nearest_km', 'knn_approval_rate', 'knn_count_solar',
        # New Features
        '[REDACTED_BY_SCRIPT]', 'lpa_total_experience', 'lpa_workload_trend',
        'lpa_major_commercial_approval_rate', '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]',
        'nearby_legacy_count', '[REDACTED_BY_SCRIPT]',
        'knn_avg_distance_km', '[REDACTED_BY_SCRIPT]'
    ]
    
    # Check which ones actually exist (some might be missing if files were missing)
    existing_new_cols = [c for c in new_features if c in master_gdf.columns]
    
    # Fill NaNs with Sentinel
    master_gdf[existing_new_cols] = master_gdf[existing_new_cols].fillna(NULL_SENTINEL)
    
    # Defragment
    master_gdf = master_gdf.copy()
    
    logging.info(f"[REDACTED_BY_SCRIPT]")
    return master_gdf