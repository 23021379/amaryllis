import pandas as pd
import re
import numpy as np

def parse_lpa_stats(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    lpa_sections = re.split(r'={10,}\nLPA: ', content)
    
    stats_db = {}

    for section in lpa_sections:
        if not section.strip():
            continue
        
        lines = section.split('\n')
        lpa_name = lines[0].strip().replace('=', '').strip()
        
        stats_db[lpa_name] = {}

        for line in lines[1:]:
            if '|' in line and 'Mean=' in line:
                parts = [p.strip() for p in line.split('|')]
                bin_name = parts[0]
                
                # Parse metrics
                metrics = {}
                for part in parts[1:]:
                    if '=' in part:
                        key, val = part.split('=')
                        key = key.strip()
                        val = val.strip()
                        try:
                            metrics[key] = float(val)
                        except ValueError:
                            metrics[key] = np.nan
                
                stats_db[lpa_name][bin_name] = metrics

    return stats_db

def get_capacity_bin(capacity):
    if pd.isna(capacity):
        return None
    if capacity < 5:
        return '0-5 MW'
    elif capacity < 15:
        return '5-15 MW'
    elif capacity < 30:
        return '15-30 MW'
    elif capacity < 50:
        return '30-50 MW'
    else:
        return '50+ MW'

def main():
    # 1. Parse Stats
    print("Parsing LPA stats...")
    stats_db = parse_lpa_stats('lpa_stats.txt')
    print(f"[REDACTED_BY_SCRIPT]")

    # 2. Load Data
    print("Loading datasets...")
    model_ready_path = r'[REDACTED_BY_SCRIPT]'
    l2_path = r'[REDACTED_BY_SCRIPT]'
    
    df_model = pd.read_csv(model_ready_path)
    
    # Check if planning_authority is already in df_model
    if 'planning_authority' in df_model.columns:
        print("[REDACTED_BY_SCRIPT]")
        df_merged = df_model
    else:
        print("[REDACTED_BY_SCRIPT]")
        df_l2 = pd.read_csv(l2_path)
        
        # Create temporary merge keys
        df_model['merge_easting'] = df_model['easting_x']
        df_model['merge_northing'] = df_model['northing_x']
        df_model['merge_tech'] = df_model['technology_type']
        
        df_l2['merge_easting'] = df_l2['easting']
        df_l2['merge_northing'] = df_l2['northing']
        df_l2['merge_tech'] = df_l2['technology_type']

        df_l2_dedup = df_l2.drop_duplicates(subset=['merge_easting', 'merge_northing', 'merge_tech'])
        
        df_merged = pd.merge(
            df_model, 
            df_l2_dedup[['merge_easting', 'merge_northing', 'merge_tech', 'planning_authority']],
            on=['merge_easting', 'merge_northing', 'merge_tech'],
            how='left'
        )
        
        # Clean up temporary columns
        df_merged.drop(columns=['merge_easting', 'merge_northing', 'merge_tech'], inplace=True)

    print(f"[REDACTED_BY_SCRIPT]")
    
    # Check match rate
    if 'planning_authority' in df_merged.columns:
        matched_count = df_merged['planning_authority'].notna().sum()
        print(f"[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
        return

    # 4. Enrich with Stats
    print("[REDACTED_BY_SCRIPT]")
    
    # Prepare new columns
    new_cols = {
        'lpa_planning_time_overall_n': [],
        'lpa_planning_time_overall_mean': [],
        '[REDACTED_BY_SCRIPT]': [],
        '[REDACTED_BY_SCRIPT]': [],
        'lpa_planning_time_overall_min': [],
        '[REDACTED_BY_SCRIPT]': [],
        
        'lpa_planning_time_bin_n': [],
        'lpa_planning_time_bin_mean': [],
        'lpa_planning_time_bin_std': [],
        'lpa_planning_time_bin_median': [],
        'lpa_planning_time_bin_min': [],
        'lpa_planning_time_bin_max': []
    }

    for index, row in df_merged.iterrows():
        lpa = row['planning_authority']
        capacity = row['[REDACTED_BY_SCRIPT]']
        
        # Initialize with NaN
        overall_stats = {}
        bin_stats = {}
        
        if pd.notna(lpa) and lpa in stats_db:
            lpa_data = stats_db[lpa]
            
            # Get Overall Stats
            if 'OVERALL' in lpa_data:
                overall_stats = lpa_data['OVERALL']
            
            # Get Bin Stats
            bin_name = get_capacity_bin(capacity)
            if bin_name and bin_name in lpa_data:
                bin_stats = lpa_data[bin_name]
        
        # Append to lists
        new_cols['lpa_planning_time_overall_n'].append(overall_stats.get('N', np.nan))
        new_cols['lpa_planning_time_overall_mean'].append(overall_stats.get('Mean', np.nan))
        new_cols['[REDACTED_BY_SCRIPT]'].append(overall_stats.get('Std', np.nan))
        new_cols['[REDACTED_BY_SCRIPT]'].append(overall_stats.get('Median', np.nan))
        new_cols['lpa_planning_time_overall_min'].append(overall_stats.get('Min', np.nan))
        new_cols['[REDACTED_BY_SCRIPT]'].append(overall_stats.get('Max', np.nan))
        
        new_cols['lpa_planning_time_bin_n'].append(bin_stats.get('N', np.nan))
        new_cols['lpa_planning_time_bin_mean'].append(bin_stats.get('Mean', np.nan))
        new_cols['lpa_planning_time_bin_std'].append(bin_stats.get('Std', np.nan))
        new_cols['lpa_planning_time_bin_median'].append(bin_stats.get('Median', np.nan))
        new_cols['lpa_planning_time_bin_min'].append(bin_stats.get('Min', np.nan))
        new_cols['lpa_planning_time_bin_max'].append(bin_stats.get('Max', np.nan))

    # Add columns to DataFrame
    for col_name, data in new_cols.items():
        df_merged[col_name] = data
    df_merged.drop(columns=['planning_authority'], inplace=True)

    # Save
    output_path = r'[REDACTED_BY_SCRIPT]'
    df_merged.to_csv(output_path, index=False)
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    main()
