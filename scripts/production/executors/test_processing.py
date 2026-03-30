import pandas as pd
import numpy as np
import sys
import os
import h3

# Add path
sys.path.append(r"[REDACTED_BY_SCRIPT]")
import exec_99_decision_dossier as exec99

def test_processing():
    print("[REDACTED_BY_SCRIPT]")
    # Use the path from the module
    df = pd.read_csv(exec99.INPUT_DATA_PATH)
    df = exec99.sanitize_column_names(df)
    
    print("Preprocessing...")
    # Replicate main() logic
    df = exec99.coalesce_features(df, exec99.COALESCE_MAP)
    df = exec99.rename_features_logic(df)
    
    for feat in exec99.MISSING_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan
            
    df = exec99.create_artificial_features(df)
    
    print("Loading models...")
    models = exec99.load_all_models()
    historical_data = exec99.load_historical_data()
    
    print("[REDACTED_BY_SCRIPT]")
    df = exec99.generate_knn_features(df, historical_data['X'], historical_data['y'])
    
    print("[REDACTED_BY_SCRIPT]")
    success = 0
    errors = 0
    
    # Ensure hex_id is string
    if 'hex_id' in df.columns:
        df['hex_id'] = df['hex_id'].astype(str).str.strip()
        
    # Sample 50 random hexes
    sample_df = df.sample(n=50, random_state=42)
    
    for i, row in sample_df.iterrows():
        hex_id = row['hex_id']
        hex_df = df.iloc[[i]].copy()
        
        # H3 coords
        try:
            if hasattr(h3, 'cell_to_latlng'):
                lat, lon = h3.cell_to_latlng(hex_id)
            else:
                lat, lon = h3.h3_to_geo(hex_id)
            hex_df['lat'] = lat
            hex_df['lon'] = lon
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")
            continue
            
        try:
            dossier = exec99.forge_decision_dossier(
                hex_df=hex_df,
                models=models,
                historical_data=historical_data,
                capacities=exec99.CAPACITIES_TO_SIMULATE
            )
            success += 1
            if success % 50 == 0:
                print(f"Success count: {success}")
        except Exception as e:
            errors += 1
            print(f"[REDACTED_BY_SCRIPT]")
            # Print detailed traceback for the first error
            if errors == 1:
                import traceback
                traceback.print_exc()
            
            if errors >= 10:
                print("[REDACTED_BY_SCRIPT]")
                break
                
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    test_processing()
