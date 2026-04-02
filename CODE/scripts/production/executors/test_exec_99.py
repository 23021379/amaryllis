import sys
import os
import pandas as pd
import json
import logging

# Add the directory containing the script to sys.path
sys.path.append(r"[REDACTED_BY_SCRIPT]")

import exec_99_decision_dossier

# Setup paths
ORIGINAL_INPUT = r"[REDACTED_BY_SCRIPT]"
TEST_INPUT = r"[REDACTED_BY_SCRIPT]"
TEST_OUTPUT = r"[REDACTED_BY_SCRIPT]"

def setup_test_data():
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        df = pd.read_csv(ORIGINAL_INPUT, nrows=10)
        df.to_csv(TEST_INPUT, index=False)
        print(f"[REDACTED_BY_SCRIPT]")
        return True
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        return False

def verify_output():
    print(f"[REDACTED_BY_SCRIPT]")
    if not os.path.exists(TEST_OUTPUT):
        print("[REDACTED_BY_SCRIPT]")
        return

    try:
        with open(TEST_OUTPUT, 'r') as f:
            data = json.load(f)
        
        if data.get('type') != 'FeatureCollection':
            print("[REDACTED_BY_SCRIPT]")
            return
            
        features = data.get('features', [])
        print(f"[REDACTED_BY_SCRIPT]")
        
        if len(features) == 0:
            print("[REDACTED_BY_SCRIPT]")
            return

        first_feature = features[0]
        geometry = first_feature.get('geometry')
        
        if geometry.get('type') == 'Point':
            print("[REDACTED_BY_SCRIPT]")
        else:
            print(f"[REDACTED_BY_SCRIPT]'type')}, expected Point.")
            
        print("[REDACTED_BY_SCRIPT]")
        
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

def main():
    if not setup_test_data():
        return

    # Monkeypatch paths
    exec_99_decision_dossier.INPUT_DATA_PATH = TEST_INPUT
    exec_99_decision_dossier.OUTPUT_GEOJSON_PATH = TEST_OUTPUT
    
    # Run main
    print("Running pipeline...")
    try:
        exec_99_decision_dossier.main()
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        import traceback
        traceback.print_exc()
        return

    verify_output()

if __name__ == "__main__":
    main()
