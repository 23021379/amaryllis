import joblib
import os
import pandas as pd

ARTIFACTS_DIR = r"[REDACTED_BY_SCRIPT]"

def inspect_model(name, path):
    print(f"[REDACTED_BY_SCRIPT]")
    try:
        model = joblib.load(path)
        if hasattr(model, 'feature_name_'):
            print(f"[REDACTED_BY_SCRIPT]")
        elif hasattr(model, 'feature_names_in_'):
            print(f"[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            
        if hasattr(model, 'n_features_in_'):
            print(f"[REDACTED_BY_SCRIPT]")
            
        if isinstance(model, dict):
            print("[REDACTED_BY_SCRIPT]", model.keys())
            first_key = list(model.keys())[0]
            first_model = model[first_key]
            print(f"[REDACTED_BY_SCRIPT]'{first_key}':")
            if hasattr(first_model, 'feature_name_'):
                print(f"[REDACTED_BY_SCRIPT]")
            elif hasattr(first_model, 'feature_names_in_'):
                print(f"[REDACTED_BY_SCRIPT]")
            
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
    print("\n")

inspect_model("Arbiter", os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]"))
inspect_model("GM Specialist", os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]"))
inspect_model("Solar Stratified", os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]"))
inspect_model("GM Stratified", os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]"))
