import joblib
import os
import pandas as pd

ARTIFACTS_DIR = r"[REDACTED_BY_SCRIPT]"
ARBITER_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "[REDACTED_BY_SCRIPT]")

try:
    if os.path.exists(ARBITER_SCALER_PATH):
        scaler = joblib.load(ARBITER_SCALER_PATH)
        if hasattr(scaler, 'feature_names_in_'):
            print("[REDACTED_BY_SCRIPT]")
            for f in scaler.feature_names_in_:
                print(f"  - {f}")
        else:
            print("[REDACTED_BY_SCRIPT]")
    else:
        print(f"[REDACTED_BY_SCRIPT]")

    # Also check one head
    HEADS_DIR = os.path.join(ARTIFACTS_DIR, "heads")
    if os.path.exists(HEADS_DIR):
        files = os.listdir(HEADS_DIR)
        if files:
            head_path = os.path.join(HEADS_DIR, files[0])
            head = joblib.load(head_path)
            print(f"[REDACTED_BY_SCRIPT]")
except Exception as e:
    print(f"Error: {e}")
