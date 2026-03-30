# Quick debug test - check if models and historical_data are actually loaded

import sys
sys.path.append(r'[REDACTED_BY_SCRIPT]')

# Import the functions
from exec_99_decision_dossier import load_all_models, load_historical_data

print("[REDACTED_BY_SCRIPT]")
try:
    models = load_all_models()
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]'shap_explainer')}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("[REDACTED_BY_SCRIPT]")
try:
    historical_data = load_historical_data()
    print(f"[REDACTED_BY_SCRIPT]")
    if historical_data.get('coordinates') is not None:
        print(f"[REDACTED_BY_SCRIPT]'coordinates'].shape}")
    else:
        print(f"  Coordinates: None")
except Exception as e:
    print(f"✗ Failed: {e}")
