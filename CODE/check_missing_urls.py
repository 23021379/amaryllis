import json
from pathlib import Path

# Configuration
REGISTRY_PATH = Path("[REDACTED_BY_SCRIPT]")

def find_missing_urls():
    if not REGISTRY_PATH.exists():
        print(f"[REDACTED_BY_SCRIPT]")
        return

    try:
        with open(REGISTRY_PATH, 'r') as f:
            registry = json.load(f)
    except json.JSONDecodeError:
        print("[REDACTED_BY_SCRIPT]")
        return

    missing_url_lpas = []
    
    for entry in registry:
        # Check if portal_url is missing (None/null)
        if entry.get("portal_url") is None:
            # Only include if there is actually a name (skipping empty rows if any)
            if entry.get("lpa_name_raw") or entry.get("lpa_name_clean"):
                missing_url_lpas.append(entry)

    print(f"--- Scan Complete ---")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print("-" * 30)

    if missing_url_lpas:
        print("[REDACTED_BY_SCRIPT]")
        for lpa in missing_url_lpas:
            name = lpa.get("lpa_name_clean") or lpa.get("lpa_name_raw") or "Unknown"
            print(f" - {name}")
    else:
        print("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    find_missing_urls()
