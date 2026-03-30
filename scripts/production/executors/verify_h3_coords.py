import h3
import json

def verify_h3_conversion():
    # Sample hex from user
    hex_id = "87195d5acffffff"
    
    try:
        # Try H3 v4.x API
        if hasattr(h3, 'cell_to_latlng'):
            lat, lon = h3.cell_to_latlng(hex_id)
            print("[REDACTED_BY_SCRIPT]")
        # Fallback to v3.x API (though previous test failed)
        elif hasattr(h3, 'h3_to_geo'):
            lat, lon = h3.h3_to_geo(hex_id)
            print("[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            return

        print(f"Hex ID: {hex_id}")
        print(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        
        # Check if it's in the UK (approx bounds: Lat 50-60, Lon -8 to 2)
        if 49 < lat < 61 and -9 < lon < 2:
            print("[REDACTED_BY_SCRIPT]")
        else:
            print("[REDACTED_BY_SCRIPT]")
            
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    verify_h3_conversion()
