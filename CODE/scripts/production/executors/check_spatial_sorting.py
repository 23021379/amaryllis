import pandas as pd
import h3
import numpy as np
import re

INPUT_DATA_PATH = r"[REDACTED_BY_SCRIPT]"

def check_sorting():
    print("Reading hex_ids...")
    # Read header to find hex_id column name
    headers = pd.read_csv(INPUT_DATA_PATH, nrows=0).columns.tolist()
    sanitized_headers = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in headers]
    
    hex_col = None
    for orig, san in zip(headers, sanitized_headers):
        if san == 'hex_id':
            hex_col = orig
            break
            
    if not hex_col:
        print("[REDACTED_BY_SCRIPT]")
        return

    df = pd.read_csv(INPUT_DATA_PATH, usecols=[hex_col])
    hexes = df[hex_col].astype(str).str.strip().tolist()
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    def get_coords(hex_list):
        lats, lons = [], []
        for h in hex_list:
            try:
                if hasattr(h3, 'cell_to_latlng'):
                    lat, lon = h3.cell_to_latlng(h)
                else:
                    lat, lon = h3.h3_to_geo(h)
                lats.append(lat)
                lons.append(lon)
            except:
                pass
        return lats, lons

    # First 100
    first_lats, first_lons = get_coords(hexes[:100])
    print(f"\nFirst 100 hexes:")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Last 100
    last_lats, last_lons = get_coords(hexes[-100:])
    print(f"\nLast 100 hexes:")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    # Middle 100
    mid = len(hexes) // 2
    mid_lats, mid_lons = get_coords(hexes[mid:mid+100])
    print(f"\nMiddle 100 hexes:")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    check_sorting()
