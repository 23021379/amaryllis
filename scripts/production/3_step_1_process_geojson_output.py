import json
import os
import h3
import statistics
from collections import defaultdict
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = r"[REDACTED_BY_SCRIPT]"
OUTPUT_DIR = r"[REDACTED_BY_SCRIPT]"

# Colors: Green (Best/Top 25%), Amber, Orange, Red (Worst/Bottom 25%)
COLOR_RAMP = ["#00cc00", "#ffbf00", "#ff8000", "#ff0000"]

RISK_ICON_MAP = {
    "SIC_LPA_INSTABILITY_RISK": "LPA",
    "[REDACTED_BY_SCRIPT]": "GRID_Q",
    "[REDACTED_BY_SCRIPT]": "GRID_PQ",
    "[REDACTED_BY_SCRIPT]": "GRID_LTDS",
    "[REDACTED_BY_SCRIPT]": "CAP",
    "feature_missing": "NONE"
}

def generate_h3_geometry(h3_index):
    try:
        boundary = h3.cell_to_boundary(h3_index)
        geojson_coords = [[lon, lat] for lat, lon in boundary]
        geojson_coords.append(geojson_coords[0]) 
        return {"type": "Polygon", "coordinates": [geojson_coords]}
    except:
        return None

def apply_rank_styling_inplace(features, metric_key):
    """
    Sorts strictly within the provided list (capacity silo) and injects visual styles.
    """
    # Filter valid
    valid_items = []
    for f in features:
        val = f['properties'].get(metric_key)
        if val is not None:
            valid_items.append((val, f))
    
    # Sort Ascending (Lower Duration = Better)
    valid_items.sort(key=lambda x: x[0])
    
    total = len(valid_items)
    if total == 0: return

    # Assign Ranks
    for rank, (val, feat) in enumerate(valid_items):
        percentile = (rank / total) * 100
        
        if percentile < 25:
            color, label = COLOR_RAMP[0], "Top 25% (Best)"
        elif percentile < 50:
            color, label = COLOR_RAMP[1], "25% - 50%"
        elif percentile < 75:
            color, label = COLOR_RAMP[2], "50% - 75%"
        else:
            color, label = COLOR_RAMP[3], "Bottom 25% (Worst)"
            
        # Inject standard style props
        feat['properties']['fill'] = color
        feat['properties']['marker-color'] = color
        feat['properties']['stroke'] = "#ffffff"
        feat['properties']['stroke-width'] = 1
        feat['properties']['fill-opacity'] = 0.8
        
        feat['properties']['[REDACTED_BY_SCRIPT]'] = label
        feat['properties']['rank_percentile'] = round(percentile, 1)

def process_dossiers_thematic():
    print(f"[REDACTED_BY_SCRIPT]")
    
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    scenarios = defaultdict(list)

    print("[REDACTED_BY_SCRIPT]")
    for feature in tqdm(data['features']):
        p = feature['properties']
        raw_id = p.get('hex_id', '')
        if '_' not in raw_id: continue
        
        base_h3, cap_suffix = raw_id.split('_')
        clean_geom = generate_h3_geometry(base_h3)
        if not clean_geom: continue
        
        # 1. Extract Master Data Dictionary
        # We grab everything needed for BOTH files here.
        master_props = {
            "hex_id": base_h3,
            # Core
            "capacity_mw": int(p.get('optimal_capacity_mw', 0)),
            "duration_days": int(p.get('[REDACTED_BY_SCRIPT]', 999)),
            
            # Uncertainty & Error (For Prediction File)
            "uncert_p90": int(p.get('[REDACTED_BY_SCRIPT]', 0)),
            "uncert_spread": int(p.get('[REDACTED_BY_SCRIPT]', 0) - p.get('[REDACTED_BY_SCRIPT]', 0)),
            "prob_accurate": p.get('error_prob_prob_accurate', 0),
            "prob_under": p.get('[REDACTED_BY_SCRIPT]', 0),
            "prob_over": p.get('[REDACTED_BY_SCRIPT]', 0),
            "geo_mae": p.get('geo_bench_mae', 0),
            "geo_count": p.get('geo_bench_count', 0),
            
            # Drivers (For Drivers File)
            "driver_1_feat": p.get('driver_1_feature'),
            "driver_1_imp": p.get('driver_1_impact'),
            "driver_1_desc": p.get('driver_1_desc'),
            "driver_2_feat": p.get('driver_2_feature'),
            "driver_2_imp": p.get('driver_2_impact'),
            "driver_2_desc": p.get('driver_2_desc'),
            "driver_3_feat": p.get('driver_3_feature'),
            "driver_3_imp": p.get('driver_3_impact'),
            "driver_3_desc": p.get('driver_3_desc'),
            
            # Icons
            "risk_icon": RISK_ICON_MAP.get(p.get('[REDACTED_BY_SCRIPT]'), "GEN")
        }

        # Store master object
        scenarios[f"{master_props['capacity_mw']}MW"].append({
            "type": "Feature",
            "geometry": clean_geom,
            "properties": master_props
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[REDACTED_BY_SCRIPT]")
    for cap_key, features in scenarios.items():
        # 1. Apply Ranking IN-PLACE (Calculates fill color based on master list)
        apply_rank_styling_inplace(features, "duration_days")
        
        # 2. Split into Themes
        pred_features = []
        driver_features = []
        
        for f in features:
            base_props = f['properties']
            
            # Shared Styling Props
            style_props = {
                "hex_id": base_props['hex_id'],
                "fill": base_props.get('fill'),
                "marker-color": base_props.get('marker-color'),
                "stroke": base_props.get('stroke'),
                "stroke-width": base_props.get('stroke-width'),
                "fill-opacity": base_props.get('fill-opacity'),
                "risk_icon": base_props['risk_icon']
            }
            
            # Theme A: Prediction
            p_props = style_props.copy()
            p_props.update({
                "capacity_mw": base_props['capacity_mw'],
                "duration_days": base_props['duration_days'],
                "[REDACTED_BY_SCRIPT]": base_props['[REDACTED_BY_SCRIPT]'], # Calculated Rank
                "rank_percentile": base_props['rank_percentile'],
                "uncert_p90": base_props['uncert_p90'],
                "uncert_spread": base_props['uncert_spread'],
                "prob_accurate": base_props['prob_accurate'],
                "prob_under": base_props['prob_under'],
                "prob_over": base_props['prob_over'],
                "geo_mae": base_props['geo_mae'],
                "geo_count": base_props['geo_count']
            })
            pred_features.append({"type": "Feature", "geometry": f['geometry'], "properties": p_props})
            
            # Theme B: Drivers
            d_props = style_props.copy()
            d_props.update({
                "driver_1_feature": base_props['driver_1_feat'],
                "driver_1_impact": base_props['driver_1_imp'],
                "driver_1_desc": base_props['driver_1_desc'],
                "driver_2_feature": base_props['driver_2_feat'],
                "driver_2_impact": base_props['driver_2_imp'],
                "driver_2_desc": base_props['driver_2_desc'],
                "driver_3_feature": base_props['driver_3_feat'],
                "driver_3_impact": base_props['driver_3_imp'],
                "driver_3_desc": base_props['driver_3_desc']
            })
            driver_features.append({"type": "Feature", "geometry": f['geometry'], "properties": d_props})
            
        # 3. Write Files
        # File A
        with open(os.path.join(OUTPUT_DIR, f"[REDACTED_BY_SCRIPT]"), 'w') as f:
            json.dump({"type": "FeatureCollection", "features": pred_features}, f)
            
        # File B
        with open(os.path.join(OUTPUT_DIR, f"[REDACTED_BY_SCRIPT]"), 'w') as f:
            json.dump({"type": "FeatureCollection", "features": driver_features}, f)

    print(f"--- OPERATION COMPLETE ---")
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    process_dossiers_thematic()