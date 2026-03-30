import math

def haversine_distance(lat1, lon1, lat2, lon2):
    # Not applicable for projected coordinates (EPSG:27700), but useful for order of magnitude
    # For projected coordinates, Euclidean distance is exact.
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def demonstrate_precision():
    # Coordinates from user's report
    original = [427300.470986298634671, 236895.448960518522654]
    rounded =  [427300.47098629863,     236895.44896051852]
    
    # Calculate Euclidean distance (since these are meters in EPSG:27700)
    delta_x = original[0] - rounded[0]
    delta_y = original[1] - rounded[1]
    
    distance_meters = math.sqrt(delta_x**2 + delta_y**2)
    
    print(f"Original: {original}")
    print(f"Rounded:  {rounded}")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    if distance_meters < 0.001:
        print("[REDACTED_BY_SCRIPT]")
        print("This confirms the 'other side of the earth'[REDACTED_BY_SCRIPT]")
        print("[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    demonstrate_precision()
