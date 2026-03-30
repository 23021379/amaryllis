import geopandas as gpd
import fiona
import os

# Input path
gpkg_path = r"[REDACTED_BY_SCRIPT]"

# Ensure input exists
if not os.path.exists(gpkg_path):
    print(f"[REDACTED_BY_SCRIPT]")
    exit(1)

# Output directory (same as input)
output_dir = os.path.dirname(gpkg_path)

print(f"[REDACTED_BY_SCRIPT]")

try:
    # List all layers in the GeoPackage
    layers = fiona.listlayers(gpkg_path)
    print(f"[REDACTED_BY_SCRIPT]")

    for layer in layers:
        print(f"Reading layer: '{layer}'...")
        
        # Read the layer
        gdf = gpd.read_file(gpkg_path, layer=layer)
        
        # Construct output filename
        # If there's only one layer, we can use the original filename, otherwise append layer name
        if len(layers) == 1:
            base_name = os.path.splitext(os.path.basename(gpkg_path))[0]
            output_filename = f"{base_name}.geojson"
        else:
            output_filename = f"{layer}.geojson"
            
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[REDACTED_BY_SCRIPT]")
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"[REDACTED_BY_SCRIPT]")

except Exception as e:
    print(f"An error occurred: {e}")
