import json
import os
import geopandas as gpd

def create_geojson_sample(input_path, output_path, num_features=100):
    """
    Creates a small sample of a large geospatial file (GPKG, Shapefile, etc.)
    by taking the first few features and saving as GeoJSON.

    Args:
        input_path (str): Path to the large input file (e.g., .gpkg).
        output_path (str): Path to write the sampled GeoJSON file.
        num_features (int): The number of features to include in the sample.
    """
    if not os.path.exists(input_path):
        print(f"[REDACTED_BY_SCRIPT]")
        return

    try:
        print("[REDACTED_BY_SCRIPT]")
        # Read only the first n rows
        gdf = gpd.read_file(input_path, rows=num_features)
        
        # Reproject to WGS84 (EPSG:4326) for standard GeoJSON compatibility
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            print(f"[REDACTED_BY_SCRIPT]")
            gdf = gdf.to_crs(epsg=4326)
        
        print(f"[REDACTED_BY_SCRIPT]")
        gdf.to_file(output_path, driver='GeoJSON')
        
        print(f"[REDACTED_BY_SCRIPT]")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    # Please update these file paths according to your needs.
    
    # This is one of the large GeoJSON files found in your workspace.
    # You might want to change it to the one you are interested in.
    INPUT_FILE = r"[REDACTED_BY_SCRIPT]"
    
    # The output file will be much smaller and contain only a few features.
    OUTPUT_SAMPLE_GEOJSON = r"[REDACTED_BY_SCRIPT]"
    
    # Number of features to extract.
    NUM_FEATURES_TO_SAMPLE = 10000
    # ---------------------

    # Use exact paths provided if they are absolute, otherwise resolve
    if os.path.isabs(INPUT_FILE):
        input_file_path = INPUT_FILE
    else:
        # Fallback logic if relative paths are used
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        input_file_path = os.path.join(workspace_dir, '..', INPUT_FILE)

    if os.path.isabs(OUTPUT_SAMPLE_GEOJSON):
        output_file_path = OUTPUT_SAMPLE_GEOJSON
    else:
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        output_file_path = os.path.join(workspace_dir, '..', OUTPUT_SAMPLE_GEOJSON)

    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    create_geojson_sample(input_file_path, output_file_path, NUM_FEATURES_TO_SAMPLE)
