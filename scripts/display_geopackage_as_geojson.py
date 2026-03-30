import geopandas as gpd
import os

# Set the SHAPE_RESTORE_SHX config option to YES to allow recreating the .shx file if it's missing
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# 1. Define the path to your input shapefile
shapefile_paths = [
    r"[REDACTED_BY_SCRIPT]"
]



def convert_shapefile_to_geojson(shapefile_path, sample_size=10):
    # Define the number of rows to read (the sample size)
    sample_size = 100

    # 2. Read a small chunk of the shapefile instead of the whole file
    try:
        # By passing the 'rows' argument, we limit the number of features read into memory.
        # This is efficient for large files when only a sample is needed.

        if "geoparquet" in shapefile_path.lower():
            gdf = gpd.read_parquet(shapefile_path).head(sample_size)
        else:
            gdf = gpd.read_file(shapefile_path, rows=sample_size)

        # 3. Define the output path for the GeoJSON file, indicating it's a sample
        output_filename = os.path.splitext(os.path.basename(shapefile_path))[0] + f'[REDACTED_BY_SCRIPT]'
        geojson_path = os.path.join(os.path.dirname(shapefile_path), output_filename)

        # 4. Save the GeoDataFrame to a GeoJSON file
        gdf.to_file(geojson_path, driver='GeoJSON')

        print(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"[REDACTED_BY_SCRIPT]")

for path in shapefile_paths:
    convert_shapefile_to_geojson(path, sample_size=10)
