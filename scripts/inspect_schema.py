import geopandas as gpd
from pathlib import Path

# --- CONFIGURATION ---
TARGET_FILE = Path(r"[REDACTED_BY_SCRIPT]")
TARGET_COLUMN = 'CMZ_CODE'
SAMPLE_SIZE = 5000000 #Inspect the first N rows to get a representative sample

# ---------------------

def inspect_schema(file_path: Path, column: str, sample_size: int):
    """[REDACTED_BY_SCRIPT]"""
    if not file_path.exists():
        print(f"[REDACTED_BY_SCRIPT]")
        return

    print(f"Inspecting column '{column}' in file '{file_path.name}'...")
    
    try:
        gdf_sample = gpd.read_file(file_path, rows=slice(0, sample_size))
        
        if column not in gdf_sample.columns:
            print(f"ERROR: Column '{column}' not found in the file.")
            print(f"[REDACTED_BY_SCRIPT]")
            return
            
        unique_values = gdf_sample[column].unique().tolist()
        
        print(f"[REDACTED_BY_SCRIPT]")
        for value in sorted(unique_values, key=lambda x: (x is None, x)):
             print(f"- {value}")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == '__main__':
    inspect_schema(TARGET_FILE, TARGET_COLUMN, SAMPLE_SIZE)