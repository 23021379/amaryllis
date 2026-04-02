import pandas as pd
import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the project's root directory
project_root = os.path.dirname(script_dir)

# Define the input and output file paths relative to the project root
input_csv_path = os.path.join(project_root, '[REDACTED_BY_SCRIPT]')
output_csv_path = os.path.join(project_root, 'baseline.csv')

# Define the required columns from the source file
id_col = 'solar_farm_id'
easting_col = 'easting_x'
northing_col = 'northing_x'

# Define the desired column names in the output file
output_columns = {
    id_col: 'solar_farm_id',
    easting_col: 'easting_x',
    northing_col: 'northing_x'
}

# Read the source CSV file
# Use a try-except block to handle potential encoding issues
try:
    df = pd.read_csv(input_csv_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(input_csv_path, encoding='latin1')

# Select the specified columns for the baseline file
baseline_df = df[[id_col, easting_col, northing_col]]

# Rename the columns to the desired output names
baseline_df.rename(columns=output_columns, inplace=True)

# Save the resulting DataFrame to a new CSV file
baseline_df.to_csv(output_csv_path, index=False)

# Print a confirmation message with the path to the new file and the number of records
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
