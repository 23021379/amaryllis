import pandas as pd

def count_accepted_solar_submissions(file_path):
    try:
        # Load the dataset with 'ISO-8859-1' encoding to fix the byte error
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

        # Filter for Solar technology (case-insensitive)
        df = df[df['Technology Type'].str.contains('Solar', case=False, na=False)]

        # Convert the '[REDACTED_BY_SCRIPT]' column to datetime objects
        df['[REDACTED_BY_SCRIPT]'] = pd.to_datetime(df['[REDACTED_BY_SCRIPT]'], dayfirst=True, errors='coerce')

        # Filter for the date range: Jan 1, 2019 to Dec 31, 2025
        date_mask = (df['[REDACTED_BY_SCRIPT]'] >= '2019-01-01') & (df['[REDACTED_BY_SCRIPT]'] <= '2025-12-31')
        
        # Apply the filter and create a copy to avoid warnings
        filtered_df = df[date_mask].copy()

        # Ensure capacity is numeric
        filtered_df['[REDACTED_BY_SCRIPT]'] = pd.to_numeric(filtered_df['[REDACTED_BY_SCRIPT]'], errors='coerce')

        # Define size bins and labels
        # Bins: 0-1, 1-5, 5-10, 10-20, 20-50, 50+
        bins = [0, 1, 5, 10, 20, 50, float('inf')]
        labels = ['0-1MW', '1-5MW', '5-10MW', '10-20MW', '20-50MW', '50MW+']
        
        # Create a new column for the size range
        filtered_df['Size Range'] = pd.cut(filtered_df['[REDACTED_BY_SCRIPT]'], bins=bins, labels=labels)

        # Get the total count
        total_count = len(filtered_df)
        print(f"[REDACTED_BY_SCRIPT]")
        
        # Extract year for grouping
        filtered_df['Year'] = filtered_df['[REDACTED_BY_SCRIPT]'].dt.year

        # Group by Year and Size Range
        # observed=False ensures we see all categories even if some are empty for a specific year
        breakdown = filtered_df.groupby(['Year', 'Size Range'], observed=False).size()

        # Get list of years to iterate through
        years = sorted(filtered_df['Year'].unique())

        print("[REDACTED_BY_SCRIPT]")
        print("-" * 30)

        for year in years:
            year_total = len(filtered_df[filtered_df['Year'] == year])
            print(f"[REDACTED_BY_SCRIPT]")
            
            # Print the breakdown for this specific year
            year_data = breakdown[year]
            for category in labels:
                count = year_data.get(category, 0)
                if count > 0:
                    print(f"[REDACTED_BY_SCRIPT]")
            print("") # Add a blank line between years

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
# Ensure the CSV file is in the same directory as this script
file_name = r"[REDACTED_BY_SCRIPT]"
count_accepted_solar_submissions(file_name)