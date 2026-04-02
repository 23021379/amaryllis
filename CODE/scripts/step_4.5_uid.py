import hashlib
import pandas as pd

# Project-wide constant for salting the unique identifier.
# This ensures our generated IDs are unique to this project.
AMARYLLIS_SALT = "[REDACTED_BY_SCRIPT]"

def generate_amaryllis_id(df: pd.DataFrame) -> pd.Series:
    """
    Generates a deterministic, collision-resistant unique identifier (amaryllis_id)
    for each row in the input DataFrame based on a set of immutable columns.

    Args:
        df: The L4 DataFrame.

    Returns:
        A pandas Series containing the amaryllis_id for each row.
    """
    key_cols = [
        'easting', 
        'northing', 
        '[REDACTED_BY_SCRIPT]', 
        'submission_year', 
        'submission_month', 
        'submission_day'
    ]
    
    # Stage 2: Canonical String Conversion Protocol
    # Create a temporary DataFrame to avoid modifying the original until the end.
    df_keys = pd.DataFrame(index=df.index)
    df_keys['easting_str'] = df['easting'].map('{:.2f}'.format)
    df_keys['northing_str'] = df['northing'].map('{:.2f}'.format)
    df_keys['capacity_str'] = df['[REDACTED_BY_SCRIPT]'].map('{:.3f}'.format)
    df_keys['year_str'] = df['submission_year'].astype(int).astype(str)
    df_keys['month_str'] = df['submission_month'].astype(int).astype(str)
    df_keys['day_str'] = df['submission_day'].astype(int).astype(str)

    canonical_string = (
        df_keys['easting_str'] + '|' +
        df_keys['northing_str'] + '|' +
        df_keys['capacity_str'] + '|' +
        df_keys['year_str'] + '|' +
        df_keys['month_str'] + '|' +
        df_keys['day_str']
    )

    # Stage 3: Hashing & Salting Protocol
    def create_hash(value: str) -> str:
        salted_string = AMARYLLIS_SALT + value
        return hashlib.sha256(salted_string.encode('utf-8')).hexdigest()

    return canonical_string.apply(create_hash)

l4_df = pd.read_csv('[REDACTED_BY_SCRIPT]')
# --- Assume l4_df is loaded and has undergone initial cleaning ---

# --- Assume l4_df is loaded and has undergone initial cleaning ---

# Remediation: Surgically remove duplicate records based on the composite key.
# This must be done BEFORE generating the amaryllis_id.
composite_key_cols = [
    'easting', 
    'northing', 
    '[REDACTED_BY_SCRIPT]', 
    'submission_year', 
    'submission_month', 
    'submission_day'
]
initial_row_count = len(l4_df)
l4_df.drop_duplicates(subset=composite_key_cols, keep='first', inplace=True)
final_row_count = len(l4_df)

if initial_row_count > final_row_count:
    print(f"[REDACTED_BY_SCRIPT]")
else:
    print("[REDACTED_BY_SCRIPT]")

print("[REDACTED_BY_SCRIPT]")
l4_df['amaryllis_id'] = generate_amaryllis_id(l4_df)

# Stage 4: The Uniqueness Assertion and Diagnostic Report
if not l4_df['amaryllis_id'].is_unique:
    print("[REDACTED_BY_SCRIPT]")
    duplicated_ids = l4_df[l4_df['amaryllis_id'].duplicated(keep=False)]['amaryllis_id']
    print("[REDACTED_BY_SCRIPT]")
    conflicting_records = l4_df[l4_df['amaryllis_id'].isin(duplicated_ids)].sort_values('amaryllis_id')
    
    # Using pandas options to ensure the full rows are visible for diagnosis
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(conflicting_records)
    
    raise AssertionError("[REDACTED_BY_SCRIPT]")
else:
    print("[REDACTED_BY_SCRIPT]")

print("Setting 'amaryllis_id'[REDACTED_BY_SCRIPT]")
all_cols = l4_df.columns.tolist()
all_cols.insert(0, all_cols.pop(all_cols.index('amaryllis_id')))
l4_df = l4_df[all_cols]

# --- The script would now proceed with saving the L4 artifact ---

l4_df.to_csv('[REDACTED_BY_SCRIPT]', index=False)