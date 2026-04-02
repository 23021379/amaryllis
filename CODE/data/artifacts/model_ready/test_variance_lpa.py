import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 1. Load the Feature file (X)
x_path = r'[REDACTED_BY_SCRIPT]' 

# 2. Load the Target file (y)
y_path = r'[REDACTED_BY_SCRIPT]'

print("Loading datasets...")
try:
    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Verify lengths match
if len(df_x) != len(df_y):
    print(f"[REDACTED_BY_SCRIPT]")
    min_len = min(len(df_x), len(df_y))
    df_x = df_x.iloc[:min_len]
    df_y = df_y.iloc[:min_len]

# 3. Combine relevant columns
combined_df = pd.DataFrame()
combined_df['planning_authority'] = df_x['planning_authority']
combined_df['[REDACTED_BY_SCRIPT]'] = df_x['[REDACTED_BY_SCRIPT]']
combined_df['capacity'] = df_x['[REDACTED_BY_SCRIPT]']
combined_df['y'] = df_y['[REDACTED_BY_SCRIPT]']

# 4. Filter for Ground Mounted Solar (Type 1)
print("[REDACTED_BY_SCRIPT]")
gm_solar_df = combined_df[combined_df['[REDACTED_BY_SCRIPT]'] == 1].copy()

# Drop rows where essential data is missing
gm_solar_df = gm_solar_df.dropna(subset=['planning_authority', 'y', 'capacity'])

if len(gm_solar_df) == 0:
    print("[REDACTED_BY_SCRIPT]")
    exit()

# 5. Create Capacity Strata
def get_strata(cap):
    if cap < 5: return '0-5 MW'
    elif cap < 15: return '5-15 MW'
    elif cap < 30: return '15-30 MW'
    elif cap < 50: return '30-50 MW'
    else: return '50+ MW'

gm_solar_df['strata'] = gm_solar_df['capacity'].apply(get_strata)

# Define order for the plot
strata_order = ['0-5 MW', '5-15 MW', '15-30 MW', '30-50 MW', '50+ MW']
gm_solar_df['strata'] = pd.Categorical(gm_solar_df['strata'], categories=strata_order, ordered=True)

# Calculate global limits for consistent axes across all plots
global_min = gm_solar_df['y'].min()
global_max = gm_solar_df['y'].max()
# Add a 5% buffer for visual breathing room
buffer = (global_max - global_min) * 0.05
global_xlim = (global_min - buffer, global_max + buffer)
print(f"[REDACTED_BY_SCRIPT]")

# 6. Setup Output
output_folder = '[REDACTED_BY_SCRIPT]'
os.makedirs(output_folder, exist_ok=True)
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

unique_lpas = gm_solar_df['planning_authority'].unique()
print(f"[REDACTED_BY_SCRIPT]")

count_generated = 0
for lpa_val in unique_lpas:
    subset = gm_solar_df[gm_solar_df['planning_authority'] == lpa_val]
    
    # We need enough data points to plot a distribution (KDE)
    # If an LPA has very few points total, a ridgeline plot won't work well.
    if len(subset) < 5:
        continue

    # Check if we have enough data in at least one stratum to make a plot worthwhile
    # (KDE requires at least 2 distinct points in a group usually)
    valid_strata_counts = subset['strata'].value_counts()
    if valid_strata_counts.max() < 1:
        continue

    # --- Print Metrics ---
    print(f"\n{'='*65}")
    print(f"LPA: {lpa_val}")
    print(f"{'='*65}")
    
    # Overall Stats for this LPA
    desc = subset['y'].describe()
    print(f"[REDACTED_BY_SCRIPT]'count'[REDACTED_BY_SCRIPT]'mean'[REDACTED_BY_SCRIPT]'std'[REDACTED_BY_SCRIPT]'50%'[REDACTED_BY_SCRIPT]'min'[REDACTED_BY_SCRIPT]'max']:<5.0f}")
    print("-" * 65)

    # Stats per Strata
    # observed=False ensures we check against the categorical definition, though we filter for count > 0
    strata_stats = subset.groupby('strata', observed=False)['y'].describe()
    
    for cat in strata_order:
        if cat in strata_stats.index:
            row = strata_stats.loc[cat]
            if row['count'] > 0:
                print(f"[REDACTED_BY_SCRIPT]'count']):<4} | Mean={row['mean'[REDACTED_BY_SCRIPT]'std'[REDACTED_BY_SCRIPT]'50%'[REDACTED_BY_SCRIPT]'min'[REDACTED_BY_SCRIPT]'max']:<5.0f}")
    print("-" * 65)
    # ---------------------

    try:
        # Initialize the FacetGrid object
        g = sns.FacetGrid(subset, row="strata", hue="strata", aspect=5, height=1.2, 
                          palette="viridis", row_order=strata_order, hue_order=strata_order)

        # Set consistent x-axis limits
        g.set(xlim=global_xlim)

        # Draw the densities
        # warn_singular=False suppresses warnings when a stratum has constant values (variance=0)
        g.map(sns.kdeplot, "y", clip_on=False, fill=True, alpha=0.7, linewidth=1.5, warn_singular=False)
        g.map(sns.kdeplot, "y", clip_on=False, color="w", lw=2, warn_singular=False)
        
        # Add a horizontal line for reference
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        #add vertical lines at different y intervals for reference
        def vertical_line(x, **kwargs):
            for xpos in [500, 1000, 1500, 2000, 2500]:
                plt.axvline(x=xpos, color='gray', linestyle='--', linewidth=0.2, alpha=1.0)
        g.map(vertical_line, "y")

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, "y")

        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-0.5)

        # Remove axes details that don't matter
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        # Add Main Title
        plt.suptitle(f'[REDACTED_BY_SCRIPT]', fontsize=16, y=0.98)
        
        # Save
        safe_lpa_name = str(lpa_val).replace('/', '_').replace('\\', '_').replace(' ', '_')
        filename = f"[REDACTED_BY_SCRIPT]"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        count_generated += 1
        
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")
        plt.close()

print(f"[REDACTED_BY_SCRIPT]'{output_folder}/'")