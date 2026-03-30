import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import xgboost as xgb

# Load the datasets
X = pd.read_csv('[REDACTED_BY_SCRIPT]')
y = pd.read_csv('[REDACTED_BY_SCRIPT]')

# Merge the features (X) and target (y) on 'amaryllis_id'
df = pd.merge(X, y, on='amaryllis_id')

# Drop rows where the target '[REDACTED_BY_SCRIPT]' is missing
df = df.dropna(subset=['[REDACTED_BY_SCRIPT]'])

# Set a consistent style for the plots
sns.set_theme(style="whitegrid")
#df = df[(df['[REDACTED_BY_SCRIPT]'] != 1)]
def run_stratified_analysis(df, range_label):
    """
    Runs the full suite of analysis on a specific subset of data (stratified by capacity).
    """
    # Create a directory to save the graphs for this specific range
    output_dir = f'[REDACTED_BY_SCRIPT]'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    if len(df) < 5:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    # ---------------------------------------------------------
    # 1. Primary Analysis: Solar Specific Duration vs. Capacity
    # ---------------------------------------------------------
    # Filter for Solar Technology (Type 21)
    # Note: df is already filtered by capacity range in the main loop
    solar_df = df[df['technology_type'] == 21]

    if not solar_df.empty:
        # Plot A: All Solar
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=solar_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.6, color='orange')
        if len(solar_df) > 1:
            sns.regplot(data=solar_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('[REDACTED_BY_SCRIPT]')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

        # Plot B: Ground-Mounted Solar Only (Mounting Type 1)
        gm_solar_df = solar_df[solar_df['[REDACTED_BY_SCRIPT]'] == 1]

        if not gm_solar_df.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=gm_solar_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.6, color='brown')
            if len(gm_solar_df) > 1:
                sns.regplot(data=gm_solar_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # B. Duration by Technology Type (if available)
    tech_col = next((col for col in ['technology', 'technology_type', 'type'] if col in df.columns), None)
    if tech_col:
        plt.figure(figsize=(10, 6))
        order = df.groupby(tech_col)['[REDACTED_BY_SCRIPT]'].median().sort_values().index
        sns.boxplot(data=df, x=tech_col, y='[REDACTED_BY_SCRIPT]', order=order)
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # C. Top 15 Feature Correlations (Targeted Analysis)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    # Drop columns with 0 variance to avoid warnings
    numeric_df = numeric_df.loc[:, numeric_df.var() > 0]
    
    if not numeric_df.empty and '[REDACTED_BY_SCRIPT]' in numeric_df.columns:
        corr_series = numeric_df.corrwith(df['[REDACTED_BY_SCRIPT]'])
        # Get top 15 features by absolute correlation
        top_corr = corr_series.abs().sort_values(ascending=False).head(16) 
        top_corr = corr_series[top_corr.index].drop('[REDACTED_BY_SCRIPT]', errors='ignore') 

        if not top_corr.empty:
            sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # ---------------------------------------------------------
    # 3. Variance Hypothesis Testing
    # ---------------------------------------------------------

    # A. The "Grid Friction" Hypothesis
    if '[REDACTED_BY_SCRIPT]' in df.columns:
        plt.figure(figsize=(10, 6))
        grid_df = df[df['[REDACTED_BY_SCRIPT]'] < 500] 
        if not grid_df.empty:
            sns.scatterplot(data=grid_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', hue=(df['technology_type'] == 21), palette={True: 'orange', False: 'grey'}, alpha=0.5)
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.legend(title='Is Solar?')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # B. The "NIMBY" Hypothesis
    if '[REDACTED_BY_SCRIPT]' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('NIMBY Risk Score')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # C. The "LPA Bottleneck" Hypothesis
    if '[REDACTED_BY_SCRIPT]' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5)
        max_val = df['[REDACTED_BY_SCRIPT]'].max()
        if pd.notna(max_val):
            plt.plot([0, max_val], [0, max_val], 'r--', label='Average Pace')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('[REDACTED_BY_SCRIPT]')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # ---------------------------------------------------------
    # 4. Deep Dive: Constraint Analysis
    # ---------------------------------------------------------
    focus_df = df[df['technology_type'].isin([21, 4])]

    if not focus_df.empty:
        # A. The "Food vs. Fuel" Conflict
        if 'alc_is_bmv_at_site' in focus_df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=focus_df, x='alc_is_bmv_at_site', y='[REDACTED_BY_SCRIPT]', palette='Set2')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. The "Ecological Drag"
        if 'sssi_dist_to_nearest_m' in focus_df.columns:
            plt.figure(figsize=(10, 6))
            eco_df = focus_df[focus_df['sssi_dist_to_nearest_m'] < 2000]
            if not eco_df.empty:
                sns.scatterplot(data=eco_df, x='sssi_dist_to_nearest_m', y='[REDACTED_BY_SCRIPT]', hue='technology_type', alpha=0.7)
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                if len(eco_df) > 1:
                    sns.regplot(data=eco_df, x='sssi_dist_to_nearest_m', y='[REDACTED_BY_SCRIPT]', scatter=False, color='green', line_kws={"linestyle": "--"})
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. The "Urban Fringe" Battleground
        if '[REDACTED_BY_SCRIPT]' in focus_df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=focus_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='purple')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 5. Phase II & III: Administrative & Technical Drivers
    # ---------------------------------------------------------
    solar_focus = df[df['technology_type'] == 21]

    if not solar_focus.empty:
        # A. The "LPA Lottery" (Tiers)
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            lpa_clean = solar_focus[
                (solar_focus['[REDACTED_BY_SCRIPT]'] > 0) & 
                (solar_focus['[REDACTED_BY_SCRIPT]'] < 730)
            ].copy()
            
            if not lpa_clean.empty:
                def classify_lpa(days):
                    if days <= 112: return 'Fast (<16 wks)'
                    elif days <= 250: return 'Moderate'
                    else: return 'Sluggish (>36 wks)'
                    
                lpa_clean['LPA_Speed_Tier'] = lpa_clean['[REDACTED_BY_SCRIPT]'].apply(classify_lpa)
                order = ['Fast (<16 wks)', 'Moderate', 'Sluggish (>36 wks)']
                
                sns.boxplot(data=lpa_clean, x='LPA_Speed_Tier', y='[REDACTED_BY_SCRIPT]', order=order, palette='viridis')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. The "Deep Rural Dividend"
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns and '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            dist_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] <= 15]
            if not dist_df.empty:
                sns.scatterplot(
                    data=dist_df, 
                    x='[REDACTED_BY_SCRIPT]', 
                    y='[REDACTED_BY_SCRIPT]', 
                    hue='[REDACTED_BY_SCRIPT]',
                    palette='Greens',
                    alpha=0.7
                )
                if len(dist_df) > 1:
                    sns.regplot(data=dist_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.legend(title='Ag Area % (1km)', loc='upper right')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. The "Site Sprawl" (Log Scale)
        if 'solar_site_area_sqm' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            area_df = solar_focus[solar_focus['solar_site_area_sqm'] > 1000]
            if not area_df.empty:
                sns.scatterplot(data=area_df, x='solar_site_area_sqm', y='[REDACTED_BY_SCRIPT]', alpha=0.4, color='darkgreen')
                plt.xscale('log')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 6. Phase IV: Advanced Hypothesis Testing
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. The "Voltage Ceiling"
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            valid_voltages = [11.0, 33.0, 66.0, 132.0]
            voltage_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'].isin(valid_voltages)].copy()
            if not voltage_df.empty:
                sns.boxplot(data=voltage_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', palette='magma')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. The "Wealth Barrier"
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=solar_focus, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='gold')
            if len(solar_focus) > 1:
                sns.regplot(data=solar_focus, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='black')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. The "Grid Queue"
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            constraint_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] <= 10]
            if not constraint_df.empty:
                sns.violinplot(data=constraint_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', palette='cool', inner="quartile")
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 7. Phase V: Contextual & Environmental Drivers
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Cumulative Fatigue
        if 'knn_count_solar' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            neighbor_df = solar_focus[solar_focus['knn_count_solar'] <= 20]
            if not neighbor_df.empty:
                sns.boxplot(data=neighbor_df, x='knn_count_solar', y='[REDACTED_BY_SCRIPT]', palette='Blues')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Hostile Environment
        if 'lpa_approval_rate_cps2' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=solar_focus, x='lpa_approval_rate_cps2', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='crimson')
            if len(solar_focus) > 1:
                sns.regplot(data=solar_focus, x='lpa_approval_rate_cps2', y='[REDACTED_BY_SCRIPT]', scatter=False, color='black')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('LPA Approval Rate')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 8. Phase VI: Complexity, History, and Capacity
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Constraint Thicket
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            constraint_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] <= 15]
            if not constraint_df.empty:
                sns.boxplot(data=constraint_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', palette='Reds')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Baggage Effect
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            def classify_baggage(count):
                if count == 0: return 'Clean History (0)'
                elif count == 1: return 'Resubmission (1)'
                else: return 'Contentious (2+)'
            
            solar_focus_copy = solar_focus.copy()
            solar_focus_copy['Site_Baggage'] = solar_focus_copy['[REDACTED_BY_SCRIPT]'].apply(classify_baggage)
            order = ['Clean History (0)', 'Resubmission (1)', 'Contentious (2+)']
            
            sns.violinplot(data=solar_focus_copy, x='Site_Baggage', y='[REDACTED_BY_SCRIPT]', order=order, palette='muted')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 9. Phase VII: The "Killer" Micro-Drivers
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Linear Obstacle Course
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            crossing_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] <= 10]
            if not crossing_df.empty:
                sns.boxplot(data=crossing_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', palette='coolwarm')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Killer Constraint
        constraint_cols = [c for c in solar_focus.columns if c.startswith('[REDACTED_BY_SCRIPT]')]
        if constraint_cols:
            type_df = solar_focus.copy()
            def get_dominant_constraint(row):
                for col in constraint_cols:
                    if row[col] == 1:
                        return col.replace('[REDACTED_BY_SCRIPT]', '')
                return 'None'

            type_df['Dominant_Constraint'] = type_df.apply(get_dominant_constraint, axis=1)
            valid_types = type_df['Dominant_Constraint'].value_counts()
            valid_types = valid_types[valid_types > 5].index.tolist()
            type_df = type_df[(type_df['Dominant_Constraint'].isin(valid_types)) & (type_df['Dominant_Constraint'] != 'None')]
            
            if not type_df.empty:
                order = type_df.groupby('Dominant_Constraint')['[REDACTED_BY_SCRIPT]'].median().sort_values().index
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=type_df, x='Dominant_Constraint', y='[REDACTED_BY_SCRIPT]', order=order, palette='Set2')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xticks(rotation=45)
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 10. Phase VIII: Human Factors & Efficiency
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Digital Activism
        oac_cols = [c for c in solar_focus.columns if c.startswith('site_lsoa_oac_')]
        if oac_cols:
            oac_df = solar_focus.copy()
            def get_dominant_oac(row):
                row_oac = row[oac_cols]
                return row_oac.idxmax().replace('site_lsoa_oac_', '')

            oac_df['Demographic_Profile'] = oac_df.apply(get_dominant_oac, axis=1)
            valid_groups = oac_df['Demographic_Profile'].value_counts()
            valid_groups = valid_groups[valid_groups > 10].index.tolist()
            oac_df = oac_df[oac_df['Demographic_Profile'].isin(valid_groups)]
            
            if not oac_df.empty:
                order = oac_df.groupby('Demographic_Profile')['[REDACTED_BY_SCRIPT]'].median().sort_values().index
                plt.figure(figsize=(12, 6))
                sns.violinplot(data=oac_df, x='Demographic_Profile', y='[REDACTED_BY_SCRIPT]', order=order, palette='plasma')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Inefficient Sprawl
        if 'solar_site_area_sqm' in solar_focus.columns and '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            density_df = solar_focus[(solar_focus['solar_site_area_sqm'] > 0) & (solar_focus['[REDACTED_BY_SCRIPT]'] > 0)].copy()
            density_df['[REDACTED_BY_SCRIPT]'] = (density_df['[REDACTED_BY_SCRIPT]'] * 1_000_000) / density_df['solar_site_area_sqm']
            density_df = density_df[density_df['[REDACTED_BY_SCRIPT]'] < 100] 
            
            if not density_df.empty:
                sns.scatterplot(data=density_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='darkcyan')
                if len(density_df) > 1:
                    sns.regplot(data=density_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 11. Phase IX: Grid Politics & Legacy Infrastructure
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Grid Monopoly
        dno_df = solar_focus.copy()
        def get_dno_region(row):
            if row.get('dno_source_ukpn', 0) == 1: return 'UKPN (South/East)'
            elif row.get('dno_source_nged', 0) == 1: return '[REDACTED_BY_SCRIPT]'
            else: return '[REDACTED_BY_SCRIPT]'

        dno_df['DNO_Provider'] = dno_df.apply(get_dno_region, axis=1)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=dno_df, x='DNO_Provider', y='[REDACTED_BY_SCRIPT]', palette='viridis')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('[REDACTED_BY_SCRIPT]')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # ---------------------------------------------------------
    # 12. Phase X: Technical Engineering & Admin Chaos
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Fault Level Floor
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            fault_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] > 0]
            fault_df = fault_df[fault_df['[REDACTED_BY_SCRIPT]'] < 50]
            if not fault_df.empty:
                sns.scatterplot(data=fault_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='indigo')
                if len(fault_df) > 1:
                    sns.regplot(data=fault_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Stewardship Conflict
        if 'cs_on_site_bool' in solar_focus.columns:
            plt.figure(figsize=(8, 6))
            stewardship_df = solar_focus.copy()
            stewardship_df['Has_Stewardship'] = stewardship_df['cs_on_site_bool'].apply(lambda x: 'Yes (Conflict)' if x == 1 else 'No (Clean)')
            sns.boxplot(data=stewardship_df, x='Has_Stewardship', y='[REDACTED_BY_SCRIPT]', palette='Pastel1')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 13. Phase XI: Systemic Inertia & Political Cycles
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # B. Election Paralysis
        if 'submission_year' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            cycle_df = solar_focus[solar_focus['submission_year'].notna()].copy()
            cycle_df['submission_year'] = cycle_df['submission_year'].astype(int)
            sns.boxplot(data=cycle_df, x='submission_year', y='[REDACTED_BY_SCRIPT]', palette='coolwarm')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('Year of Submission')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 14. Phase XII: XGBoost Feature Importance
    # ---------------------------------------------------------
    if not solar_focus.empty and len(solar_focus) > 20:
        model_df = solar_focus.copy()
        features = [
            '[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm', 
            'nearby_legacy_count', '[REDACTED_BY_SCRIPT]', 
            '[REDACTED_BY_SCRIPT]', 'submission_month', 'submission_year', 
            '[REDACTED_BY_SCRIPT]', 'site_lsoa_ruc_rural_score', 
            '[REDACTED_BY_SCRIPT]', 'lpa_withdrawal_rate', 
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
            'knn_count_solar', 'cs_on_site_bool'
        ]
        # Only keep features present in df
        features = [f for f in features if f in model_df.columns]
        
        model_df = model_df.dropna(subset=features + ['[REDACTED_BY_SCRIPT]'])
        
        if not model_df.empty:
            X_xgb = model_df[features]
            y_xgb = model_df['[REDACTED_BY_SCRIPT]']
            
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
            model.fit(X_xgb, y_xgb)
            
            importance = model.get_booster().get_score(importance_type='gain')
            importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])
            importance_df = importance_df.sort_values(by='Gain', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='Gain', y='Feature', palette='magma')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('Predictive Feature')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 15. Phase XIII: Step-Change Drivers
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Ecological Calendar
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns and 'submission_month' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            calendar_df = solar_focus.copy()
            calendar_df['Submission_Season'] = calendar_df['submission_month'].apply(
                lambda m: '[REDACTED_BY_SCRIPT]' if 3 <= m <= 8 else '[REDACTED_BY_SCRIPT]'
            )
            calendar_df['Has_Constraints'] = calendar_df['[REDACTED_BY_SCRIPT]'] > 0
            sns.boxplot(data=calendar_df, x='Has_Constraints', y='[REDACTED_BY_SCRIPT]', hue='Submission_Season', palette='Set1')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.legend(title='Submission Timing')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. Engineering Mismatch
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns and '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            eng_df = solar_focus[(solar_focus['[REDACTED_BY_SCRIPT]'] > 0) & (solar_focus['[REDACTED_BY_SCRIPT]'] > 0)].copy()
            eng_df['Engineering_Ratio'] = eng_df['[REDACTED_BY_SCRIPT]'] / eng_df['[REDACTED_BY_SCRIPT]']
            eng_df = eng_df[eng_df['Engineering_Ratio'] < 5] 
            
            if not eng_df.empty:
                sns.scatterplot(data=eng_df, x='Engineering_Ratio', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='teal')
                if len(eng_df) > 5:
                    sns.regplot(data=eng_df, x='Engineering_Ratio', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red', lowess=True)
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 19. Phase XVII: Visual Amenity
    # ---------------------------------------------------------
    solar_gm = solar_focus[solar_focus["[REDACTED_BY_SCRIPT]"] == 1]
    if not solar_gm.empty:
        # B. Beauty Contest
        if 'aonb_is_within' in solar_gm.columns and 'np_is_within' in solar_gm.columns:
            plt.figure(figsize=(10, 6))
            def get_landscape_type(row):
                if row['np_is_within'] == 1: return '[REDACTED_BY_SCRIPT]'
                elif row['aonb_is_within'] == 1: return 'AONB (High)'
                else: return '[REDACTED_BY_SCRIPT]'
                
            solar_gm_copy = solar_gm.copy()
            solar_gm_copy['Landscape_Type'] = solar_gm_copy.apply(get_landscape_type, axis=1)
            order = ['[REDACTED_BY_SCRIPT]', 'AONB (High)', '[REDACTED_BY_SCRIPT]']
            
            sns.violinplot(data=solar_gm_copy, x='Landscape_Type', y='[REDACTED_BY_SCRIPT]', order=order, palette='Spectral')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. Goldilocks Field
        if 'alc_grade_at_site' in solar_gm.columns:
            plt.figure(figsize=(12, 6))
            valid_grades = ['Grade 1', 'Grade 2', 'Grade 3a', 'Grade 3b', 'Grade 4', 'Grade 5']
            alc_df = solar_gm[solar_gm['alc_grade_at_site'].isin(valid_grades)]
            order = ['Grade 1', 'Grade 2', 'Grade 3a', 'Grade 3b', 'Grade 4', 'Grade 5']
            
            if not alc_df.empty:
                sns.boxplot(data=alc_df, x='alc_grade_at_site', y='[REDACTED_BY_SCRIPT]', order=order, palette='YlOrBr')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('ALC Grade')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 26. Phase XXIV: Co-Location Variance
    # ---------------------------------------------------------
    if not solar_focus.empty and '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
        plt.figure(figsize=(10, 6))
        bess_df = solar_focus.copy()
        bess_df['Project_Type'] = bess_df['[REDACTED_BY_SCRIPT]'].apply(
            lambda x: '[REDACTED_BY_SCRIPT]' if x > 5 else 'Solar Only'
        )
        sns.boxenplot(data=bess_df, x='Project_Type', y='[REDACTED_BY_SCRIPT]', palette='magma')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('[REDACTED_BY_SCRIPT]')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()


# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================

# Define Capacity Bins
capacity_bins = [
    (0, 1),
    (1, 5),
    (5, 10),
    (10, 15),
    (15, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 100),
    (0, 100) # Aggregate for reference
]

print("[REDACTED_BY_SCRIPT]")

for min_mw, max_mw in capacity_bins:
    label = f"{min_mw}-{max_mw}MW"
    
    # Filter Data for this Bin
    if min_mw == 0 and max_mw == 100:
        label = "All_0-100MW"
        subset_df = df[(df['[REDACTED_BY_SCRIPT]'] > 0) & (df['[REDACTED_BY_SCRIPT]'] <= 100)].copy()
    else:
        # Inclusive of max, exclusive of min (except 0)
        if min_mw == 0:
             subset_df = df[(df['[REDACTED_BY_SCRIPT]'] >= 0) & (df['[REDACTED_BY_SCRIPT]'] <= max_mw)].copy()
        else:
             subset_df = df[(df['[REDACTED_BY_SCRIPT]'] > min_mw) & (df['[REDACTED_BY_SCRIPT]'] <= max_mw)].copy()
    
    # Run Analysis
    run_stratified_analysis(subset_df, label)

# filepath: c:\Users\brand\Desktop\renewables\data\artifacts\model_ready\test_affect.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import xgboost as xgb

# Load the datasets
X = pd.read_csv('[REDACTED_BY_SCRIPT]')
y = pd.read_csv('[REDACTED_BY_SCRIPT]')

# Merge the features (X) and target (y) on 'amaryllis_id'
df = pd.merge(X, y, on='amaryllis_id')

# Drop rows where the target '[REDACTED_BY_SCRIPT]' is missing
df = df.dropna(subset=['[REDACTED_BY_SCRIPT]'])

# Set a consistent style for the plots
sns.set_theme(style="whitegrid")

def run_stratified_analysis(df, range_label):
    """
    Runs the full suite of analysis on a specific subset of data (stratified by capacity).
    """
    # Create a directory to save the graphs for this specific range
    output_dir = f'[REDACTED_BY_SCRIPT]'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    if len(df) < 5:
        print(f"[REDACTED_BY_SCRIPT]")
        return

    # ---------------------------------------------------------
    # 1. Primary Analysis: Solar Specific Duration vs. Capacity
    # ---------------------------------------------------------
    # Filter for Solar Technology (Type 21)
    # Note: df is already filtered by capacity range in the main loop
    solar_df = df[df['technology_type'] == 21]

    if not solar_df.empty:
        # Plot A: All Solar
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=solar_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.6, color='orange')
        if len(solar_df) > 1:
            sns.regplot(data=solar_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('[REDACTED_BY_SCRIPT]')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

        # Plot B: Ground-Mounted Solar Only (Mounting Type 1)
        gm_solar_df = solar_df[solar_df['[REDACTED_BY_SCRIPT]'] == 1]

        if not gm_solar_df.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=gm_solar_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.6, color='brown')
            if len(gm_solar_df) > 1:
                sns.regplot(data=gm_solar_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # B. Duration by Technology Type (if available)
    tech_col = next((col for col in ['technology', 'technology_type', 'type'] if col in df.columns), None)
    if tech_col:
        plt.figure(figsize=(10, 6))
        order = df.groupby(tech_col)['[REDACTED_BY_SCRIPT]'].median().sort_values().index
        sns.boxplot(data=df, x=tech_col, y='[REDACTED_BY_SCRIPT]', order=order)
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # C. Top 15 Feature Correlations (Targeted Analysis)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    # Drop columns with 0 variance to avoid warnings
    numeric_df = numeric_df.loc[:, numeric_df.var() > 0]
    
    if not numeric_df.empty and '[REDACTED_BY_SCRIPT]' in numeric_df.columns:
        corr_series = numeric_df.corrwith(df['[REDACTED_BY_SCRIPT]'])
        # Get top 15 features by absolute correlation
        top_corr = corr_series.abs().sort_values(ascending=False).head(16) 
        top_corr = corr_series[top_corr.index].drop('[REDACTED_BY_SCRIPT]', errors='ignore') 

        if not top_corr.empty:
            sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # ---------------------------------------------------------
    # 3. Variance Hypothesis Testing
    # ---------------------------------------------------------

    # A. The "Grid Friction" Hypothesis
    if '[REDACTED_BY_SCRIPT]' in df.columns:
        plt.figure(figsize=(10, 6))
        grid_df = df[df['[REDACTED_BY_SCRIPT]'] < 500] 
        if not grid_df.empty:
            sns.scatterplot(data=grid_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', hue=(df['technology_type'] == 21), palette={True: 'orange', False: 'grey'}, alpha=0.5)
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.legend(title='Is Solar?')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # B. The "NIMBY" Hypothesis
    if '[REDACTED_BY_SCRIPT]' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('NIMBY Risk Score')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # C. The "LPA Bottleneck" Hypothesis
    if '[REDACTED_BY_SCRIPT]' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5)
        max_val = df['[REDACTED_BY_SCRIPT]'].max()
        if pd.notna(max_val):
            plt.plot([0, max_val], [0, max_val], 'r--', label='Average Pace')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('[REDACTED_BY_SCRIPT]')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # ---------------------------------------------------------
    # 4. Deep Dive: Constraint Analysis
    # ---------------------------------------------------------
    focus_df = df[df['technology_type'].isin([21, 4])]

    if not focus_df.empty:
        # A. The "Food vs. Fuel" Conflict
        if 'alc_is_bmv_at_site' in focus_df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=focus_df, x='alc_is_bmv_at_site', y='[REDACTED_BY_SCRIPT]', palette='Set2')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. The "Ecological Drag"
        if 'sssi_dist_to_nearest_m' in focus_df.columns:
            plt.figure(figsize=(10, 6))
            eco_df = focus_df[focus_df['sssi_dist_to_nearest_m'] < 2000]
            if not eco_df.empty:
                sns.scatterplot(data=eco_df, x='sssi_dist_to_nearest_m', y='[REDACTED_BY_SCRIPT]', hue='technology_type', alpha=0.7)
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                if len(eco_df) > 1:
                    sns.regplot(data=eco_df, x='sssi_dist_to_nearest_m', y='[REDACTED_BY_SCRIPT]', scatter=False, color='green', line_kws={"linestyle": "--"})
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. The "Urban Fringe" Battleground
        if '[REDACTED_BY_SCRIPT]' in focus_df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=focus_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='purple')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 5. Phase II & III: Administrative & Technical Drivers
    # ---------------------------------------------------------
    solar_focus = df[df['technology_type'] == 21]

    if not solar_focus.empty:
        # A. The "LPA Lottery" (Tiers)
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            lpa_clean = solar_focus[
                (solar_focus['[REDACTED_BY_SCRIPT]'] > 0) & 
                (solar_focus['[REDACTED_BY_SCRIPT]'] < 730)
            ].copy()
            
            if not lpa_clean.empty:
                def classify_lpa(days):
                    if days <= 112: return 'Fast (<16 wks)'
                    elif days <= 250: return 'Moderate'
                    else: return 'Sluggish (>36 wks)'
                    
                lpa_clean['LPA_Speed_Tier'] = lpa_clean['[REDACTED_BY_SCRIPT]'].apply(classify_lpa)
                order = ['Fast (<16 wks)', 'Moderate', 'Sluggish (>36 wks)']
                
                sns.boxplot(data=lpa_clean, x='LPA_Speed_Tier', y='[REDACTED_BY_SCRIPT]', order=order, palette='viridis')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. The "Deep Rural Dividend"
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns and '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            dist_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] <= 15]
            if not dist_df.empty:
                sns.scatterplot(
                    data=dist_df, 
                    x='[REDACTED_BY_SCRIPT]', 
                    y='[REDACTED_BY_SCRIPT]', 
                    hue='[REDACTED_BY_SCRIPT]',
                    palette='Greens',
                    alpha=0.7
                )
                if len(dist_df) > 1:
                    sns.regplot(data=dist_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.legend(title='Ag Area % (1km)', loc='upper right')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. The "Site Sprawl" (Log Scale)
        if 'solar_site_area_sqm' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            area_df = solar_focus[solar_focus['solar_site_area_sqm'] > 1000]
            if not area_df.empty:
                sns.scatterplot(data=area_df, x='solar_site_area_sqm', y='[REDACTED_BY_SCRIPT]', alpha=0.4, color='darkgreen')
                plt.xscale('log')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 6. Phase IV: Advanced Hypothesis Testing
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. The "Voltage Ceiling"
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            valid_voltages = [11.0, 33.0, 66.0, 132.0]
            voltage_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'].isin(valid_voltages)].copy()
            if not voltage_df.empty:
                sns.boxplot(data=voltage_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', palette='magma')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. The "Wealth Barrier"
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=solar_focus, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='gold')
            if len(solar_focus) > 1:
                sns.regplot(data=solar_focus, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='black')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. The "Grid Queue"
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            constraint_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] <= 10]
            if not constraint_df.empty:
                sns.violinplot(data=constraint_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', palette='cool', inner="quartile")
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 7. Phase V: Contextual & Environmental Drivers
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Cumulative Fatigue
        if 'knn_count_solar' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            neighbor_df = solar_focus[solar_focus['knn_count_solar'] <= 20]
            if not neighbor_df.empty:
                sns.boxplot(data=neighbor_df, x='knn_count_solar', y='[REDACTED_BY_SCRIPT]', palette='Blues')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Hostile Environment
        if 'lpa_approval_rate_cps2' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=solar_focus, x='lpa_approval_rate_cps2', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='crimson')
            if len(solar_focus) > 1:
                sns.regplot(data=solar_focus, x='lpa_approval_rate_cps2', y='[REDACTED_BY_SCRIPT]', scatter=False, color='black')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('LPA Approval Rate')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 8. Phase VI: Complexity, History, and Capacity
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Constraint Thicket
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            constraint_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] <= 15]
            if not constraint_df.empty:
                sns.boxplot(data=constraint_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', palette='Reds')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Baggage Effect
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            def classify_baggage(count):
                if count == 0: return 'Clean History (0)'
                elif count == 1: return 'Resubmission (1)'
                else: return 'Contentious (2+)'
            
            solar_focus_copy = solar_focus.copy()
            solar_focus_copy['Site_Baggage'] = solar_focus_copy['[REDACTED_BY_SCRIPT]'].apply(classify_baggage)
            order = ['Clean History (0)', 'Resubmission (1)', 'Contentious (2+)']
            
            sns.violinplot(data=solar_focus_copy, x='Site_Baggage', y='[REDACTED_BY_SCRIPT]', order=order, palette='muted')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 9. Phase VII: The "Killer" Micro-Drivers
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Linear Obstacle Course
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            crossing_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] <= 10]
            if not crossing_df.empty:
                sns.boxplot(data=crossing_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', palette='coolwarm')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Killer Constraint
        constraint_cols = [c for c in solar_focus.columns if c.startswith('[REDACTED_BY_SCRIPT]')]
        if constraint_cols:
            type_df = solar_focus.copy()
            def get_dominant_constraint(row):
                for col in constraint_cols:
                    if row[col] == 1:
                        return col.replace('[REDACTED_BY_SCRIPT]', '')
                return 'None'

            type_df['Dominant_Constraint'] = type_df.apply(get_dominant_constraint, axis=1)
            valid_types = type_df['Dominant_Constraint'].value_counts()
            valid_types = valid_types[valid_types > 5].index.tolist()
            type_df = type_df[(type_df['Dominant_Constraint'].isin(valid_types)) & (type_df['Dominant_Constraint'] != 'None')]
            
            if not type_df.empty:
                order = type_df.groupby('Dominant_Constraint')['[REDACTED_BY_SCRIPT]'].median().sort_values().index
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=type_df, x='Dominant_Constraint', y='[REDACTED_BY_SCRIPT]', order=order, palette='Set2')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xticks(rotation=45)
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 10. Phase VIII: Human Factors & Efficiency
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Digital Activism
        oac_cols = [c for c in solar_focus.columns if c.startswith('site_lsoa_oac_')]
        if oac_cols:
            oac_df = solar_focus.copy()
            def get_dominant_oac(row):
                row_oac = row[oac_cols]
                return row_oac.idxmax().replace('site_lsoa_oac_', '')

            oac_df['Demographic_Profile'] = oac_df.apply(get_dominant_oac, axis=1)
            valid_groups = oac_df['Demographic_Profile'].value_counts()
            valid_groups = valid_groups[valid_groups > 10].index.tolist()
            oac_df = oac_df[oac_df['Demographic_Profile'].isin(valid_groups)]
            
            if not oac_df.empty:
                order = oac_df.groupby('Demographic_Profile')['[REDACTED_BY_SCRIPT]'].median().sort_values().index
                plt.figure(figsize=(12, 6))
                sns.violinplot(data=oac_df, x='Demographic_Profile', y='[REDACTED_BY_SCRIPT]', order=order, palette='plasma')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Inefficient Sprawl
        if 'solar_site_area_sqm' in solar_focus.columns and '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            density_df = solar_focus[(solar_focus['solar_site_area_sqm'] > 0) & (solar_focus['[REDACTED_BY_SCRIPT]'] > 0)].copy()
            density_df['[REDACTED_BY_SCRIPT]'] = (density_df['[REDACTED_BY_SCRIPT]'] * 1_000_000) / density_df['solar_site_area_sqm']
            density_df = density_df[density_df['[REDACTED_BY_SCRIPT]'] < 100] 
            
            if not density_df.empty:
                sns.scatterplot(data=density_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='darkcyan')
                if len(density_df) > 1:
                    sns.regplot(data=density_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 11. Phase IX: Grid Politics & Legacy Infrastructure
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Grid Monopoly
        dno_df = solar_focus.copy()
        def get_dno_region(row):
            if row.get('dno_source_ukpn', 0) == 1: return 'UKPN (South/East)'
            elif row.get('dno_source_nged', 0) == 1: return '[REDACTED_BY_SCRIPT]'
            else: return '[REDACTED_BY_SCRIPT]'

        dno_df['DNO_Provider'] = dno_df.apply(get_dno_region, axis=1)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=dno_df, x='DNO_Provider', y='[REDACTED_BY_SCRIPT]', palette='viridis')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('[REDACTED_BY_SCRIPT]')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()

    # ---------------------------------------------------------
    # 12. Phase X: Technical Engineering & Admin Chaos
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Fault Level Floor
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            fault_df = solar_focus[solar_focus['[REDACTED_BY_SCRIPT]'] > 0]
            fault_df = fault_df[fault_df['[REDACTED_BY_SCRIPT]'] < 50]
            if not fault_df.empty:
                sns.scatterplot(data=fault_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='indigo')
                if len(fault_df) > 1:
                    sns.regplot(data=fault_df, x='[REDACTED_BY_SCRIPT]', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # B. Stewardship Conflict
        if 'cs_on_site_bool' in solar_focus.columns:
            plt.figure(figsize=(8, 6))
            stewardship_df = solar_focus.copy()
            stewardship_df['Has_Stewardship'] = stewardship_df['cs_on_site_bool'].apply(lambda x: 'Yes (Conflict)' if x == 1 else 'No (Clean)')
            sns.boxplot(data=stewardship_df, x='Has_Stewardship', y='[REDACTED_BY_SCRIPT]', palette='Pastel1')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 13. Phase XI: Systemic Inertia & Political Cycles
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # B. Election Paralysis
        if 'submission_year' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            cycle_df = solar_focus[solar_focus['submission_year'].notna()].copy()
            cycle_df['submission_year'] = cycle_df['submission_year'].astype(int)
            sns.boxplot(data=cycle_df, x='submission_year', y='[REDACTED_BY_SCRIPT]', palette='coolwarm')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('Year of Submission')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 14. Phase XII: XGBoost Feature Importance
    # ---------------------------------------------------------
    if not solar_focus.empty and len(solar_focus) > 20:
        model_df = solar_focus.copy()
        features = [
            '[REDACTED_BY_SCRIPT]', 'solar_site_area_sqm', 
            'nearby_legacy_count', '[REDACTED_BY_SCRIPT]', 
            '[REDACTED_BY_SCRIPT]', 'submission_month', 'submission_year', 
            '[REDACTED_BY_SCRIPT]', 'site_lsoa_ruc_rural_score', 
            '[REDACTED_BY_SCRIPT]', 'lpa_withdrawal_rate', 
            '[REDACTED_BY_SCRIPT]', '[REDACTED_BY_SCRIPT]', 
            'knn_count_solar', 'cs_on_site_bool'
        ]
        # Only keep features present in df
        features = [f for f in features if f in model_df.columns]
        
        model_df = model_df.dropna(subset=features + ['[REDACTED_BY_SCRIPT]'])
        
        if not model_df.empty:
            X_xgb = model_df[features]
            y_xgb = model_df['[REDACTED_BY_SCRIPT]']
            
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
            model.fit(X_xgb, y_xgb)
            
            importance = model.get_booster().get_score(importance_type='gain')
            importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])
            importance_df = importance_df.sort_values(by='Gain', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='Gain', y='Feature', palette='magma')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('Predictive Feature')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 15. Phase XIII: Step-Change Drivers
    # ---------------------------------------------------------
    if not solar_focus.empty:
        # A. Ecological Calendar
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns and 'submission_month' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            calendar_df = solar_focus.copy()
            calendar_df['Submission_Season'] = calendar_df['submission_month'].apply(
                lambda m: '[REDACTED_BY_SCRIPT]' if 3 <= m <= 8 else '[REDACTED_BY_SCRIPT]'
            )
            calendar_df['Has_Constraints'] = calendar_df['[REDACTED_BY_SCRIPT]'] > 0
            sns.boxplot(data=calendar_df, x='Has_Constraints', y='[REDACTED_BY_SCRIPT]', hue='Submission_Season', palette='Set1')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.legend(title='Submission Timing')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. Engineering Mismatch
        if '[REDACTED_BY_SCRIPT]' in solar_focus.columns and '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
            plt.figure(figsize=(10, 6))
            eng_df = solar_focus[(solar_focus['[REDACTED_BY_SCRIPT]'] > 0) & (solar_focus['[REDACTED_BY_SCRIPT]'] > 0)].copy()
            eng_df['Engineering_Ratio'] = eng_df['[REDACTED_BY_SCRIPT]'] / eng_df['[REDACTED_BY_SCRIPT]']
            eng_df = eng_df[eng_df['Engineering_Ratio'] < 5] 
            
            if not eng_df.empty:
                sns.scatterplot(data=eng_df, x='Engineering_Ratio', y='[REDACTED_BY_SCRIPT]', alpha=0.5, color='teal')
                if len(eng_df) > 5:
                    sns.regplot(data=eng_df, x='Engineering_Ratio', y='[REDACTED_BY_SCRIPT]', scatter=False, color='red', lowess=True)
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('[REDACTED_BY_SCRIPT]')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 19. Phase XVII: Visual Amenity
    # ---------------------------------------------------------
    solar_gm = solar_focus[solar_focus["[REDACTED_BY_SCRIPT]"] == 1]
    if not solar_gm.empty:
        # B. Beauty Contest
        if 'aonb_is_within' in solar_gm.columns and 'np_is_within' in solar_gm.columns:
            plt.figure(figsize=(10, 6))
            def get_landscape_type(row):
                if row['np_is_within'] == 1: return '[REDACTED_BY_SCRIPT]'
                elif row['aonb_is_within'] == 1: return 'AONB (High)'
                else: return '[REDACTED_BY_SCRIPT]'
                
            solar_gm_copy = solar_gm.copy()
            solar_gm_copy['Landscape_Type'] = solar_gm_copy.apply(get_landscape_type, axis=1)
            order = ['[REDACTED_BY_SCRIPT]', 'AONB (High)', '[REDACTED_BY_SCRIPT]']
            
            sns.violinplot(data=solar_gm_copy, x='Landscape_Type', y='[REDACTED_BY_SCRIPT]', order=order, palette='Spectral')
            plt.title(f'[REDACTED_BY_SCRIPT]')
            plt.xlabel('[REDACTED_BY_SCRIPT]')
            plt.ylabel('[REDACTED_BY_SCRIPT]')
            plt.tight_layout()
            plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

        # C. Goldilocks Field
        if 'alc_grade_at_site' in solar_gm.columns:
            plt.figure(figsize=(12, 6))
            valid_grades = ['Grade 1', 'Grade 2', 'Grade 3a', 'Grade 3b', 'Grade 4', 'Grade 5']
            alc_df = solar_gm[solar_gm['alc_grade_at_site'].isin(valid_grades)]
            order = ['Grade 1', 'Grade 2', 'Grade 3a', 'Grade 3b', 'Grade 4', 'Grade 5']
            
            if not alc_df.empty:
                sns.boxplot(data=alc_df, x='alc_grade_at_site', y='[REDACTED_BY_SCRIPT]', order=order, palette='YlOrBr')
                plt.title(f'[REDACTED_BY_SCRIPT]')
                plt.xlabel('ALC Grade')
                plt.ylabel('[REDACTED_BY_SCRIPT]')
                plt.tight_layout()
                plt.savefig(f'[REDACTED_BY_SCRIPT]')
            plt.close()

    # ---------------------------------------------------------
    # 26. Phase XXIV: Co-Location Variance
    # ---------------------------------------------------------
    if not solar_focus.empty and '[REDACTED_BY_SCRIPT]' in solar_focus.columns:
        plt.figure(figsize=(10, 6))
        bess_df = solar_focus.copy()
        bess_df['Project_Type'] = bess_df['[REDACTED_BY_SCRIPT]'].apply(
            lambda x: '[REDACTED_BY_SCRIPT]' if x > 5 else 'Solar Only'
        )
        sns.boxenplot(data=bess_df, x='Project_Type', y='[REDACTED_BY_SCRIPT]', palette='magma')
        plt.title(f'[REDACTED_BY_SCRIPT]')
        plt.xlabel('[REDACTED_BY_SCRIPT]')
        plt.ylabel('[REDACTED_BY_SCRIPT]')
        plt.tight_layout()
        plt.savefig(f'[REDACTED_BY_SCRIPT]')
        plt.close()


# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================

# Define Capacity Bins
capacity_bins = [
    (0, 1),
    (1, 5),
    (5, 10),
    (10, 15),
    (15, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 100),
    (0, 100) # Aggregate for reference
]

print("[REDACTED_BY_SCRIPT]")

for min_mw, max_mw in capacity_bins:
    label = f"{min_mw}-{max_mw}MW"
    
    # Filter Data for this Bin
    if min_mw == 0 and max_mw == 100:
        label = "All_0-100MW"
        subset_df = df[(df['[REDACTED_BY_SCRIPT]'] > 0) & (df['[REDACTED_BY_SCRIPT]'] <= 100)].copy()
    else:
        # Inclusive of max, exclusive of min (except 0)
        if min_mw == 0:
             subset_df = df[(df['[REDACTED_BY_SCRIPT]'] >= 0) & (df['[REDACTED_BY_SCRIPT]'] <= max_mw)].copy()
        else:
             subset_df = df[(df['[REDACTED_BY_SCRIPT]'] > min_mw) & (df['[REDACTED_BY_SCRIPT]'] <= max_mw)].copy()
    
    # Run Analysis
    run_stratified_analysis(subset_df, label)
