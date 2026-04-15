import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

data_csv_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/analysis_ready_data.csv"
stats_csv_path = "/data4/Agri/yukaijie/DeepEco/data/Codes/comprehensive_statistics_summary_raw.csv"
output_dir = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/Codes/pics_paper/All_Raw_Regressions"

def main():
    if not os.path.exists(data_csv_path):
        return

    df = pd.read_csv(data_csv_path)

    if 'dda' not in df.columns and 'temperature' in df.columns:
        df['dda'] = np.maximum(df['temperature'] - 10, 0) * 365 

    env_factors = [
        'elevation', 'temperature', 'bio4_temp_seasonality', 
        'bio5_max_temp', 'bio6_min_temp', 'precipitation', 
        'bio15_precip_seasonality', 'ndvi_mean', 'human_footprint', 'dda'
    ]
    
    phenotypes_ind = ['lightness', 'pattern_complexity', 'phenotypic_disparity', 'dino_pc1']
    phenotypes_grid = ['functional_beta_diversity']

    grid_level_df = df.drop_duplicates(subset=['spatial_grid'])

    completed_pairs = set()
    if os.path.exists(stats_csv_path):
        existing_stats_df = pd.read_csv(stats_csv_path)
        if 'Phenotype' in existing_stats_df.columns and 'Environment' in existing_stats_df.columns:
            completed_pairs = set(zip(existing_stats_df['Phenotype'], existing_stats_df['Environment']))
    else:
        existing_stats_df = pd.DataFrame()

    print("\n>>> Stage 1: Calculating Raw Spatial Statistics (GAMM & OLS)...")
    new_results = []
    spatial_formula = " + bs(longitude, df=4) + bs(latitude, df=4)"

    for p in phenotypes_ind:
        if p not in df.columns:
            continue
        for e in tqdm(env_factors, desc=f"Modeling {p}", ncols=100, leave=False):
            if e not in df.columns:
                continue
            if (p, e) in completed_pairs:
                continue
            try:
                f = f"{p} ~ {e}{spatial_formula}"
                mod = smf.mixedlm(f, df, groups=df["spatial_grid"]).fit(disp=False)
                coef = mod.params[e]
                pval = mod.pvalues[e]
                new_results.append({'Phenotype': p, 'Environment': e, 'Model': 'Spatial GAMM', 'Beta': coef, 'P_Value': pval})
            except:
                pass

    for p in phenotypes_grid:
        if p not in grid_level_df.columns:
            continue
        for e in tqdm(env_factors, desc=f"Modeling {p}", ncols=100, leave=False):
            if e not in grid_level_df.columns:
                continue
            if (p, e) in completed_pairs:
                continue
            try:
                f = f"{p} ~ {e}{spatial_formula}"
                mod = smf.ols(f, data=grid_level_df).fit()
                coef = mod.params[e]
                pval = mod.pvalues[e]
                new_results.append({'Phenotype': p, 'Environment': e, 'Model': 'Spatial OLS', 'Beta': coef, 'P_Value': pval})
            except:
                pass

    if new_results:
        new_df = pd.DataFrame(new_results)
        if not existing_stats_df.empty:
            stats_df = pd.concat([existing_stats_df, new_df], ignore_index=True)
        else:
            stats_df = new_df
        stats_df.to_csv(stats_csv_path, index=False)
        print(f">>> Updated Statistics calculated and appended to: {stats_csv_path}")
    else:
        stats_df = existing_stats_df
        print(f">>> All combinations already calculated. Loaded existing stats from: {stats_csv_path}")

    print("\n>>> Stage 2: Generating all raw regression plots...")
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="ticks", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']

    color_palettes = [
        ('#31688E', '#D3515B'), ('#35B779', '#440154'), 
        ('#FDE725', '#D3515B'), ('#440154', '#35B779'),
        ('#E76F51', '#264653'), ('#2A9D8F', '#E9C46A'),
        ('#9B5DE5', '#F15BB5'), ('#00BBF9', '#F15BB5')
    ]

    if 'Abs_Beta' not in stats_df.columns:
        stats_df['Abs_Beta'] = stats_df['Beta'].abs()
    stats_df = stats_df.sort_values(by='Abs_Beta', ascending=False)

    for index, row in tqdm(stats_df.iterrows(), total=len(stats_df), desc="Plotting", ncols=100):
        phenotype = row['Phenotype']
        environment = row['Environment']
        beta = row['Beta']
        p_val = row['P_Value']

        y_col = phenotype
        x_col = environment

        save_name = f"{phenotype}_VS_{environment}.png"
        save_path = os.path.join(output_dir, save_name)
        
        if os.path.exists(save_path):
            continue

        if y_col not in df.columns or x_col not in df.columns:
            continue

        is_grid_level = (phenotype == 'functional_beta_diversity')
        
        plot_df = grid_level_df if is_grid_level else df.sample(min(10000, len(df)), random_state=42)
        plot_df = plot_df.dropna(subset=[x_col, y_col])
        
        if len(plot_df) < 10:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        color_idx = index % len(color_palettes)
        scatter_color, line_color = color_palettes[color_idx]

        if is_grid_level:
            scatter_kws = {'alpha': 0.6, 's': 50, 'color': scatter_color, 'linewidths': 0.5, 'edgecolor': 'white'}
        elif scatter_color == '#FDE725':
            scatter_kws = {'alpha': 0.15, 's': 20, 'color': scatter_color, 'edgecolors': 'black', 'linewidths': 0.2}
        else:
            scatter_kws = {'alpha': 0.15, 's': 20, 'color': scatter_color, 'linewidths': 0}

        sns.regplot(data=plot_df, x=x_col, y=y_col, ax=ax, 
                    scatter_kws=scatter_kws, line_kws={'color': line_color, 'linewidth': 3})

        xlab = environment.replace('_', ' ').title()
        ylab = phenotype.replace('_', ' ').title()

        ax.set_xlabel(f'{xlab}', fontweight='medium')
        ax.set_ylabel(f'{ylab}', fontweight='medium')
        ax.set_title(f'{ylab} vs {xlab}', loc='left', fontweight='bold', pad=15)

        p_str = "P < 0.001" if pd.notna(p_val) and p_val < 0.001 else f"P = {p_val:.3e}" if pd.notna(p_val) else "P = N/A"
        
        if pd.notna(beta):
            if abs(beta) < 0.001 or abs(beta) > 1000:
                beta_str = f"$\\beta$ = {beta:.3e}"
            else:
                beta_str = f"$\\beta$ = {beta:.3f}"
        else:
            beta_str = "$\\beta$ = N/A"
        
        ax.text(0.05, 0.95, f"{beta_str}\n{p_str}", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='lightgray'))

        sns.despine(trim=True, offset=5)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"\n>>> Operations complete. All raw regression plots are saved in: {output_dir}")

if __name__ == "__main__":
    main()