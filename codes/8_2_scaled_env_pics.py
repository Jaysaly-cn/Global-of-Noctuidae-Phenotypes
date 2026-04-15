import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

data_csv_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/analysis_ready_data.csv"
stats_csv_path = "/data4/Agri/yukaijie/DeepEco/data/Codes/comprehensive_statistics_summary_scaled.csv"
output_dir = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/Codes/pics_paper/All_Scaled_Regressions_TrueGAMM"

def clean_label(name):
    name = re.sub(r'^bio\d+_', '', name)
    return name.replace('_', ' ').title()

def main():
    if not os.path.exists(data_csv_path):
        return

    df = pd.read_csv(data_csv_path)

    if 'dda' not in df.columns and 'temperature' in df.columns:
        df['dda'] = np.maximum(df['temperature'] - 10, 0) * 365 

    if 'dino_pc1' in df.columns:
        df['dino_pc1_mutation'] = np.abs(df['dino_pc1'] - df['dino_pc1'].mean())

    env_factors = [
        'elevation', 'temperature', 'bio4_temp_seasonality', 
        'bio5_max_temp', 'bio6_min_temp', 'precipitation', 
        'bio15_precip_seasonality', 'ndvi_mean', 'human_footprint', 'dda'
    ]
    
    phenotypes_ind = ['lightness', 'pattern_complexity', 'phenotypic_disparity', 'dino_pc1_mutation']
    phenotypes_grid = ['functional_beta_diversity']
    
    log_transform_cols = [
        'human_footprint', 'precipitation', 
        'pattern_complexity', 'phenotypic_disparity', 'functional_beta_diversity'
    ]

    all_cols = env_factors + phenotypes_ind + phenotypes_grid
    
    for col in all_cols:
        if col in df.columns:
            if col in log_transform_cols:
                trans_data = np.log1p(np.maximum(df[col], 0))
            else:
                trans_data = df[col]
                
            std_val = trans_data.std()
            if pd.notna(std_val) and std_val != 0:
                df[f'{col}_scaled'] = (trans_data - trans_data.mean()) / std_val
            else:
                df[f'{col}_scaled'] = 0.0

    grid_level_df = df.drop_duplicates(subset=['spatial_grid'])

    completed_pairs = set()
    if os.path.exists(stats_csv_path):
        existing_stats_df = pd.read_csv(stats_csv_path)
        if 'Phenotype' in existing_stats_df.columns and 'Environment' in existing_stats_df.columns:
            completed_pairs = set(zip(existing_stats_df['Phenotype'], existing_stats_df['Environment']))
    else:
        existing_stats_df = pd.DataFrame()

    new_results = []
    spatial_formula = " + bs(longitude, df=4) + bs(latitude, df=4)"

    for p in phenotypes_ind:
        p_scaled = f"{p}_scaled"
        if p_scaled not in df.columns:
            continue
        for e in tqdm(env_factors, desc=f"Modeling {p}", ncols=100, leave=False):
            e_scaled = f"{e}_scaled"
            if e_scaled not in df.columns:
                continue
            if (p, e) in completed_pairs:
                continue
            
            try:
                fit_df = df[[p_scaled, e_scaled, 'longitude', 'latitude', 'spatial_grid']].dropna()
                
                p_up, p_low = fit_df[p_scaled].quantile(0.995), fit_df[p_scaled].quantile(0.005)
                e_up, e_low = fit_df[e_scaled].quantile(0.995), fit_df[e_scaled].quantile(0.005)
                fit_df = fit_df[(fit_df[p_scaled] <= p_up) & (fit_df[p_scaled] >= p_low) &
                                (fit_df[e_scaled] <= e_up) & (fit_df[e_scaled] >= e_low)]

                group_counts = fit_df['spatial_grid'].value_counts()
                valid_groups = group_counts[group_counts >= 3].index
                fit_df = fit_df[fit_df['spatial_grid'].isin(valid_groups)]
                
                if len(fit_df) < 50 or len(valid_groups) < 5:
                    continue

                f = f"{p_scaled} ~ {e_scaled}{spatial_formula}"
                mod = smf.mixedlm(f, fit_df, groups=fit_df["spatial_grid"]).fit(method='lbfgs', maxiter=200, disp=False)
                
                coef = mod.params[e_scaled]
                pval = mod.pvalues[e_scaled]
                new_results.append({'Phenotype': p, 'Environment': e, 'Model': 'Spatial GAMM', 'Beta': coef, 'P_Value': pval})
            except:
                pass

    for p in phenotypes_grid:
        p_scaled = f"{p}_scaled"
        if p_scaled not in grid_level_df.columns:
            continue
        for e in tqdm(env_factors, desc=f"Modeling {p}", ncols=100, leave=False):
            e_scaled = f"{e}_scaled"
            if e_scaled not in grid_level_df.columns:
                continue
            if (p, e) in completed_pairs:
                continue
            
            try:
                fit_df = grid_level_df[[p_scaled, e_scaled, 'longitude', 'latitude']].dropna()
                
                p_up, p_low = fit_df[p_scaled].quantile(0.995), fit_df[p_scaled].quantile(0.005)
                e_up, e_low = fit_df[e_scaled].quantile(0.995), fit_df[e_scaled].quantile(0.005)
                fit_df = fit_df[(fit_df[p_scaled] <= p_up) & (fit_df[p_scaled] >= p_low) &
                                (fit_df[e_scaled] <= e_up) & (fit_df[e_scaled] >= e_low)]

                if len(fit_df) < 20:
                    continue
                    
                f = f"{p_scaled} ~ {e_scaled}{spatial_formula}"
                mod = smf.ols(f, data=fit_df).fit()
                
                coef = mod.params[e_scaled]
                pval = mod.pvalues[e_scaled]
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
    else:
        stats_df = existing_stats_df

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

    print("\n>>> Stage 2: Drawing Exact GAMM Regression Lines...")
    for index, row in tqdm(stats_df.iterrows(), total=len(stats_df), desc="Plotting", ncols=100):
        phenotype = row['Phenotype']
        environment = row['Environment']
        beta = row['Beta']
        p_val = row['P_Value']

        y_col = f"{phenotype}_scaled"
        x_col = f"{environment}_scaled"

        save_name = f"{phenotype}_VS_{environment}_TrueGAMM.png"
        save_path = os.path.join(output_dir, save_name)
        
        if os.path.exists(save_path):
            continue

        if y_col not in df.columns or x_col not in df.columns:
            continue

        is_grid_level = (phenotype == 'functional_beta_diversity')
        
        plot_df = grid_level_df if is_grid_level else df.sample(min(10000, len(df)), random_state=42)
        plot_df = plot_df.dropna(subset=[x_col, y_col])
        
        q_y_u, q_y_l = plot_df[y_col].quantile(0.995), plot_df[y_col].quantile(0.005)
        q_x_u, q_x_l = plot_df[x_col].quantile(0.995), plot_df[x_col].quantile(0.005)
        plot_df = plot_df[(plot_df[y_col] <= q_y_u) & (plot_df[y_col] >= q_y_l) &
                          (plot_df[x_col] <= q_x_u) & (plot_df[x_col] >= q_x_l)]

        if len(plot_df) < 10:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        color_idx = index % len(color_palettes)
        scatter_color, line_color = color_palettes[color_idx]

        if is_grid_level:
            scatter_kws = {'alpha': 0.6, 's': 50, 'color': scatter_color, 'linewidths': 0.5, 'edgecolors': 'white'}
        elif scatter_color == '#FDE725':
            scatter_kws = {'alpha': 0.15, 's': 20, 'color': scatter_color, 'edgecolors': 'black', 'linewidths': 0.2}
        else:
            scatter_kws = {'alpha': 0.15, 's': 20, 'color': scatter_color, 'linewidths': 0}

        ax.scatter(plot_df[x_col], plot_df[y_col], **scatter_kws)

        if pd.notna(beta):
            x_mean = plot_df[x_col].mean()
            y_mean = plot_df[y_col].mean()
            
            intercept = y_mean - beta * x_mean
            x_left, x_right = ax.get_xlim()
            
            x_line = np.array([x_left, x_right])
            y_line = intercept + beta * x_line
            ax.plot(x_line, y_line, color=line_color, linewidth=3.5, linestyle='--', zorder=10)
            
            ax.set_xlim(x_left, x_right)

        xlab_base = clean_label(environment)
        ylab_base = clean_label(phenotype)
        
        xlab_suffix = " (Log-Scaled)" if environment in log_transform_cols else " (Scaled)"
        ylab_suffix = " (Log-Scaled)" if phenotype in log_transform_cols else " (Scaled)"

        ax.set_xlabel(f'{xlab_base}{xlab_suffix}', fontweight='medium')
        ax.set_ylabel(f'{ylab_base}{ylab_suffix}', fontweight='medium')
        ax.set_title(f'{ylab_base} vs {xlab_base}', loc='left', fontweight='bold', pad=15)

        p_str = "P < 0.001" if pd.notna(p_val) and p_val < 0.001 else f"P = {p_val:.3e}" if pd.notna(p_val) else "P = N/A"
        beta_str = f"$\\beta$ = {beta:.3f}" if pd.notna(beta) else "$\\beta$ = N/A"
        
        ax.text(0.05, 0.95, f"{beta_str}\n{p_str}", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='lightgray'))

        sns.despine(trim=True, offset=5)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"\n>>> Operations complete. True GAMM regressions saved in: {output_dir}")

if __name__ == "__main__":
    main()