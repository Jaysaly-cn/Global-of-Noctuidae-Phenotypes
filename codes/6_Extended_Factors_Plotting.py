import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

data_csv_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/analysis_ready_data.csv"
stats_csv_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/comprehensive_statistics_summary.csv"
output_dir = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/Codes/pics_paper"

def main():
    if not os.path.exists(data_csv_path) or not os.path.exists(stats_csv_path):
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_csv_path)
    stats_df = pd.read_csv(stats_csv_path)

    needed_cols = set(stats_df['Phenotype'].tolist() + stats_df['Environment'].tolist())
    for col in needed_cols:
        scaled_col = f"{col}_scaled"
        if col in df.columns and scaled_col not in df.columns:
            std_val = df[col].std()
            if pd.notna(std_val) and std_val != 0:
                df[scaled_col] = (df[col] - df[col].mean()) / std_val
            else:
                df[scaled_col] = 0.0

    grid_level_df = df.drop_duplicates(subset=['spatial_grid'])

    sns.set_theme(style="ticks", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']

    color_palettes = [
        ('#31688E', '#D3515B'), ('#35B779', '#440154'), 
        ('#FDE725', '#D3515B'), ('#440154', '#35B779'),
        ('#E76F51', '#264653'), ('#2A9D8F', '#E9C46A'),
        ('#9B5DE5', '#F15BB5'), ('#00BBF9', '#F15BB5')
    ]

    for index, row in tqdm(stats_df.iterrows(), total=len(stats_df), desc="Generating Regression Plots", ncols=100):
        phenotype = row['Phenotype']
        environment = row['Environment']
        beta = row['Beta']
        p_val = row['P_Value']

        y_col = f"{phenotype}_scaled"
        x_col = f"{environment}_scaled"

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

        ax.set_xlabel(f'{xlab} (Scaled)', fontweight='medium')
        ax.set_ylabel(f'{ylab} (Scaled)', fontweight='medium')
        ax.set_title(f'{ylab} vs {xlab}', loc='left', fontweight='bold', pad=15)

        p_str = "P < 0.001" if pd.notna(p_val) and p_val < 0.001 else f"P = {p_val:.3e}" if pd.notna(p_val) else "P = N/A"
        beta_str = f"$\\beta$ = {beta:.3f}" if pd.notna(beta) else "$\\beta$ = N/A"
        
        ax.text(0.05, 0.95, f"{beta_str}\n{p_str}", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='lightgray'))

        sns.despine(trim=True, offset=5)
        plt.tight_layout()

        save_name = f"{phenotype}_VS_{environment}.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"\n>>> Successfully generated all regression plots in: {output_dir}")

if __name__ == "__main__":
    main()