import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

data_csv_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/analysis_ready_data.csv"
taxonomy_csv_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/species_taxonomy_mapping.csv"
output_dir = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/Codes/pics_paper/Phylo_Robustness_All"

def main():
    if not os.path.exists(data_csv_path) or not os.path.exists(taxonomy_csv_path):
        print(">>> [Error] Data or Taxonomy CSV not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(">>> Loading datasets and merging taxonomy...")
    df = pd.read_csv(data_csv_path)
    tax_df = pd.read_csv(taxonomy_csv_path)
    
    df = pd.merge(df, tax_df, on='species', how='inner')
    
    genus_counts = df['genus'].value_counts()
    valid_genera = genus_counts[genus_counts >= 10].index
    df = df[df['genus'].isin(valid_genera)].copy()
    
    if 'dino_pc1' in df.columns:
        df['dino_pc1_mutation'] = np.abs(df['dino_pc1'] - df['dino_pc1'].mean())
        
    test_pairs = [
        ('lightness', 'bio4_temp_seasonality'),
        ('pattern_complexity', 'bio15_precip_seasonality'),
        ('phenotypic_disparity', 'bio6_min_temp'),
        ('functional_beta_diversity', 'temperature'),
        ('dino_pc1_mutation', 'bio4_temp_seasonality')
    ]
    
    log_transform_cols = [
        'human_footprint', 'precipitation', 
        'pattern_complexity', 'phenotypic_disparity', 'functional_beta_diversity'
    ]
    
    spatial_formula = " + bs(longitude, df=4) + bs(latitude, df=4)"
    
    print("\n>>> Executing Phylogenetic Residual Extraction & Spatial GAMM for All Phenotypes...")
    
    sns.set_theme(style="ticks", font_scale=1.0)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

    for y, x in test_pairs:
        if y not in df.columns or x not in df.columns:
            print(f"  -> Skipping {y} vs {x}: Columns not found.")
            continue
            
        print(f"\n--- Testing Pair: {y} vs {x} ---")
        
        y_raw = df[y]
        if y in log_transform_cols:
            y_raw = np.log1p(np.maximum(y_raw, 0))
        df[f"{y}_scaled"] = (y_raw - y_raw.mean()) / y_raw.std()
        
        x_raw = df[x]
        if x in log_transform_cols:
            x_raw = np.log1p(np.maximum(x_raw, 0))
        df[f"{x}_scaled"] = (x_raw - x_raw.mean()) / x_raw.std()
        
        fit_df = df[[f"{y}_scaled", f"{x}_scaled", 'genus', 'spatial_grid', 'longitude', 'latitude']].dropna()
        
        phylo_mod = smf.ols(f"{y}_scaled ~ C(genus)", data=fit_df).fit()
        fit_df['phenotype_phylo_free'] = phylo_mod.resid
        
        genetic_variance_ratio = phylo_mod.rsquared * 100
        print(f"  => Variance Explained by Genus (Genetic Effect): {genetic_variance_ratio:.2f}%")
        
        try:
            group_counts = fit_df['spatial_grid'].value_counts()
            valid_groups = group_counts[group_counts >= 3].index
            model_df = fit_df[fit_df['spatial_grid'].isin(valid_groups)]
            
            f_spatial_orig = f"{y}_scaled ~ {x}_scaled{spatial_formula}"
            orig_mod = smf.mixedlm(f_spatial_orig, model_df, groups=model_df["spatial_grid"]).fit(method='lbfgs', maxiter=200, disp=False)
            beta_orig = orig_mod.params[f"{x}_scaled"]
            pval_orig = orig_mod.pvalues[f"{x}_scaled"]
            print(f"  => Uncorrected Beta: {beta_orig:.4f} (P-value: {pval_orig:.2e})")

            f_spatial_corr = f"phenotype_phylo_free ~ {x}_scaled{spatial_formula}"
            final_mod = smf.mixedlm(f_spatial_corr, model_df, groups=model_df["spatial_grid"]).fit(method='lbfgs', maxiter=200, disp=False)
            beta_corr = final_mod.params[f"{x}_scaled"]
            pval_corr = final_mod.pvalues[f"{x}_scaled"]
            print(f"  => Corrected Beta:   {beta_corr:.4f} (P-value: {pval_corr:.2e})")

            fig, ax = plt.subplots(figsize=(8, 6))
            plot_df = model_df.sample(min(10000, len(model_df)), random_state=42)
            
            ax.scatter(plot_df[f"{x}_scaled"], plot_df["phenotype_phylo_free"], alpha=0.15, s=20, color="#31688E", linewidths=0)
            
            x_left, x_right = ax.get_xlim()
            x_line = np.array([x_left, x_right])
            x_mean = plot_df[f"{x}_scaled"].mean()

            if pd.notna(beta_orig):
                y_mean_orig = plot_df[f"{y}_scaled"].mean()
                intercept_orig = y_mean_orig - beta_orig * x_mean
                y_line_orig = intercept_orig + beta_orig * x_line
                ax.plot(x_line, y_line_orig, color="#808080", linewidth=3.5, linestyle=':', label="Uncorrected Lineage", zorder=9)

            if pd.notna(beta_corr):
                y_mean_corr = plot_df["phenotype_phylo_free"].mean()
                intercept_corr = y_mean_corr - beta_corr * x_mean
                y_line_corr = intercept_corr + beta_corr * x_line
                ax.plot(x_line, y_line_corr, color="#D3515B", linewidth=3.5, linestyle='-', solid_capstyle='round', label="Phylo-Corrected Effect", zorder=10)

            ax.set_xlim(x_left, x_right)

            xlab = x.replace('_', ' ').title()
            ylab = y.replace('_', ' ').title()
            ax.set_xlabel(f"{xlab} (Scaled)", fontweight='medium')
            ax.set_ylabel(f"{ylab} (Phylo-Free Residuals)", fontweight='medium')
            ax.set_title(f"Phylogenetic Decoupling: {ylab} vs {xlab}", loc='left', fontweight='bold', pad=15)

            p_str_orig = "P < 0.001" if pval_orig < 0.001 else f"P = {pval_orig:.3e}"
            p_str_corr = "P < 0.001" if pval_corr < 0.001 else f"P = {pval_corr:.3e}"
            
            stat_text = (f"Genus Genetic Effect: {genetic_variance_ratio:.1f}%\n\n"
                         f"[Uncorrected GAMM]\n$\\beta$ = {beta_orig:.3f} | {p_str_orig}\n\n"
                         f"[Phylo-Corrected GAMM]\n$\\beta$ = {beta_corr:.3f} | {p_str_corr}")
            
            ax.text(0.05, 0.95, stat_text, 
                    transform=ax.transAxes, fontsize=11, verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='lightgray'))

            ax.legend(loc='lower right', frameon=True, fontsize=10)

            sns.despine(trim=True, offset=5)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"Decoupled_{y}_vs_{x}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  => Saved Plot: {save_path}")

        except Exception as e:
            print(f"  => GAMM/Plot Failed: {e}")

    print("\n>>> All Phylogenetic Validations Complete.")

if __name__ == "__main__":
    main()
