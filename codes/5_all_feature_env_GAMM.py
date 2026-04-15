import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from tqdm import tqdm

env_csv_path = "/data4/Agri/yukaijie/DeepEco/data/processed/final_dataset_segmented_metrics_full.csv"
cv_cache_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/cv_features_cache.csv"
output_dir = "/data4/Agri/yukaijie/DeepEco/data/Codes"

def main():
    if not os.path.exists(env_csv_path):
        return
    if not os.path.exists(cv_cache_path):
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\n>>> Loading Datasets...")
    env_df = pd.read_csv(env_csv_path, dtype={'gbif_id': str, 'file_path': str})
    
    rename_mapping = {
        'gbif_id': 'image_id',
        'bio1_mean_temp': 'temperature',
        'bio12_precip': 'precipitation',
        'alan_nightlight': 'human_footprint'
    }
    env_df = env_df.rename(columns=rename_mapping)

    if 'file_path' in env_df.columns:
        env_df['image_id'] = env_df['file_path'].astype(str).apply(lambda x: os.path.basename(str(x)).split('.')[0])
    else:
        env_df['image_id'] = env_df['image_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    if 'dda' not in env_df.columns and 'temperature' in env_df.columns:
        env_df['dda'] = np.maximum(env_df['temperature'] - 10, 0) * 365 

    cv_df = pd.read_csv(cv_cache_path)
    cv_df['image_id'] = cv_df['image_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    print("\n>>> Merging Environmental and Visual Data...")
    merged_df = pd.merge(env_df, cv_df, on='image_id', how='inner')
    merged_df = merged_df.dropna(subset=['lightness', 'pattern_complexity', 'latitude', 'longitude'])

    if len(merged_df) == 0:
        return

    print("\n>>> Gridding data at 0.5 x 0.5 degree resolution...")
    merged_df['lat_bin'] = np.round(merged_df['latitude'] / 0.5) * 0.5
    merged_df['lon_bin'] = np.round(merged_df['longitude'] / 0.5) * 0.5
    merged_df['spatial_grid'] = merged_df['lat_bin'].astype(str) + "_" + merged_df['lon_bin'].astype(str)

    feat_cols = ['lightness', 'pattern_complexity']
    if 'dino_pc1' in merged_df.columns:
        feat_cols.append('dino_pc1')

    grid_info = []
    map_data = []
    
    grouped_grids = merged_df.groupby('spatial_grid')
    
    for grid, group in tqdm(grouped_grids, desc="Spatial Metrics", unit="grid", ncols=100):
        grid_feats = group[feat_cols].values
        centroid = np.mean(grid_feats, axis=0).reshape(1, -1)
        
        if len(grid_feats) > 1:
            distances = euclidean_distances(grid_feats, centroid).flatten()
        else:
            distances = np.array([0.0])
            
        mean_disparity = np.mean(distances)
        mean_lightness = group['lightness'].mean()
        mean_complexity = group['pattern_complexity'].mean()
        
        lat_val = group['lat_bin'].iloc[0]
        lon_val = group['lon_bin'].iloc[0]
        
        grid_info.append({
            'spatial_grid': grid,
            'lat': lat_val,
            'lon': lon_val,
            'centroid': centroid[0],
            'image_id': group['image_id'].values,
            'disparity': distances
        })
        
        map_data.append({
            'lat': lat_val,
            'lon': lon_val,
            'mean_disparity': mean_disparity,
            'mean_lightness': mean_lightness,
            'mean_complexity': mean_complexity
        })

    coords = np.array([[d['lat'], d['lon']] for d in grid_info])
    centroids = np.vstack([d['centroid'] for d in grid_info])
    spatial_dists = cdist(coords, coords)
    np.fill_diagonal(spatial_dists, np.inf)
    
    if len(spatial_dists) > 1:
        nearest_idx = np.argmin(spatial_dists, axis=1)
        beta_divs = np.linalg.norm(centroids - centroids[nearest_idx], axis=1)
    else:
        beta_divs = np.zeros(len(grid_info))

    expanded_grid_data = []
    for i, d in enumerate(grid_info):
        beta_div = beta_divs[i]
        map_data[i]['beta_diversity'] = beta_div
        for idx, img_id in enumerate(d['image_id']):
            expanded_grid_data.append({
                'image_id': img_id,
                'phenotypic_disparity': d['disparity'][idx],
                'functional_beta_diversity': beta_div
            })

    advanced_metrics_df = pd.DataFrame(expanded_grid_data)
    merged_df = pd.merge(merged_df, advanced_metrics_df, on='image_id')
    map_df = pd.DataFrame(map_data)

    analysis_ready_path = os.path.join(os.path.dirname(env_csv_path), "analysis_ready_data.csv")
    merged_df.to_csv(analysis_ready_path, index=False)
    print(f"\n>>> Cached fully merged analysis-ready data to: {analysis_ready_path}")

    phenotypes_ind = ['lightness', 'pattern_complexity', 'dino_pc1', 'phenotypic_disparity']
    phenotypes_grid = ['functional_beta_diversity']
    
    exclude_cols = ['image_id', 'latitude', 'longitude', 'file_path', 'species', 'image_url', 
                    'lat_bin', 'lon_bin', 'spatial_grid', 'gbif_id']
    
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    env_factors = [c for c in numeric_cols if c not in phenotypes_ind + phenotypes_grid + exclude_cols and not c.endswith('_scaled')]

    scale_cols = phenotypes_ind + phenotypes_grid + env_factors
    for col in scale_cols:
        if col in merged_df.columns:
            std_val = merged_df[col].std()
            if std_val != 0 and not np.isnan(std_val):
                merged_df[f'{col}_scaled'] = (merged_df[col] - merged_df[col].mean()) / std_val
            else:
                merged_df[f'{col}_scaled'] = 0.0

    spatial_formula = " + bs(longitude, df=4) + bs(latitude, df=4)"
    
    print("\n>>> Running Comprehensive Statistical Modeling (All Phenotypes vs All Dynamic Environments)...")
    
    stats_results = []
    
    for p in phenotypes_ind:
        if p not in merged_df.columns:
            continue
        for e in tqdm(env_factors, desc=f"Modeling {p}", ncols=100, leave=False):
            try:
                f = f"{p}_scaled ~ {e}_scaled{spatial_formula}"
                mod = smf.mixedlm(f, merged_df, groups=merged_df["spatial_grid"]).fit(disp=False)
                coef = mod.params[f'{e}_scaled']
                pval = mod.pvalues[f'{e}_scaled']
                stats_results.append({'Phenotype': p, 'Environment': e, 'Model': 'Spatial GAMM', 'Beta': coef, 'P_Value': pval})
            except:
                pass

    grid_level_df = merged_df.drop_duplicates(subset=['spatial_grid'])
    
    for p in phenotypes_grid:
        if p not in grid_level_df.columns:
            continue
        for e in tqdm(env_factors, desc=f"Modeling {p} (Grid Level)", ncols=100, leave=False):
            try:
                f = f"{p}_scaled ~ {e}_scaled{spatial_formula}"
                mod = smf.ols(f, data=grid_level_df).fit()
                coef = mod.params[f'{e}_scaled']
                pval = mod.pvalues[f'{e}_scaled']
                stats_results.append({'Phenotype': p, 'Environment': e, 'Model': 'Spatial OLS', 'Beta': coef, 'P_Value': pval})
            except:
                pass

    res_df = pd.DataFrame(stats_results)
    res_csv_path = os.path.join(output_dir, "comprehensive_statistics_summary.csv")
    res_df.to_csv(res_csv_path, index=False)
    print(f"\n>>> Saved Comprehensive Statistical Summary to: {res_csv_path}")

    def get_stat(p, e):
        r = res_df[(res_df['Phenotype'] == p) & (res_df['Environment'] == e)]
        if len(r) > 0:
            return r.iloc[0]['Beta'], r.iloc[0]['P_Value']
        return None, None

    print(">>> Generating Main Relationships (Figure 1)...")
    sns.set_theme(style="ticks", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sample_df = merged_df.sample(min(10000, len(merged_df)), random_state=42)
    
    if 'dda_scaled' in merged_df.columns:
        sns.regplot(data=sample_df, x='dda_scaled', y='lightness_scaled', ax=axes[0,0],
                    scatter_kws={'alpha': 0.15, 's': 20, 'color': '#31688E', 'linewidths': 0}, line_kws={'color': '#D3515B', 'linewidth': 3})
        axes[0,0].set_xlabel('Degree-Day Accumulation (Scaled)')
        axes[0,0].set_ylabel('Wing Lightness (Scaled)')
        axes[0,0].set_title('(a) Thermal Melanism Hypothesis', loc='left', fontweight='bold', pad=15)
        coef, pval = get_stat('lightness', 'dda')
        if coef is not None:
            axes[0,0].text(0.05, 0.95, f"$\\beta$ = {coef:.3f}\nP < 0.001" if pval < 0.001 else f"$\\beta$ = {coef:.3f}\nP = {pval:.3f}", transform=axes[0,0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    if 'bio4_temp_seasonality_scaled' in merged_df.columns:
        sns.regplot(data=sample_df, x='bio4_temp_seasonality_scaled', y='phenotypic_disparity_scaled', ax=axes[0,1],
                    scatter_kws={'alpha': 0.15, 's': 20, 'color': '#35B779', 'linewidths': 0}, line_kws={'color': '#440154', 'linewidth': 3})
        axes[0,1].set_xlabel('Temperature Seasonality (Scaled)')
        axes[0,1].set_ylabel('Pattern Disparity (Scaled)')
        axes[0,1].set_title('(b) Environmental Fluctuation & Disparity', loc='left', fontweight='bold', pad=15)
        coef, pval = get_stat('phenotypic_disparity', 'bio4_temp_seasonality')
        if coef is not None:
            axes[0,1].text(0.05, 0.95, f"$\\beta$ = {coef:.3f}\nP < 0.001" if pval < 0.001 else f"$\\beta$ = {coef:.3f}\nP = {pval:.3f}", transform=axes[0,1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    if 'human_footprint_scaled' in merged_df.columns:
        sns.regplot(data=sample_df, x='human_footprint_scaled', y='pattern_complexity_scaled', ax=axes[1,0],
                    scatter_kws={'alpha': 0.15, 's': 20, 'color': '#FDE725', 'edgecolors': 'black', 'linewidths': 0.2}, line_kws={'color': '#D3515B', 'linewidth': 3})
        axes[1,0].set_xlabel('ALAN / Human Footprint (Scaled)')
        axes[1,0].set_ylabel('Wing Pattern Complexity (Scaled)')
        axes[1,0].set_title('(c) Habitat Loss & Camouflage Filter', loc='left', fontweight='bold', pad=15)
        coef, pval = get_stat('pattern_complexity', 'human_footprint')
        if coef is not None:
            axes[1,0].text(0.05, 0.95, f"$\\beta$ = {coef:.3f}\nP < 0.001" if pval < 0.001 else f"$\\beta$ = {coef:.3f}\nP = {pval:.3f}", transform=axes[1,0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    if 'elevation_scaled' in merged_df.columns:
        sns.regplot(data=grid_level_df, x='elevation_scaled', y='functional_beta_diversity_scaled', ax=axes[1,1],
                    scatter_kws={'alpha': 0.6, 's': 50, 'color': '#440154', 'linewidths': 0.5, 'edgecolor': 'white'}, line_kws={'color': '#35B779', 'linewidth': 3})
        axes[1,1].set_xlabel('Elevation (Scaled)')
        axes[1,1].set_ylabel('Functional Beta-Diversity (Scaled)')
        axes[1,1].set_title('(d) Topographic Barriers & Spatial Turnover', loc='left', fontweight='bold', pad=15)
        coef, pval = get_stat('functional_beta_diversity', 'elevation')
        if coef is not None:
            axes[1,1].text(0.05, 0.95, f"$\\beta$ = {coef:.3f}\nP < 0.001" if pval < 0.001 else f"$\\beta$ = {coef:.3f}\nP = {pval:.3f}", transform=axes[1,1].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    
    fig1_path = os.path.join(output_dir, 'JANE_Figure_1_Fast_Data.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')

    print("\n>>> Generating Global Distribution Maps (Figure 2)...")
    fig_map, axes_map = plt.subplots(2, 2, figsize=(20, 12))
    axes_map = axes_map.flatten()
    
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
    except Exception:
        try:
            url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json"
            world = gpd.read_file(url)
            world = world[world.name != "Antarctica"]
        except Exception:
            world = None

    map_configs = [
        ('mean_lightness', 'viridis', '(a) Global Spatial Distribution of Wing Lightness'),
        ('mean_complexity', 'plasma', '(b) Global Spatial Distribution of Pattern Complexity'),
        ('mean_disparity', 'magma', '(c) Global Spatial Heterogeneity of Phenotypic Disparity'),
        ('beta_diversity', 'cividis', '(d) Global Spatial Turnover (Functional Beta-Diversity)')
    ]
    
    lon_bins = np.arange(-180, 180.5, 0.5)
    lat_bins = np.arange(-90, 90.5, 0.5)
    X, Y = np.meshgrid(lon_bins, lat_bins)

    for ax, (metric, cmap, title) in zip(axes_map, map_configs):
        if world is not None:
            world.plot(ax=ax, color='#EFEFEF', edgecolor='white', linewidth=0.5, zorder=1)
            
        Z = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
        for _, row in map_df.iterrows():
            lon_idx = int(np.floor((row['lon'] - (-180)) / 0.5))
            lat_idx = int(np.floor((row['lat'] - (-90)) / 0.5))
            if 0 <= lat_idx < Z.shape[0] and 0 <= lon_idx < Z.shape[1]:
                Z[lat_idx, lon_idx] = row[metric]
                
        pc = ax.pcolormesh(X, Y, Z, cmap=cmap, alpha=0.9, zorder=2, shading='flat')
        
        ax.set_title(title, fontweight='bold', pad=15, fontsize=14)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 90)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        cbar = plt.colorbar(pc, ax=ax, fraction=0.02, pad=0.04)
        cbar.outline.set_visible(False)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'JANE_Figure_2_Global_Maps_Fast.png')
    plt.savefig(fig2_path, dpi=300, facecolor='white', bbox_inches='tight')
    
    print(f"\n>>> All operations completed! Results saved in {output_dir}")

if __name__ == "__main__":
    main()