import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

csv_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/analysis_ready_data.csv"
output_dir = "/data4/Agri/yukaijie/DeepEco/data/Codes"

def main():
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    if 'dino_pc1' in df.columns:
        df['dino_pc1_mutation'] = np.abs(df['dino_pc1'] - df['dino_pc1'].mean())

    agg_dict = {
        'lon_bin': 'first',
        'lat_bin': 'first',
        'elevation': 'mean',
        'lightness': 'mean',
        'pattern_complexity': 'mean',
        'phenotypic_disparity': 'mean',
        'functional_beta_diversity': 'mean'
    }
    if 'dino_pc1_mutation' in df.columns:
        agg_dict['dino_pc1_mutation'] = 'mean'

    grid_df = df.groupby('spatial_grid').agg(agg_dict).reset_index()
    grid_df = grid_df.rename(columns={'lon_bin': 'lon', 'lat_bin': 'lat'})

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
    
    fig_map, axes_map = plt.subplots(3, 2, figsize=(20, 18))
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
        ('elevation', 'terrain', '(a) Elevation with Contours', axes_map[0]),
        ('lightness', 'viridis', '(b) Wing Lightness with Contours', axes_map[1]),
        ('pattern_complexity', 'plasma', '(c) Pattern Complexity with Contours', axes_map[2]),
        ('phenotypic_disparity', 'magma', '(d) Phenotypic Disparity with Contours', axes_map[3]),
        ('dino_pc1_mutation', 'inferno', '(e) Dino PC1 Mutation with Contours', axes_map[4]),
        ('functional_beta_diversity', 'cividis', '(f) Functional Beta-Diversity with Contours', axes_map[5])
    ]

    lon_bins = np.arange(-180, 180.5, 0.5)
    lat_bins = np.arange(-90, 90.5, 0.5)
    X, Y = np.meshgrid(lon_bins[:-1] + 0.25, lat_bins[:-1] + 0.25)
    X_mesh, Y_mesh = np.meshgrid(lon_bins, lat_bins)

    Z_elev = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    for _, row in grid_df.iterrows():
        lon_idx = int(np.floor((row['lon'] - (-180)) / 0.5))
        lat_idx = int(np.floor((row['lat'] - (-90)) / 0.5))
        if 0 <= lat_idx < Z_elev.shape[0] and 0 <= lon_idx < Z_elev.shape[1]:
            Z_elev[lat_idx, lon_idx] = row['elevation']

    Z_elev_filled = np.nan_to_num(Z_elev, nan=0.0)

    for metric, cmap, title, ax in map_configs:
        if metric not in grid_df.columns:
            ax.set_visible(False)
            continue

        if world is not None:
            world.plot(ax=ax, color='#EFEFEF', edgecolor='white', linewidth=0.5, zorder=1)

        Z_metric = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
        for _, row in grid_df.iterrows():
            lon_idx = int(np.floor((row['lon'] - (-180)) / 0.5))
            lat_idx = int(np.floor((row['lat'] - (-90)) / 0.5))
            if 0 <= lat_idx < Z_metric.shape[0] and 0 <= lon_idx < Z_metric.shape[1]:
                Z_metric[lat_idx, lon_idx] = row[metric]

        pc = ax.pcolormesh(X_mesh, Y_mesh, Z_metric, cmap=cmap, alpha=0.85, zorder=2, shading='flat')

        levels = [1000, 2500, 4000]
        contour = ax.contour(X, Y, Z_elev_filled, levels=levels, colors='#222222', linewidths=[0.6, 0.9, 1.2], alpha=0.7, zorder=3)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%d m', colors='#222222')

        ax.set_title(title, fontweight='bold', pad=15, fontsize=16)
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
    out_path = os.path.join(output_dir, 'JANE_Figure_4_All_Phenotypes_Contours_6Panels.png')
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"\n>>> Saved to: {out_path}")

if __name__ == "__main__":
    main()