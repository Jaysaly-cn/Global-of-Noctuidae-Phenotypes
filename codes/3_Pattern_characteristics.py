import os
import cv2
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
import concurrent.futures
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import random

IS_TEST_MODE = False

cv2.setNumThreads(0)

img_dir = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/images/"
csv_path = "/data4/Agri/yukaijie/DeepEco/data/processed/final_dataset_segmented_metrics_full.csv"

cv_cache_path = os.path.join(img_dir, "cv_features_cache_test.csv") if IS_TEST_MODE else os.path.join(img_dir, "cv_features_cache.csv")

class NoctuidDataset(Dataset):
    def __init__(self, img_dir, img_names, transform=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image_t = self.transform(image)
            return image_t, img_name.split('.')[0]
        except:
            return torch.zeros((3, 224, 224)), "error"

def extract_texture_phenotypes(image_path):
    cv2.setNumThreads(0)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, None
    
    if len(img.shape) == 3 and img.shape[2] == 4:
        mask = img[:, :, 3] > 0
        bgr_img = img[:, :, :3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        bgr_img = img
        
    if not np.any(mask):
        return None, None, None
        
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    v_channel = hsv_img[:, :, 2][mask]
    lightness = np.mean(v_channel)
    
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    pattern_complexity = np.var(laplacian[mask])
    
    area_pixels = np.sum(mask)
    
    return lightness, pattern_complexity, area_pixels

def process_single_image(filename):
    img_path = os.path.join(img_dir, filename)
    l_val, pc_val, area_val = extract_texture_phenotypes(img_path)
    if l_val is not None and pc_val is not None:
        return {'image_id': filename.split('.')[0], 'lightness': l_val, 'pattern_complexity': pc_val, 'area_pixels': area_val}, filename
    return None, None

if __name__ == '__main__':
    print("Loading environmental data...")
    df = pd.read_csv(csv_path, dtype={'gbif_id': str, 'file_path': str})

    rename_mapping = {
        'gbif_id': 'image_id',
        'bio1_mean_temp': 'temperature',
        'bio12_precip': 'precipitation',
        'alan_nightlight': 'human_footprint'
    }
    df = df.rename(columns=rename_mapping)

    if 'file_path' in df.columns:
        df['image_id'] = df['file_path'].astype(str).apply(lambda x: os.path.basename(str(x)).split('.')[0])
    else:
        df['image_id'] = df['image_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    if 'dda' not in df.columns and 'temperature' in df.columns:
        df['dda'] = np.maximum(df['temperature'] - 10, 0) * 365 

    all_img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    csv_ids = set(df['image_id'])
    img_ids = set([f.split('.')[0] for f in all_img_names])
    matched_ids = csv_ids.intersection(img_ids)
    
    if len(matched_ids) == 0:
        print("Match failed! Exiting...")
        exit()

    matched_img_names = [f for f in all_img_names if f.split('.')[0] in matched_ids]
    
    if os.path.exists(cv_cache_path):
        pheno_df = pd.read_csv(cv_cache_path)
        pheno_df['image_id'] = pheno_df['image_id'].astype(str)
        cached_ids = set(pheno_df['image_id'])
        
        if IS_TEST_MODE:
            valid_img_names = [f for f in matched_img_names if f.split('.')[0] in cached_ids]
            sample_size = min(500, len(valid_img_names))
            if sample_size > 0:
                valid_img_names = random.sample(valid_img_names, sample_size)
            print(f"TEST MODE ACTIVE: Found cached OpenCV features. Loading {len(valid_img_names)} images directly...")
        else:
            valid_img_names = [f for f in matched_img_names if f.split('.')[0] in cached_ids]
            print(f"FULL MODE ACTIVE: Found cached OpenCV features. Loading {len(valid_img_names)} images directly...")
            
        if len(valid_img_names) == 0:
            print("Cache mismatch. Please delete the cache file and try again.")
            exit()
    else:
        if IS_TEST_MODE:
            sample_size = min(500, len(matched_img_names))
            print(f"TEST MODE ACTIVE: Sampling {sample_size} matched images for new cache...")
            target_img_names = random.sample(matched_img_names, sample_size)
        else:
            print(f"FULL MODE ACTIVE: Processing all {len(matched_img_names)} matched images...")
            target_img_names = matched_img_names
            
        basic_results = []
        valid_img_names = []
        
        num_cores = max(1, multiprocessing.cpu_count() - 2)
        print(f"Using {num_cores} CPU cores with ProcessPoolExecutor...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            for i, (res, fname) in enumerate(executor.map(process_single_image, target_img_names, chunksize=500)):
                if res is not None:
                    basic_results.append(res)
                    valid_img_names.append(fname)
                if (i + 1) % (100 if IS_TEST_MODE else 5000) == 0:
                    print(f"  Processed {i + 1} images...")

        pheno_df = pd.DataFrame(basic_results)
        pheno_df['image_id'] = pheno_df['image_id'].astype(str)
        pheno_df.to_csv(cv_cache_path, index=False)
        print(f"Extraction complete. Results cached to {cv_cache_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dino_local_path = '/data4/Agri/yukaijie/DeepEco/data/AfterSegData/dino_model/'
    dinov2 = torch.hub.load(dino_local_path, 'dinov2_vits14', source='local').to(device)
    dinov2.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = NoctuidDataset(img_dir, valid_img_names, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=512, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        prefetch_factor=4
    )

    print("Extracting Dinov2 high-dimensional pattern features...")
    dino_features = []
    dino_ids = []
    with torch.no_grad():
        for batch_idx, (imgs, ids) in enumerate(dataloader):
            imgs = imgs.to(device)
            embeddings = dinov2(imgs)
            dino_features.append(embeddings.cpu().numpy())
            dino_ids.extend([str(x) for x in ids])
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    print("Performing PCA on Dinov2 features...")
    dino_features = np.vstack(dino_features)
    n_comp = min(3, dino_features.shape[0])
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(dino_features)

    feat_cols = [f'dino_f{i}' for i in range(dino_features.shape[1])]
    dino_df = pd.DataFrame(dino_features, columns=feat_cols)
    dino_df['image_id'] = dino_ids
    dino_df['dino_pc1'] = pcs[:, 0] if pcs.shape[1] > 0 else 0
    dino_df['dino_pc2'] = pcs[:, 1] if pcs.shape[1] > 1 else 0

    dino_df = dino_df[dino_df['image_id'] != "error"]

    df['image_id'] = df['image_id'].astype(str)
    pheno_df['image_id'] = pheno_df['image_id'].astype(str)
    dino_df['image_id'] = dino_df['image_id'].astype(str)

    merged_df = pd.merge(df, pheno_df, on='image_id', how='inner')
    merged_df = pd.merge(merged_df, dino_df, on='image_id', how='inner')
    
    required_cols = [
        'lightness', 'pattern_complexity', 'area_pixels', 'dino_pc1', 'dino_pc2', 'dda', 'precipitation', 
        'human_footprint', 'latitude', 'longitude', 'ndvi_mean', 
        'bio4_temp_seasonality', 'bio15_precip_seasonality', 'bio5_max_temp', 'bio6_min_temp', 'elevation'
    ]
    
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in CSV: {missing_cols}")
        for col in missing_cols:
            merged_df[col] = 0.0

    merged_df = merged_df.dropna(subset=required_cols)

    if len(merged_df) == 0:
        print("Merged dataframe is empty after dropna. Exiting...")
        exit()

    merged_df['lat_bin'] = np.round(merged_df['latitude'] / 5) * 5
    merged_df['lon_bin'] = np.round(merged_df['longitude'] / 5) * 5
    merged_df['spatial_grid'] = merged_df['lat_bin'].astype(str) + "_" + merged_df['lon_bin'].astype(str)

    baseline_data = merged_df[(merged_df['latitude'] >= -23.5) & (merged_df['latitude'] <= 23.5)]
    if len(baseline_data) > 0:
        global_baseline = np.mean(baseline_data[feat_cols].values, axis=0).reshape(1, -1)
    else:
        global_baseline = np.mean(merged_df[feat_cols].values, axis=0).reshape(1, -1)

    grid_info = []
    for grid, group in merged_df.groupby('spatial_grid'):
        grid_feats = group[feat_cols].values
        centroid = np.mean(grid_feats, axis=0).reshape(1, -1)
        
        if len(grid_feats) > 1:
            distances = euclidean_distances(grid_feats, centroid).flatten()
        else:
            distances = np.array([0.0])
            
        disharmony = euclidean_distances(centroid, global_baseline).flatten()[0]
        
        grid_info.append({
            'spatial_grid': grid,
            'lat': group['lat_bin'].iloc[0],
            'lon': group['lon_bin'].iloc[0],
            'centroid': centroid[0],
            'disharmony': disharmony,
            'image_id': group['image_id'].values,
            'disparity': distances
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

    for i, d in enumerate(grid_info):
        d['beta_diversity'] = beta_divs[i]

    expanded_grid_data = []
    for d in grid_info:
        for idx, img_id in enumerate(d['image_id']):
            expanded_grid_data.append({
                'image_id': img_id,
                'phenotypic_disparity': d['disparity'][idx],
                'functional_disharmony': d['disharmony'],
                'functional_beta_diversity': d['beta_diversity']
            })

    advanced_metrics_df = pd.DataFrame(expanded_grid_data)
    merged_df = pd.merge(merged_df, advanced_metrics_df, on='image_id')

    scale_cols = [
        'lightness', 'pattern_complexity', 'area_pixels', 'dino_pc1', 'dino_pc2', 'phenotypic_disparity', 
        'functional_disharmony', 'functional_beta_diversity', 'dda', 
        'precipitation', 'human_footprint', 'ndvi_mean', 
        'bio4_temp_seasonality', 'bio15_precip_seasonality', 'bio5_max_temp', 'bio6_min_temp', 'elevation'
    ]
    for col in scale_cols:
        if col in merged_df.columns:
            merged_df[f'{col}_scaled'] = (merged_df[col] - merged_df[col].mean()) / merged_df[col].std()

    spatial_formula = " + bs(longitude, df=4) + bs(latitude, df=4)"
    
    print("\nGenerating comprehensive publication-quality figures (12 panels)...")
    sns.set_theme(style="ticks", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    axes = axes.flatten()
    
    sample_df = merged_df.sample(min(10000, len(merged_df)), random_state=42)
    grid_level_df = merged_df.drop_duplicates(subset=['spatial_grid'])

    plot_configs = [
        ('dda_scaled', 'lightness_scaled', '(a) Heat Accumulation vs Lightness', 'DDA (Scaled)', 'Lightness (Scaled)', False, '#31688E', '#D3515B'),
        ('bio5_max_temp_scaled', 'lightness_scaled', '(b) Extreme Max Temp vs Lightness', 'Max Temp (Scaled)', 'Lightness (Scaled)', False, '#31688E', '#D3515B'),
        ('bio6_min_temp_scaled', 'lightness_scaled', '(c) Extreme Min Temp vs Lightness', 'Min Temp (Scaled)', 'Lightness (Scaled)', False, '#31688E', '#D3515B'),
        ('bio4_temp_seasonality_scaled', 'phenotypic_disparity_scaled', '(d) Temp Seasonality vs Disparity', 'Temp Seasonality (Scaled)', 'Phenotypic Disparity', False, '#35B779', '#440154'),
        ('bio15_precip_seasonality_scaled', 'phenotypic_disparity_scaled', '(e) Precip Seasonality vs Disparity', 'Precip Seasonality (Scaled)', 'Phenotypic Disparity', False, '#35B779', '#440154'),
        ('precipitation_scaled', 'phenotypic_disparity_scaled', '(f) Annual Precip vs Disparity', 'Precipitation (Scaled)', 'Phenotypic Disparity', False, '#35B779', '#440154'),
        ('ndvi_mean_scaled', 'pattern_complexity_scaled', '(g) NDVI vs Camouflage Complexity', 'NDVI (Scaled)', 'Pattern Complexity', False, '#FDE725', '#D3515B'),
        ('human_footprint_scaled', 'pattern_complexity_scaled', '(h) ALAN vs Camouflage Complexity', 'Human Footprint (Scaled)', 'Pattern Complexity', False, '#FDE725', '#D3515B'),
        ('human_footprint_scaled', 'area_pixels_scaled', '(i) ALAN vs Body Size Filter', 'Human Footprint (Scaled)', 'Body Size (Area)', False, '#FDE725', '#D3515B'),
        ('elevation_scaled', 'functional_beta_diversity_scaled', '(j) Topography vs Spatial Turnover', 'Elevation (Scaled)', 'Functional Beta-Diversity', True, '#440154', '#35B779'),
        ('precipitation_scaled', 'functional_beta_diversity_scaled', '(k) Precipitation vs Spatial Turnover', 'Precipitation (Scaled)', 'Functional Beta-Diversity', True, '#440154', '#35B779'),
        ('human_footprint_scaled', 'dino_pc1_scaled', '(l) ALAN vs Main Pattern Axis', 'Human Footprint (Scaled)', 'Dino PC1 (Scaled)', False, '#E76F51', '#264653')
    ]

    for i, (x_col, y_col, title, xlab, ylab, is_grid, color_pt, color_line) in enumerate(plot_configs):
        ax = axes[i]
        plot_df = grid_level_df if is_grid else sample_df
        full_df = grid_level_df if is_grid else merged_df

        scatter_kws = {'alpha': 0.15, 's': 20, 'color': color_pt, 'linewidths': 0}
        if is_grid:
            scatter_kws = {'alpha': 0.6, 's': 50, 'color': color_pt, 'linewidths': 0.5, 'edgecolor': 'white'}
        elif color_pt == '#FDE725':
            scatter_kws['edgecolors'] = 'black'
            scatter_kws['linewidths'] = 0.2

        sns.regplot(data=plot_df, x=x_col, y=y_col, ax=ax, scatter_kws=scatter_kws, line_kws={'color': color_line, 'linewidth': 3})
        
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc='left', fontweight='bold', pad=15)

        try:
            if is_grid:
                mod = smf.ols(f"{y_col} ~ {x_col}{spatial_formula}", data=full_df).fit()
            else:
                mod = smf.mixedlm(f"{y_col} ~ {x_col}{spatial_formula}", full_df, groups=full_df["spatial_grid"]).fit()
            
            if x_col in mod.params:
                coef = mod.params[x_col]
                pval = mod.pvalues[x_col]
                p_str = "P < 0.001" if pval < 0.001 else f"P = {pval:.3f}"
                ax.text(0.05, 0.95, f"$\\beta$ = {coef:.3f}\n{p_str}", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='lightgray'))
        except:
            pass

    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    
    code_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = 'JANE_Figure_1_All_Factors_Test' if IS_TEST_MODE else 'JANE_Figure_1_All_Factors_Full'
    pdf_path = os.path.join(code_dir, f'{prefix}.pdf')
    png_path = os.path.join(code_dir, f'{prefix}.png')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"Figures saved successfully at:\n - {pdf_path}\n - {png_path}")
    print("\nScript completed successfully!")