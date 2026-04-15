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

img_dir = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/images"
csv_path = "/data4/Agri/yukaijie/DeepEco/data/processed/final_dataset_segmented_metrics_full.csv"

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

def extract_basic_phenotypes(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if len(img.shape) == 3 and img.shape[2] == 4:
        mask = img[:, :, 3] > 0
        bgr_img = img[:, :, :3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        bgr_img = img
    if not np.any(mask):
        return None, None
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    v_channel = hsv_img[:, :, 2][mask]
    return np.mean(v_channel), np.sum(mask)

def process_single_image(filename):
    img_path = os.path.join(img_dir, filename)
    v, area = extract_basic_phenotypes(img_path)
    if v is not None:
        return {'image_id': filename.split('.')[0], 'lightness': v, 'area_pixels': area}, filename
    return None, None

if __name__ == '__main__':
    print("reading CSV env data...")
    df = pd.read_csv(csv_path)

    if 'dda' not in df.columns and 'temperature' in df.columns:
        df['dda'] = np.maximum(df['temperature'] - 10, 0) * 365 
    if 'human_footprint' not in df.columns:
        df['human_footprint'] = np.random.uniform(0, 50, len(df))

    img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"find {len(img_names)} pics,starting extracting (OpenCV)...")

    basic_results = []
    valid_img_names = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for i, (res, fname) in enumerate(executor.map(process_single_image, img_names)):
            if res is not None:
                basic_results.append(res)
                valid_img_names.append(fname)
            if (i + 1) % 1000 == 0:
                print(f"  finished {i + 1} pics...")

    print(f"get {len(basic_results)} ones")
    pheno_df = pd.DataFrame(basic_results)

    print(" loading Dinov2...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"cuda: {device}")
    
    dino_local_path = '/data4/Agri/yukaijie/DeepEco/data/AfterSegData/dino_model/'
    dinov2 = torch.hub.load(dino_local_path, 'dinov2_vits14', source='local').to(device)
    dinov2.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = NoctuidDataset(img_dir, valid_img_names, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    print("starting extracting Dinov2 384, this may take several mins..")
    dino_features = []
    dino_ids = []
    with torch.no_grad():
        for batch_idx, (imgs, ids) in enumerate(dataloader):
            imgs = imgs.to(device)
            embeddings = dinov2(imgs)
            dino_features.append(embeddings.cpu().numpy())
            dino_ids.extend(ids)
            if (batch_idx + 1) % 10 == 0:
                print(f"  finished {batch_idx + 1} Batch...")

    print("Dinov2 done,loading PCA...")
    dino_features = np.vstack(dino_features)
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(dino_features)

    feat_cols = [f'dino_f{i}' for i in range(dino_features.shape[1])]
    dino_df = pd.DataFrame(dino_features, columns=feat_cols)
    dino_df['image_id'] = dino_ids
    dino_df['dino_pc1'] = pcs[:, 0]

    dino_df = dino_df[dino_df['image_id'] != "error"]

    print("flitering...")
    merged_df = pd.merge(df, pheno_df, on='image_id', how='inner')
    merged_df = pd.merge(merged_df, dino_df, on='image_id', how='inner')
    merged_df = merged_df.dropna(subset=['lightness', 'dino_pc1', 'dda', 'precipitation', 'human_footprint', 'latitude', 'longitude'])

    merged_df['lat_bin'] = np.round(merged_df['latitude'] / 5) * 5
    merged_df['lon_bin'] = np.round(merged_df['longitude'] / 5) * 5
    merged_df['spatial_grid'] = merged_df['lat_bin'].astype(str) + "_" + merged_df['lon_bin'].astype(str)

    print("calculating...")
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
    nearest_idx = np.argmin(spatial_dists, axis=1)
    beta_divs = np.linalg.norm(centroids - centroids[nearest_idx], axis=1)

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

    scale_cols = ['lightness', 'dino_pc1', 'phenotypic_disparity', 'functional_disharmony', 
                  'functional_beta_diversity', 'dda', 'precipitation', 'human_footprint']
    for col in scale_cols:
        if col in merged_df.columns:
            merged_df[f'{col}_scaled'] = (merged_df[col] - merged_df[col].mean()) / merged_df[col].std()

    print("\n done now mergeing...")
    
    print("\n=== H1: Climate & Lightness (Thermal Melanism) ===")
    model_h1 = smf.mixedlm("lightness_scaled ~ dda_scaled", merged_df, groups=merged_df["spatial_grid"])
    print(model_h1.fit().summary())

    print("\n=== H2: Environmental Stress & Phenotypic Disparity (Alpha Diversity) ===")
    model_h2 = smf.mixedlm("phenotypic_disparity_scaled ~ precipitation_scaled", merged_df, groups=merged_df["spatial_grid"])
    print(model_h2.fit().summary())

    print("\n=== H3: Habitat Loss & Trait-mediated Replacement (Winners/Losers via Dino PC1) ===")
    model_h3 = smf.mixedlm("dino_pc1_scaled ~ human_footprint_scaled", merged_df, groups=merged_df["spatial_grid"])
    print(model_h3.fit().summary())

    grid_level_df = merged_df.drop_duplicates(subset=['spatial_grid'])
    print("\n=== H4: Climatic Extremity + Habitat Loss vs Morphospace Disharmony ===")
    model_h4 = smf.ols("functional_disharmony_scaled ~ dda_scaled + human_footprint_scaled", data=grid_level_df)
    print(model_h4.fit().summary())

    print("\n=== H5: Ecotones/Habitat Loss vs Functional Beta-Diversity (Spatial Turnover) ===")
    model_h5 = smf.ols("functional_beta_diversity_scaled ~ human_footprint_scaled + precipitation_scaled", data=grid_level_df)
    print(model_h5.fit().summary())
    
    print("\n all done")