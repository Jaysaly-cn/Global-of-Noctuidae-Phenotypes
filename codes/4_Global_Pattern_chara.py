import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from tqdm import tqdm
import argparse

torch.backends.cudnn.benchmark = True

img_dir = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/images/"
csv_path = "/data4/Agri/yukaijie/DeepEco/data/processed/final_dataset_segmented_metrics_full.csv"
cv_cache_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/cv_features_cache.csv"
dino_cache_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/cv_dino_cache.csv"
analysis_ready_path = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/datas/analysis_ready_data.csv"

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    print("\n>>> Loading environmental data...")
    env_df = pd.read_csv(csv_path, dtype={'gbif_id': str, 'file_path': str})
    rename_mapping = {'gbif_id': 'image_id', 'bio1_mean_temp': 'temperature', 'bio12_precip': 'precipitation', 'alan_nightlight': 'human_footprint'}
    env_df = env_df.rename(columns=rename_mapping)
    if 'file_path' in env_df.columns:
        env_df['image_id'] = env_df['file_path'].astype(str).apply(lambda x: os.path.basename(str(x)).split('.')[0])
    else:
        env_df['image_id'] = env_df['image_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    if 'dda' not in env_df.columns and 'temperature' in env_df.columns:
        env_df['dda'] = np.maximum(env_df['temperature'] - 10, 0) * 365 

    if not os.path.exists(cv_cache_path):
        print(f"\n[ERROR] Missing OpenCV cache at {cv_cache_path}")
        return

    print(f"\n>>> Directly loading OpenCV features from: {cv_cache_path}")
    pheno_df = pd.read_csv(cv_cache_path)
    pheno_df['image_id'] = pheno_df['image_id'].astype(str)

    if os.path.exists(dino_cache_path):
        print(f"\n>>> Loading Dinov2 high-dimensional features from cache: {dino_cache_path}")
        dino_df = pd.read_csv(dino_cache_path)
        dino_df['image_id'] = dino_df['image_id'].astype(str)
    else:
        print("\n>>> Local Dino cache not found. Starting GPU Dinov2 extraction...")
        
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cpu')
            
        print(f">>> Target Device: {device} | Batch Size: {args.batch_size} | Workers: {args.workers}")
        
        dino_local_path = '/data4/Agri/yukaijie/DeepEco/data/AfterSegData/dino_model/'
        dinov2 = torch.hub.load(dino_local_path, 'dinov2_vits14', source='local').to(device)
        dinov2.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        all_files_in_dir = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        id_to_filename = {f.split('.')[0]: f for f in all_files_in_dir}
        valid_dino_names = [id_to_filename[str(img_id)] for img_id in pheno_df['image_id'].tolist() if str(img_id) in id_to_filename]
        
        dataset = NoctuidDataset(img_dir, valid_dino_names, transform=transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True,
            prefetch_factor=4 if args.workers > 0 else None
        )

        dino_features = []
        dino_ids = []
        
        with torch.no_grad():
            for imgs, ids in tqdm(dataloader, desc="Dinov2 AMP Inference", ncols=100):
                imgs = imgs.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    embeddings = dinov2(imgs)
                dino_features.append(embeddings.cpu().numpy())
                dino_ids.extend([str(x) for x in ids])
        
        print(">>> Performing PCA on Dinov2 features...")
        dino_features = np.vstack(dino_features)
        n_comp = min(3, dino_features.shape[0])
        pca = PCA(n_components=n_comp)
        pcs = pca.fit_transform(dino_features)
        
        feat_cols = [f'dino_f{i}' for i in range(dino_features.shape[1])]
        dino_df = pd.DataFrame(dino_features, columns=feat_cols)
        dino_df['image_id'] = dino_ids
        dino_df['dino_pc1'] = pcs[:, 0] if pcs.shape[1] > 0 else 0
        if pcs.shape[1] > 1: dino_df['dino_pc2'] = pcs[:, 1]
        if pcs.shape[1] > 2: dino_df['dino_pc3'] = pcs[:, 2]
        dino_df = dino_df[dino_df['image_id'] != "error"]
        dino_df.to_csv(dino_cache_path, index=False)
        print(f">>> Dinov2 features successfully cached to {dino_cache_path}")

    print("\n>>> Merging Environmental, OpenCV, and Dinov2 data...")
    merged_df = pd.merge(env_df, pheno_df, on='image_id', how='inner')
    merged_df = pd.merge(merged_df, dino_df, on='image_id', how='inner')
    merged_df = merged_df.dropna(subset=['lightness', 'pattern_complexity', 'dino_pc1', 'latitude', 'longitude'])

    print(">>> Calculating spatial metrics (0.5 x 0.5 degrees)...")
    merged_df['lat_bin'] = np.round(merged_df['latitude'] / 0.5) * 0.5
    merged_df['lon_bin'] = np.round(merged_df['longitude'] / 0.5) * 0.5
    merged_df['spatial_grid'] = merged_df['lat_bin'].astype(str) + "_" + merged_df['lon_bin'].astype(str)

    feat_cols = ['lightness', 'pattern_complexity', 'dino_pc1']
    if 'dino_pc2' in merged_df.columns: 
        feat_cols.append('dino_pc2')

    grid_info = []
    for grid, group in tqdm(merged_df.groupby('spatial_grid'), desc="Spatial Heterogeneity", ncols=100):
        grid_feats = group[feat_cols].values
        centroid = np.mean(grid_feats, axis=0).reshape(1, -1)
        distances = euclidean_distances(grid_feats, centroid).flatten() if len(grid_feats) > 1 else np.array([0.0])
        grid_info.append({
            'spatial_grid': grid,
            'lat': group['lat_bin'].iloc[0],
            'lon': group['lon_bin'].iloc[0],
            'centroid': centroid[0],
            'image_id': group['image_id'].values,
            'disparity': distances
        })

    coords = np.array([[d['lat'], d['lon']] for d in grid_info])
    centroids = np.vstack([d['centroid'] for d in grid_info])
    spatial_dists = cdist(coords, coords)
    np.fill_diagonal(spatial_dists, np.inf)
    beta_divs = np.linalg.norm(centroids - centroids[np.argmin(spatial_dists, axis=1)], axis=1) if len(spatial_dists) > 1 else np.zeros(len(grid_info))

    expanded_grid_data = []
    for i, d in enumerate(grid_info):
        for idx, img_id in enumerate(d['image_id']):
            expanded_grid_data.append({
                'image_id': img_id,
                'phenotypic_disparity': d['disparity'][idx],
                'functional_beta_diversity': beta_divs[i]
            })

    advanced_metrics_df = pd.DataFrame(expanded_grid_data)
    merged_df = pd.merge(merged_df, advanced_metrics_df, on='image_id')

    merged_df.to_csv(analysis_ready_path, index=False)
    print("\n" + "="*80)
    print(f"ULTIMATE DATASET SUCCESSFULLY GENERATED AND SAVED TO:\n{analysis_ready_path}")
    print("="*80)
    print(f"Total valid samples: {len(merged_df)}")
    print("Confirmed columns present:")
    for check_col in ['lightness', 'pattern_complexity', 'dino_pc1', 'phenotypic_disparity', 'functional_beta_diversity']:
        status = "?" if check_col in merged_df.columns else "? MISSING"
        print(f" - {check_col}: {status}")

if __name__ == "__main__":
    main()