import os
import cv2
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

img_dir = "/data4/Agri/yukaijie/DeepEco/data/AfterSegData/images"
csv_path = "/data4/Agri/yukaijie/DeepEco/data/processed/final_dataset_segmented_metrics_full.csv"

def extract_phenotypes(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None, None, None, None, None, None
    
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        mask = alpha_channel > 0
        bgr_img = img[:, :, :3]
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 0
        bgr_img = img

    if not np.any(mask):
        return None, None, None, None, None, None, None

    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    
    h_channel = hsv_img[:, :, 0][mask]
    s_channel = hsv_img[:, :, 1][mask]
    v_channel = hsv_img[:, :, 2][mask]

    mean_h = np.mean(h_channel)
    mean_s = np.mean(s_channel)
    mean_v = np.mean(v_channel)
    area = np.sum(mask)

    x, y, w, h_box = cv2.boundingRect(mask.astype(np.uint8))
    roi_gray = gray[y:y+h_box, x:x+w]
    roi_mask = mask[y:y+h_box, x:x+w]

    edges = cv2.Canny(roi_gray, 50, 150)
    edges_masked = cv2.bitwise_and(edges, edges, mask=roi_mask.astype(np.uint8))
    edge_density = np.sum(edges_masked > 0) / area if area > 0 else 0

    laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
    complexity_var = np.var(laplacian[roi_mask]) if np.any(roi_mask) else 0

    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu1 = hu_moments[0] if len(hu_moments) > 0 else 0

    return mean_h, mean_s, mean_v, area, edge_density, complexity_var, hu1

df = pd.read_csv(csv_path)

results = []
for filename in os.listdir(img_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(img_dir, filename)
        h, s, v, area, edge_density, complexity_var, hu1 = extract_phenotypes(img_path)
        
        if v is not None:
            image_id = filename.split('.')[0]
            results.append({
                'image_id': image_id,
                'hue': h,
                'saturation': s,
                'lightness': v,
                'area_pixels': area,
                'edge_density': edge_density,
                'texture_complexity': complexity_var,
                'pattern_dist_hu1': hu1
            })

pheno_df = pd.DataFrame(results)

merged_df = pd.merge(df, pheno_df, on='image_id', how='inner')

merged_df = merged_df.dropna(subset=['lightness', 'texture_complexity', 'temperature', 'precipitation', 'species'])

merged_df['lightness_scaled'] = (merged_df['lightness'] - merged_df['lightness'].mean()) / merged_df['lightness'].std()
merged_df['texture_scaled'] = (merged_df['texture_complexity'] - merged_df['texture_complexity'].mean()) / merged_df['texture_complexity'].std()
merged_df['edge_scaled'] = (merged_df['edge_density'] - merged_df['edge_density'].mean()) / merged_df['edge_density'].std()
merged_df['dist_hu1_scaled'] = (merged_df['pattern_dist_hu1'] - merged_df['pattern_dist_hu1'].mean()) / merged_df['pattern_dist_hu1'].std()
merged_df['temp_scaled'] = (merged_df['temperature'] - merged_df['temperature'].mean()) / merged_df['temperature'].std()
merged_df['prec_scaled'] = (merged_df['precipitation'] - merged_df['precipitation'].mean()) / merged_df['precipitation'].std()

model_bogert = smf.mixedlm("lightness_scaled ~ temp_scaled", merged_df, groups=merged_df["species"])
result_bogert = model_bogert.fit()
print(result_bogert.summary())

model_texture_temp = smf.mixedlm("texture_scaled ~ temp_scaled", merged_df, groups=merged_df["species"])
result_texture_temp = model_texture_temp.fit()
print(result_texture_temp.summary())

model_edge_prec = smf.mixedlm("edge_scaled ~ prec_scaled", merged_df, groups=merged_df["species"])
result_edge_prec = model_edge_prec.fit()
print(result_edge_prec.summary())