import os
import shutil
from PIL import Image
import argparse
import cv2
from skimage.feature import hog
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count
from functools import partial
from os import makedirs

# Function to compute color histogram
def compute_color_histogram(image, hist_size=[4, 4, 4], hist_range=[0, 256, 0, 256, 0, 256]):
    image_cs = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([image_cs], [0, 1, 2], None, hist_size, hist_range)
    return cv2.normalize(hist, hist).flatten()

# Function to compute HOG features
def compute_hog_features(image, pixels_per_cell=(32, 32), cells_per_block=(1, 1), orientations=6):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, block_norm='L2-Hys', visualize=False)
    return features

# Feature extraction for a single frame
def extract_features(args):
    img_path, mask_path, scene_path, resize_dim, hist_params, hog_params = args
    frame = np.array(Image.open(os.path.join(scene_path, "rgb", img_path)))
    mask = np.array(Image.open(os.path.join(scene_path, "mask_visib", mask_path)))

    frame_resized = cv2.resize(frame, resize_dim)
    mask_resized = cv2.resize(mask, resize_dim)
    frame_masked = cv2.bitwise_and(frame_resized, frame_resized, mask=mask_resized)

    hist = compute_color_histogram(frame_masked, **hist_params)
    hog_features = compute_hog_features(frame_masked, **hog_params)
    return np.hstack((hist, hog_features))

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--object_id", type=int, required=True)
    parser.add_argument("--n_views", type=int, default=5)
    parser.add_argument("--face", type=str, required=True, help="face down or up")
    return parser

# Main function to select keyframes
def select_keyframes(args):
    scene_path = os.path.join(args.data_base_path, f"obj_{args.object_id:06d}_{args.face}")
    imgs_path = sorted(os.listdir(os.path.join(scene_path, "rgb")))
    masks_path = sorted(os.listdir(os.path.join(scene_path, "mask_visib")))
    resize_dim = (200, 200)

    # Feature extraction parameters
    hist_params = {'hist_size': [4, 4, 4], 'hist_range': [0, 256, 0, 256, 0, 256]}
    hog_params = {'pixels_per_cell': (32, 32), 'cells_per_block': (1, 1), 'orientations': 6}

    # Prepare arguments for parallel feature extraction
    extraction_args = [(img_path, mask_path, scene_path, resize_dim, hist_params, hog_params) 
                       for img_path, mask_path in zip(imgs_path, masks_path)]

    # Parallel feature extraction
    with Pool(cpu_count()) as pool:
        combined_features = list(tqdm(pool.imap(extract_features, extraction_args), total=len(imgs_path)))
    #results = [extract_features(arg) for arg in extraction_args]

    combined_features = np.array(combined_features)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)

    # Dimensionality reduction
    pca = PCA(n_components=0.95, svd_solver='full')
    features_reduced = pca.fit_transform(features_scaled)

    # Clustering for keyframes
    kmeans = KMeans(n_clusters=args.n_views, random_state=42)
    kmeans.fit(features_reduced)
    cluster_labels = kmeans.labels_

    key_frame_indices = []
    for cluster_num in range(args.n_views):
        cluster_indices = np.where(cluster_labels == cluster_num)[0]
        distances = np.linalg.norm(features_reduced[cluster_indices] - kmeans.cluster_centers_[cluster_num], axis=1)
        key_frame_indices.append(cluster_indices[np.argmin(distances)])

    # Save selected frames and masks
    dest_img_path = os.path.join("./data", args.dataset_name, f"obj_{args.object_id:06d}", args.face, f"{args.n_views}_views", "images")
    dest_mask_path = os.path.join("./data", args.dataset_name, f"obj_{args.object_id:06d}", args.face, f"{args.n_views}_views", "masks")
    makedirs(dest_img_path, exist_ok=True)
    makedirs(dest_mask_path, exist_ok=True)

    for i, idx in enumerate(key_frame_indices):
        shutil.copy(os.path.join(scene_path, "rgb", imgs_path[idx]), os.path.join(dest_img_path, f"frame_{i:02d}.jpg"))
        shutil.copy(os.path.join(scene_path, "mask_visib", masks_path[idx]), os.path.join(dest_mask_path, f"mask_{i:02d}.png"))

    print(f"Keyframe selection for {args.face} completed.")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    select_keyframes(args)
