import os
import torch
import numpy as np
import argparse
import time
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "dust3r")))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="Image size")
    parser.add_argument("--model_path", type=str, default="submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="Path to model weights")
    parser.add_argument("--device", type=str, default='cuda', help="PyTorch device")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--schedule", type=str, default='linear', help="Learning rate schedule")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--niter", type=int, default=300, help="Number of iterations for optimization")
    parser.add_argument("--focal_avg", action="store_true")
    parser.add_argument("--n_views", type=int, default=12, help="Number of views to process")
    parser.add_argument("--img_base_path", type=str, required=True, help="Base path to images")
    parser.add_argument("--use_masks", action='store_true', help="Use masks during processing")
    return parser

def load_and_preprocess_images(img_folder_path, size):
    images, ori_size = load_images(img_folder_path, size=size)
    return images, ori_size

def load_and_resize_masks(mask_folder_path, target_size):
    masks = []
    for mask_name in sorted(os.listdir(mask_folder_path)):
        mask_path = os.path.join(mask_folder_path, mask_name)
        mask = np.array(Image.open(mask_path).convert("L"))  # Ensure grayscale
        mask_resized = cv2.resize(mask, target_size).astype(np.uint8)
        masks.append(mask_resized)
    return masks

def apply_masks(data, masks):
    masked_data = [cv2.bitwise_and(d, d, mask=m) for d, m in zip(data, masks)]
    return masked_data

def apply_mask2mask(mask1, mask2):
    masked_mask = [
        cv2.bitwise_and(
            cm.astype(np.uint8), cm.astype(np.uint8), mask=mask
        ).astype(bool) for cm, mask in zip(mask1, mask2)
    ]
    return masked_mask

def save_masked_images(images_folder, masks_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images_path = sorted(os.listdir(images_folder))
    masks_path = sorted(os.listdir(masks_folder))
    for i, (img_path, mask_path) in enumerate(zip(images_path, masks_path)):
        img = np.array(Image.open(os.path.join(images_folder, img_path)))
        mask = np.array(Image.open(os.path.join(masks_folder, mask_path)))
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        output_path = os.path.join(output_folder, f"masked_image_{i:03d}.png")
        cv2.imwrite(output_path, masked_img)

def clean_with_dbscan(points, colors, eps=0.02, min_samples=100):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    unique_labels = set(labels)
    filtered_points, filtered_colors = [], []

    for label in unique_labels:
        if label == -1:  # Noise
            continue
        cluster_mask = labels == label
        filtered_points.append(points[cluster_mask])
        filtered_colors.append(colors[cluster_mask])

    return np.vstack(filtered_points), np.vstack(filtered_colors)

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Directories and Model Setup
    img_folder_path = os.path.join(args.img_base_path, "images")
    mask_folder_path = os.path.join(args.img_base_path, "masks")
    masked_images_folder = os.path.join(args.img_base_path, "masked_images")
    output_colmap_path = os.path.join(args.img_base_path, "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_path).to(args.device)

    # Load Images
    images, ori_size = load_and_preprocess_images(img_folder_path, size=args.image_size)

    # Inference
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=args.batch_size)

    # Global Alignment
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    compute_global_alignment(scene, init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr, focal_avg=args.focal_avg)
    scene = scene.clean_pointcloud()

    # Extract Results
    imgs = to_numpy(scene.imgs)
    poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())

    # Handle Masks
    if args.use_masks:
        if not os.path.isdir(mask_folder_path):
            raise FileNotFoundError(f"Mask folder not found: {mask_folder_path}")
        target_size = imgs[0].shape[:2][::-1]  # (W, H)
        masks = load_and_resize_masks(mask_folder_path, target_size)
        save_masked_images(img_folder_path, mask_folder_path, masked_images_folder)  # Save masked images
        pts3d = apply_masks(pts3d, masks)
        imgs = apply_masks(imgs, masks)
        confidence_masks = apply_mask2mask(confidence_masks, masks)

    # Save Colmap Outputs
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), sorted(os.listdir(img_folder_path)))

    # Mask Point Clouds
    pts3d = [p[m] for p, m in zip(pts3d, confidence_masks)]
    imgs = [img[m] for img, m in zip(imgs, confidence_masks)]

    # DBSCAN Filtering
    pts_4_3dgs = np.concatenate(pts3d)
    color_4_3dgs = np.concatenate(imgs)
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)

    print("Cleaning with DBSCAN...")
    filtered_points, filtered_colors = clean_with_dbscan(pts_4_3dgs, color_4_3dgs, eps=0.02, min_samples=100)

    # Save Final Point Cloud
    storePly(os.path.join(output_colmap_path, "points3D.ply"), filtered_points, filtered_colors)

    print("Processing completed.")

if __name__ == "__main__":
    main()
