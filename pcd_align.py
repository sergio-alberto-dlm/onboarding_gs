import os
import shutil
import numpy as np
import open3d as o3d
from pathlib import Path
from utils.colmap_loader import qvec2rotmat, rotmat2qvec
from plyfile import PlyData, PlyElement

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source point cloud file")
    parser.add_argument("--target_path", type=str, required=True, help="Path to the target point cloud file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the aligned point cloud")
    parser.add_argument("--threshold", type=float, default=0.02, help="ICP convergence threshold")
    parser.add_argument("--masked_target_path", type=str, required=True, help="Path to masked images for face up")
    parser.add_argument("--masked_source_path", type=str, required=True, help="Path to masked images for face down")
    parser.add_argument("--target_camera_path", type=str, required=True, help="Path to cameras.txt for face up")
    parser.add_argument("--source_camera_path", type=str, required=True, help="Path to cameras.txt for face down")
    parser.add_argument("--target_image_path", type=str, required=True, help="Path to images.txt for face up")
    parser.add_argument("--source_image_path", type=str, required=True, help="Path to images.txt for face down")
    return parser

def align_point_clouds(source_path, target_path, threshold):
    # Load source and target point clouds
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # Define rotation matrix (180 degrees around X-axis)
    R = np.array([[1,  0,  0],
                  [0, -1,  0],
                  [0,  0, -1]])

    # Rotate source point cloud
    centroid = source.get_center()
    source.translate(-centroid)
    source.rotate(R, center=(0, 0, 0))
    source.translate(centroid)

    # Perform ICP registration
    initial_transform = np.eye(4)
    result_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Transform source point cloud
    source.transform(result_icp.transformation)

    return source, target, result_icp.transformation

def merge_masked_images(source_path, target_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Move and rename images from target
    target_images = sorted(os.listdir(target_path))
    for i, image in enumerate(target_images):
        src = os.path.join(target_path, image)
        dest = os.path.join(output_path, f"frame_{i:02d}.png")
        shutil.move(src, dest)

    # Move and rename images from source, continuing numbering
    source_images = sorted(os.listdir(source_path))
    offset = len(target_images)
    for i, image in enumerate(source_images):
        src = os.path.join(source_path, image)
        dest = os.path.join(output_path, f"frame_{i + offset:02d}.png")
        shutil.move(src, dest)

    # Remove original directories
    shutil.rmtree(target_path)
    shutil.rmtree(source_path)

def merge_intrinsics_and_extrinsics(target_camera_path, source_camera_path, 
                                    target_image_path, source_image_path, 
                                    output_camera_path, output_image_path, 
                                    source_transform):
    # Merge cameras.txt (intrinsics are identical, copy target's file and replicate lines)
    with open(target_camera_path, 'r') as file:
        target_camera_data = file.readlines()

    num_cameras_target = len(target_camera_data)
    num_cameras_source = len(open(source_camera_path).readlines())

    with open(output_camera_path, 'w') as file:
        for _ in range(num_cameras_target + num_cameras_source):
            file.writelines(target_camera_data[0])  # Write identical intrinsics line for each camera

    # Merge images.txt
    target_lines = open(target_image_path).readlines()
    source_lines = open(source_image_path).readlines()

    def save_colmap_images(poses, images_file, train_img_list):
        with open(images_file, 'w') as f:
            for i, pose in enumerate(poses, 1):  # Starting index at 1
                pose = np.linalg.inv(pose)
                R = pose[:3, :3]
                t = pose[:3, 3]
                q = rotmat2qvec(R)  # Convert rotation matrix to quaternion
                f.write(f"{i} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i} {train_img_list[i-1]}")
                f.write("\n")

    with open(output_image_path, 'w') as outfile:
        frame_offset = 0
        for line in target_lines:
            outfile.write(line)
            if not line.strip(): frame_offset+=1 

        for i, line in enumerate(source_lines):
            parts = line.strip().split()
            if len(parts) > 0:
                index = int(parts[0]) + frame_offset
                qvec = np.array(parts[1:5], dtype=float)
                tvec = np.array(parts[5:8], dtype=float)
                R = qvec2rotmat(qvec)  # Convert quaternion to rotation matrix
                pose = np.vstack([np.hstack([R, tvec.reshape(3, -1)]), np.array([0, 0, 0, 1])])
                pose = np.linalg.inv(pose)
                # Apply transformation
                updated_pose = source_transform @ pose
                updated_pose = np.linalg.inv(updated_pose)
                updated_R, updated_t = updated_pose[:3, :3], updated_pose[:3, 3]
                updated_q = rotmat2qvec(updated_R)

                frame_name = f"frame_{index:02d}.jpg"
                outfile.write(f"{index} {updated_q[0]} {updated_q[1]} {updated_q[2]} {updated_q[3]} {updated_t[0]} {updated_t[1]} {updated_t[2]} {index} {frame_name}\n")
                outfile.write("\n")

def concatenate_point_clouds(source, target):
    # Load vertices and colors from source and target
    source_xyz = np.asarray(source.points)
    target_xyz = np.asarray(target.points)

    source_rgb = np.asarray(source.colors) * 255  # Convert to 0-255 scale
    target_rgb = np.asarray(target.colors) * 255

    source_normals = np.asarray(source.normals) if source.has_normals() else np.zeros_like(source_xyz)
    target_normals = np.asarray(target.normals) if target.has_normals() else np.zeros_like(target_xyz)

    # Concatenate points, colors, and normals
    combined_xyz = np.vstack((source_xyz, target_xyz))
    combined_rgb = np.vstack((source_rgb, target_rgb)).astype(np.uint8)
    combined_normals = np.vstack((source_normals, target_normals))

    return combined_xyz, combined_rgb, combined_normals

def save_combined_point_cloud(combined_xyz, combined_rgb, combined_normals, output_path):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(combined_xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((combined_xyz, combined_normals, combined_rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(output_path)

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Align point clouds
    source, target, transformation = align_point_clouds(
        args.source_path,
        args.target_path,
        args.threshold
    )

    # Concatenate point clouds
    combined_xyz, combined_rgb, combined_normals = concatenate_point_clouds(source, target)

    # Save combined point cloud
    colmap_path = os.path.join(args.output_path, "sparse", "0")
    os.makedirs(colmap_path, exist_ok=True)
    save_combined_point_cloud(combined_xyz, combined_rgb, combined_normals, Path(colmap_path) / "points3D.ply")

    # Merge masked images
    merge_masked_images(Path(args.masked_source_path), Path(args.masked_target_path), Path(args.output_path) / "images")

    # Merge intrinsic and extrinsic parameters, first combined the flip + ICP transformation alignment 
    R = np.array([[1,  0,  0, 0],
                  [0, -1,  0, 0],
                  [0,  0, -1, 0], 
                  [0, 0, 0, 1]])
    #print(transformation)
    transformation = transformation @ R
    merge_intrinsics_and_extrinsics(
        Path(args.target_camera_path),
        Path(args.source_camera_path),
        Path(args.target_image_path),
        Path(args.source_image_path),
        Path(colmap_path) / "cameras.txt",
        Path(colmap_path) / "images.txt",
        transformation
    )

    print(f"All merged data saved to {args.output_path}")

if __name__ == "__main__":
    main()