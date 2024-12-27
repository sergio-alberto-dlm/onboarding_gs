import sys
sys.path.append("..")  # Add the parent directory to the path
import open3d as o3d
import numpy as np
import os
from PIL import Image
from utils.colmap_loader import read_pose_matrices


def render_point_cloud(pcd_path, poses, intrinsic, output_dir, img_size):
    """
    Renders 2D projection images of a point cloud given camera poses.

    Args:
        pcd_path (str): Path to the point cloud file (.ply format).
        poses (list): List of 4x4 numpy arrays representing camera poses.
        intrinsic (o3d.camera.PinholeCameraIntrinsic): Camera intrinsic parameters.
        output_dir (str): Directory to save the rendered images.
        img_size (tuple): Image size as (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1], visible=False)
    vis.add_geometry(pcd)

    # Configure rendering settings
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])  # white background
    render_option.point_size = 1.0  # Adjust as needed for better visualization

    # Loop through poses to render images
    for i, pose in enumerate(poses):
        print(f"Intrinsic width: {intrinsic.width}, height: {intrinsic.height}")
        print(f"Window size: {img_size[0]}x{img_size[1]}")
        print(f"Pose {i}: {pose}")
        assert pose.shape == (4, 4), f"Pose {i} is not a 4x4 matrix!"
        assert np.allclose(pose[3], [0, 0, 0, 1]), f"Pose {i} bottom row is incorrect!"
        R = pose[:3, :3]
        #assert np.allclose(np.dot(R, R.T), np.eye(3)), f"Pose {i} rotation matrix is not orthonormal!"
        #print("orthonormal: ", np.dot(R, R.T))
        #assert np.isclose(np.linalg.det(R), 1), f"Pose {i} rotation matrix determinant is not 1!"
        #print("det: ", np.linalg.det(R))

        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=img_size[0],
            height=img_size[1],
            fx=intrinsic.get_focal_length()[0],
            fy=intrinsic.get_focal_length()[1],
            cx=intrinsic.get_principal_point()[0],
            cy=intrinsic.get_principal_point()[1]
        )
        camera_params.extrinsic = pose

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        
        vis.poll_events()
        vis.update_renderer()

        # Save rendered image
        image_path = os.path.join(output_dir, f"frame_{i:02d}.png")
        vis.capture_screen_image(image_path)
        print(f"Rendered image saved to {image_path}")

    vis.destroy_window()


def main():
    # paths
    dataset = "hope"
    object_id = 1
    num_views = 7
    pose_file = f"/home/sergio/onboarding_stage/data/{dataset}/obj_{object_id:06d}/align/sparse/0/images.txt"
    pcd_path = f"/home/sergio/onboarding_stage/data/{dataset}/obj_{object_id:06d}/align/sparse/0/points3D.ply"
    output_dir = "dummy_test/camera_pose_images"
    img_size = (1080, 1080)  # HD resolution

    # Camera intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=img_size[0], height=img_size[1], fx=100.0, fy=100.0, cx=img_size[0] // 2, cy=img_size[1] // 2)

    # Load camera poses
    poses = read_pose_matrices(pose_file)
    poses_mat = [pose['pose'] for pose in poses]

    # Render point cloud from poses
    render_point_cloud(pcd_path, poses_mat, intrinsic, output_dir, img_size)


if __name__ == "__main__":
    main()
