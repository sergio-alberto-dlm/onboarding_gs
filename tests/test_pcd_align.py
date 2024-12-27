import os
import numpy as np
import open3d as o3d
from pathlib import Path
import cv2

# Create dummy directories and files for testing
test_root = Path("dummy_test")
os.makedirs(test_root, exist_ok=True)

# Create dummy point cloud files
source_pcd_path = test_root / "source.ply"
target_pcd_path = test_root / "target.ply"

source_pcd = o3d.geometry.PointCloud()
source_pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
source_pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))
o3d.io.write_point_cloud(str(source_pcd_path), source_pcd)

target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(np.random.rand(150, 3))
target_pcd.colors = o3d.utility.Vector3dVector(np.random.rand(150, 3))
o3d.io.write_point_cloud(str(target_pcd_path), target_pcd)

# Create dummy image directories and images
masked_target_path = test_root / "masked_target"
masked_source_path = test_root / "masked_source"
os.makedirs(masked_target_path, exist_ok=True)
os.makedirs(masked_source_path, exist_ok=True)

for i in range(5):
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(masked_target_path / f"target_{i}.png"), img)
    cv2.imwrite(str(masked_source_path / f"source_{i}.png"), img)

# Create dummy camera and image parameter files
cameras_txt = """1 PINHOLE 1920 1440 1654.8506927490234 1654.8506927490234 960.0 720.0
2 PINHOLE 1920 1440 1654.8506927490234 1654.8506927490234 960.0 720.0
3 PINHOLE 1920 1440 1654.8506927490234 1654.8506927490234 960.0 720.0
4 PINHOLE 1920 1440 1654.8506927490234 1654.8506927490234 960.0 720.0
5 PINHOLE 1920 1440 1654.8506927490234 1654.8506927490234 960.0 720.0
"""
images_txt = """1 0.8714897042748023 0.12381329605631622 0.3220551210742562 0.3485060462241677 -0.13785144686698914 0.03173702582716942 0.03165799379348755 1 frame_00.jpg

2 0.9455167061163462 0.28849622925046153 -0.08931139596061945 -0.12162134936787995 0.04443303123116493 0.12043661624193192 0.009951060637831688 2 frame_01.jpg

3 0.952829002320039 0.19265485840496147 0.1460825622114043 0.18346887241686413 -0.0846940279006958 0.07686378061771393 0.013240816071629524 3 frame_02.jpg

4 0.999978065250156 -0.0034216065765935136 -0.00416008008577 0.0038461346040998376 0.0 0.0 0.0 4 frame_03.jpg

5 0.9839501750919332 0.17758884084903648 -0.006005552029042651 -0.01637448513185249 -0.015018966048955917 0.028789782896637917 0.13071659207344055 5 frame_04.jpg

"""

target_camera_path = test_root / "cameras_target.txt"
source_camera_path = test_root / "cameras_source.txt"
target_image_path = test_root / "images_target.txt"
source_image_path = test_root / "images_source.txt"

with open(target_camera_path, "w") as f:
    f.write(cameras_txt)
with open(source_camera_path, "w") as f:
    f.write(cameras_txt)
with open(target_image_path, "w") as f:
    f.write(images_txt)
with open(source_image_path, "w") as f:
    f.write(images_txt)

print(f"Dummy test files created in {test_root}")
