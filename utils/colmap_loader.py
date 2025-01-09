import os 
import numpy as np 
from plyfile import PlyData
from collections import defaultdict

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def read_pose_matrices(file_path):
    poses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                index = int(parts[0])
                R = qvec2rotmat(np.array(parts[1:5], dtype=float))
                T = np.array(parts[5:8], dtype=float)
                pose = np.vstack([np.hstack([R, T.reshape(3, -1)]), np.array([0, 0, 0, 1])])
                poses.append({'index': index, 'pose': pose})
    return poses

class image:
    def __init__(self, rotation, translation, id, name):
        self.R = rotation
        self.tvec = translation
        self.camera_id = id
        self.name = name 

class camera:
    def __init__(self, camera_type,  width, hight, fx, fy, cx, cy, k1=0, k2=0, k3=0, k4=0, p1=0, p2=0):
        self.camera_type = camera_type
        self.width = width
        self.hight = hight
        self.fx = fx 
        self.fy = fy 
        self.cx = cx 
        self.cy = cy 
        self.k1 = k1 
        self.k2 = k2
        self.k3 = k3 
        self.k4 = k4 
        self.p1 = p1 
        self.p2 = p2 

class SceneManager:
    def __init__(self, colmap_dir):
        self.colmap_dir = colmap_dir
        self.image_path = os.path.join(colmap_dir, "images.txt")
        self.cameras_path = os.path.join(colmap_dir, "cameras.txt")
        self.points_path = os.path.join(colmap_dir, "points3D.ply")

        assert os.path.isfile(self.image_path), f"images file not found in {self.image_path}"
        assert os.path.isfile(self.cameras_path), f"cameras file not found in {self.cameras_path}"
        assert os.path.isfile(self.points_path), f"3D points file not found in {self.points_path}"

        self.images = {}
        self.cameras = {}
        self.points3D = None
        self.points3D_colors = None
        self.normals = None

        self.name_to_image_id = {}
        self.point3D_id_to_images = defaultdict(list)
        self.point3D_id_to_point3D_idx = {}

    def load_images(self):
        with open(self.image_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    index = int(parts[0])
                    R = qvec2rotmat(np.array(parts[1:5], dtype=float))
                    T = np.array(parts[5:8], dtype=float)
                    name = str(parts[9])
                    self.images[index] = image(R, T, index, name)
                    self.name_to_image_id[name] = index

    def load_cameras(self):
        with open(self.cameras_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    index = int(parts[0])
                    camera_type = str(parts[1])
                    width, height = int(parts[2]), int(parts[3])
                    fx, fy = float(parts[4]), float(parts[5])
                    cx, cy = int(parts[6]), int(parts[7])
                    if camera_type in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"}:
                        k1 = float(parts[8])
                        k2 = float(parts[9]) if len(parts) > 9 else 0
                        k3 = float(parts[10]) if len(parts) > 10 else 0
                        k4 = float(parts[11]) if len(parts) > 11 else 0
                        p1 = float(parts[12]) if len(parts) > 12 else 0
                        p2 = float(parts[13]) if len(parts) > 13 else 0
                    else:
                        k1, k2, k3, k4, p1, p2 = 0, 0, 0, 0, 0, 0
                    self.cameras[index] = camera(camera_type, width, height, fx, fy, cx, cy, k1, k2, k3, k4, p1, p2)

    def load_points3D(self):
        ply_data = PlyData.read(self.points_path)
        vertex = ply_data['vertex']
        self.points3D = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
        self.points3D_colors = np.vstack((vertex['red'], vertex['green'], vertex['blue'])).T
        self.normals = np.vstack((vertex['nx'], vertex['ny'], vertex['nz'])).T
        self.points_err = np.zeros((len(self.points3D,)))

        # Extract additional mappings
        for i, (x, y, z, nx, ny, nz, red, green, blue) in enumerate(
            zip(vertex['x'], vertex['y'], vertex['z'], vertex['nx'], vertex['ny'], vertex['nz'], vertex['red'], vertex['green'], vertex['blue'])
        ):
            point_id = i  # Example placeholder, replace with actual point ID from data if available
            self.point3D_id_to_point3D_idx[point_id] = i

        # Simulate image associations (if available in the dataset)
        # For actual data, replace the following with real associations
        self.point3D_id_to_images = defaultdict(list)  # Example empty association

    def get_camera_matrix(self, camera_id):
        camera = self.cameras[camera_id]
        K = np.array([
            [camera.fx, 0, camera.cx],
            [0, camera.fy, camera.cy],
            [0, 0, 1]
        ])
        return K

    def get_image_extrinsics(self, image_id):
        image = self.images[image_id]
        R = image.R
        tvec = image.tvec
        extrinsic = np.hstack((R, tvec.reshape(3, 1)))
        return extrinsic

    def build_point_indices(self):
        point_indices = {}
        image_id_to_name = {v: k for k, v in self.name_to_image_id.items()}
        for point_id, data in self.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = self.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {k: np.array(v).astype(np.int32) for k, v in point_indices.items()}
        return point_indices