import os 
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from munch import munchify
import json

import torch 
from torch import Tensor
from gsplat.rendering import rasterization

def compute_centroid_and_radius(points):
    """
    Compute the centroid (mean of all points) and a radius
    proportional to the object size. 
    
    Options for the radius:
    1) Half of the maximum bounding-box extent (simple bounding sphere).
    2) An actual minimal bounding sphere (more involved).
    Here we do the bounding box approach for simplicity.
    """
    centroid = np.mean(points, axis=0)
    
    # bounding box extents
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    bbox_size = max_xyz - min_xyz
    radius = 0.5 * np.linalg.norm(bbox_size)  # half-diagonal
    
    return centroid, radius

def icosahedron_vertices():
    """
    Return the base icosahedron vertices (12) and faces (20).
    The returned vertices are on the unit sphere.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    # Twelve vertices of an icosahedron
    verts = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ], dtype=np.float64)
    
    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    
    # Faces of the icosahedron (20 triangular faces)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    
    return verts, faces

def subdivide_icosahedron_once(verts, faces):
    """
    Subdivide each triangle in the icosahedron once.
    This yields 42 unique vertices on the unit sphere.
    """
    edge_map = {}
    new_faces = []
    next_vertex_index = len(verts)
    # We’ll store new vertices in a list to append them as we generate them
    new_verts = verts.tolist()
    
    def get_midpoint_index(i1, i2):
        """
        For the edge (i1, i2), compute the midpoint on the unit sphere.
        Use a dictionary to avoid duplicating edges.
        """
        # Ensure smaller index first for consistent edge key
        if i2 < i1:
            i1, i2 = i2, i1
        edge_key = (i1, i2)
        if edge_key in edge_map:
            return edge_map[edge_key]
        
        # Compute midpoint and normalize
        v1 = np.array(new_verts[i1])
        v2 = np.array(new_verts[i2])
        midpoint = 0.5 * (v1 + v2)
        midpoint /= np.linalg.norm(midpoint)
        
        # Add to new_verts list
        new_verts.append(midpoint.tolist())
        edge_map[edge_key] = len(new_verts) - 1
        return edge_map[edge_key]
    
    # Subdivide each face
    for tri in faces:
        iA, iB, iC = tri[0], tri[1], tri[2]
        iAB = get_midpoint_index(iA, iB)
        iBC = get_midpoint_index(iB, iC)
        iCA = get_midpoint_index(iC, iA)
        
        # Create new faces
        new_faces.append([iA, iAB, iCA])
        new_faces.append([iB, iBC, iAB])
        new_faces.append([iC, iCA, iBC])
        new_faces.append([iAB, iBC, iCA])
    
    new_verts = np.array(new_verts, dtype=np.float64)
    new_faces = np.array(new_faces, dtype=np.int32)
    
    return new_verts, new_faces

def generate_icosphere_level_1():
    """
    Generates an icosphere at subdivision level 1.
    This should have 42 vertices on the unit sphere.
    """
    base_verts, base_faces = icosahedron_vertices()
    verts, faces = subdivide_icosahedron_once(base_verts, base_faces)
    # Optionally, you can remove duplicates if needed, but with the edge_map approach,
    # we should already have unique vertices.
    return verts, faces

def look_at(camera_pos, target, up=np.array([0, 1, 0], dtype=float)):
    """
    Compute a standard right-handed look-at camera extrinsic matrix:
    
    - camera_pos: 3D position of the camera.
    - target: 3D position to look at.
    - up: approximate "world up" vector.
    
    Returns:
        A 4x4 extrinsic matrix (world->camera).
        The camera looks down the -Z axis in its local coordinate system.
    """
    forward = camera_pos - target
    forward /= np.linalg.norm(forward)  # The -Z axis in camera coords
    
    # Recompute orthonormal basis
    right = np.cross(up, forward)
    right /= (np.linalg.norm(right) + 3e-10)
    
    up_new = np.cross(forward, right)
    up_new /= (np.linalg.norm(up_new) + 3e-10)
    
    # Rotation part
    R = np.eye(4, dtype=float)
    # Camera’s X axis = 'right'
    R[0, 0:3] = right
    # Camera’s Y axis = 'up'
    R[1, 0:3] = up_new
    # Camera’s Z axis = 'forward' (which is -Z from camera’s perspective)
    R[2, 0:3] = forward
    
    # Translation part
    T = np.eye(4, dtype=float)
    T[0:3, 3] = -camera_pos
    
    # The extrinsic matrix is R * T
    extrinsic = R @ T
    
    return extrinsic

def load_ckpt(ckpt_path : str, device="cuda"):
    assert os.path.isfile(ckpt_path), f"checkpoint not found in {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    return ckpt

def read_splats(ckpt):
    """ function to read the splats from a pre-trained checkpoint
        input  : pytorch checkpooint
        output : dictionary of splats """
    splats = {key : ckpt['splats'][key] for key in ckpt['splats'].keys()}
    return splats 

# def read_config(config_path):
#     with open(config_path, 'r') as file:
#         conf