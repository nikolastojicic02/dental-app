import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from numba import jit, prange

# --- 1. PRE-PROCESSING UTILS ---

@jit(nopython=True, parallel=True, fastmath=True)
def _farthest_point_sampling_core(points, num_samples):
    """Numba-optimized FPS core."""
    N = points.shape[0]
    centroids = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float32)
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid_point = points[farthest]
        
        # Parallel distance computation
        for j in prange(N):
            dist = 0.0
            for k in range(points.shape[1]):
                diff = points[j, k] - centroid_point[k]
                dist += diff * diff
            if dist < distances[j]:
                distances[j] = dist
        
        farthest = np.argmax(distances)
    
    return centroids

def farthest_point_sampling(points, num_samples):
    """Uniformly samples points from the cloud."""
    N, D = points.shape
    if N <= num_samples:
        indices = np.random.choice(N, num_samples, replace=True)
        return indices
    return _farthest_point_sampling_core(points, num_samples)

def load_stl_for_inference(stl_path, num_points=4096):
    """Loads STL and returns points with normals."""
    mesh = trimesh.load(stl_path)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]
    
    # Zero-center (vectorized)
    centroid = points.mean(axis=0)
    points -= centroid
    
    features = np.hstack((points, normals)).astype(np.float32)
    return features, centroid, mesh

# --- 2. MATH UTILS ---

@jit(nopython=True, fastmath=True)
def rotation_6d_to_matrix(r6d):
    """Converts 6D vector to 3x3 rotation matrix."""
    x = r6d[:3] / (np.linalg.norm(r6d[:3]) + 1e-8)
    y_raw = r6d[3:]
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z) + 1e-8)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))

@jit(nopython=True, fastmath=True)
def transform_to_local(vertices, center, orientation):
    """Moves mesh vertices to the predicted bounding box's local space."""
    centered = vertices - center
    return (orientation.T @ centered.T).T

# --- 3. POST-PROCESSING (Cleaning the AI's predictions) ---

def apply_knn_smoothing(points, labels, k=20):
    """Optimized KNN smoothing with BallTree."""
    from sklearn.neighbors import BallTree
    tree = BallTree(points, leaf_size=40)
    _, indices = tree.query(points, k=k)
    
    # Vectorized mode computation
    neighbor_labels = labels[indices]
    mode_results, _ = stats.mode(neighbor_labels, axis=1, keepdims=True)
    return mode_results.flatten()

def refine_segmentation(points, labels, eps=1.2, min_cluster_size=20):
    """Removes small floating 'noise' islands from the dental prep."""
    refined_labels = labels.copy()
    prep_classes = [2, 3, 4, 5, 6]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    for cls in prep_classes:
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue

        cls_pcd = pcd.select_by_index(cls_indices)
        cluster_labels = np.array(cls_pcd.cluster_dbscan(eps=eps, min_points=5))
        
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        
        # Vectorized filtering
        noise_mask = (cluster_labels == -1)
        for cluster_id, count in zip(unique_clusters, counts):
            if cluster_id != -1 and count < min_cluster_size:
                noise_mask |= (cluster_labels == cluster_id)
        
        if np.any(noise_mask):
            original_indices = cls_indices[noise_mask]
            refined_labels[original_indices] = 1
                
    return refined_labels