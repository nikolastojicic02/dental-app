import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from numba import jit, prange

# Numba-optimized helper functions

@jit(nopython=True, parallel=True, fastmath=True)
def compute_distances_squared_parallel(points_a, points_b):
    """Fast squared distance computation."""
    n_a, n_b = points_a.shape[0], points_b.shape[0]
    dists = np.zeros((n_a, n_b), dtype=np.float32)
    
    for i in prange(n_a):
        for j in range(n_b):
            dist = 0.0
            for k in range(3):
                diff = points_a[i, k] - points_b[j, k]
                dist += diff * diff
            dists[i, j] = dist
    
    return dists

@jit(nopython=True, fastmath=True)
def compute_wall_angles_numba(normals, vertical):
    """Vectorized angle computation."""
    angles = np.zeros(normals.shape[0], dtype=np.float32)
    for i in range(normals.shape[0]):
        dot_prod = 0.0
        for j in range(3):
            dot_prod += normals[i, j] * vertical[j]
        dot_prod = min(max(dot_prod, -1.0), 1.0)
        angles[i] = np.degrees(np.arccos(dot_prod))
    return angles

@jit(nopython=True, fastmath=True)
def compute_norm_2d_fast(vectors):
    """Fast 2D norm computation."""
    norms = np.zeros(vectors.shape[0], dtype=np.float32)
    for i in range(vectors.shape[0]):
        norms[i] = np.sqrt(vectors[i, 0]**2 + vectors[i, 1]**2)
    return norms

def calculate_all_metrics(vertices, labels, normals):
    """
    Calculates all 10 clinical metrics from segmented tooth preparation.
    Optimized for CPU performance.
    
    Args:
        vertices: (N, 3) array of 3D point coordinates
        labels: (N,) array of class labels (0-6)
        normals: (N, 3) array of vertex normals
    
    Returns:
        dict: Dictionary containing all metric results with values, grades, and colors
    """
    results = {}
    
    # CLASS MAPPING:
    # 0: Background, 1: Intact Tooth, 2: Pulpal Floor, 3: Gingival Floor
    # 4: F-L Walls, 5: Axial Wall, 6: Distal Wall
    
    # ========================================================================
    # METRIC 1: PULPAL DEPTH
    # ========================================================================
    intact_mask = (labels == 1)
    pulpal_mask = (labels == 2)
    
    if np.any(pulpal_mask) and np.any(intact_mask):
        highest_intact_z = vertices[intact_mask, 2].max()
        avg_pulpal_z = np.median(vertices[pulpal_mask, 2])
        depth = float(highest_intact_z - avg_pulpal_z)
        
        if 1.5 <= depth <= 2.0:
            grade, color = "Excellent", "green"
        elif 2.0 < depth <= 4.0:
            grade, color = "Clinically Acceptable", "orange"
        else:
            grade, color = "Standard Not Met", "red"
        
        results["pulpal_depth"] = {
            "value": round(depth, 2),
            "grade": grade,
            "unit": "mm",
            "color": color,
            "description": "Depth from intact surface to pulpal floor"
        }
    else:
        results["pulpal_depth"] = {"value": None, "grade": "N/A", "unit": "mm", "color": "gray"}
    
    # ========================================================================
    # METRIC 2: BUCCO-LINGUAL RATIO
    # ========================================================================
    fl_mask = (labels == 4)
    floor_mask = np.isin(labels, [2, 3, 5, 6])
    
    fl_v = vertices[fl_mask]
    floor_v = vertices[floor_mask]
    intact_v = vertices[intact_mask]
    
    if len(fl_v) >= 20 and len(floor_v) >= 20 and len(intact_v) >= 20:
        try:
            # Calculate isthmus width
            floor_center = floor_v.mean(axis=0)
            centered_floor = floor_v[:, :2] - floor_center[:2]
            _, _, vh = np.linalg.svd(centered_floor)
            md_axis = vh[0]
            bl_axis = np.array([-md_axis[1], md_axis[0]])
            
            side_vecs = fl_v[:, :2] - floor_center[:2]
            side_idx = np.dot(side_vecs, bl_axis) > 0
            side_a, side_b = fl_v[side_idx], fl_v[~side_idx]
            
            if len(side_a) > 0 and len(side_b) > 0:
                # Use optimized distance for large arrays, otherwise cdist
                if len(side_a) * len(side_b) > 10000:
                    dists_sq = compute_distances_squared_parallel(
                        side_a.astype(np.float32), 
                        side_b.astype(np.float32)
                    )
                    dists = np.sqrt(dists_sq)
                else:
                    dists = cdist(side_a, side_b)
                
                m_idx = np.unravel_index(np.argmin(dists), dists.shape)
                isthmus_w = float(dists[m_idx])
                is_p1, is_p2 = side_a[m_idx[0]], side_b[m_idx[1]]
                
                # Calculate intercuspal distance
                mid_p = (is_p1 + is_p2) / 2
                dir_vec = (is_p2 - is_p1)[:2]
                dir_vec /= np.linalg.norm(dir_vec)
                norm_vec = np.array([-dir_vec[1], dir_vec[0]])
                
                corridor_half_width = 2.5
                corridor_mask = np.abs(np.dot(intact_v[:, :2] - mid_p[:2], norm_vec)) < corridor_half_width
                corridor_pts = intact_v[corridor_mask]
                
                if len(corridor_pts) >= 2:
                    projs = np.dot(corridor_pts[:, :2] - mid_p[:2], dir_vec)
                    side1, side2 = corridor_pts[projs > 0], corridor_pts[projs <= 0]
                    
                    if len(side1) > 0 and len(side2) > 0:
                        cusp1 = side1[np.argmax(side1[:, 2])]
                        cusp2 = side2[np.argmax(side2[:, 2])]
                        icd = float(np.linalg.norm(cusp1 - cusp2))
                        bl_ratio = isthmus_w / icd
                        
                        if 0.25 <= bl_ratio <= 0.33:
                            grade, color = "Excellent", "green"
                        elif 0.33 < bl_ratio <= 0.66:
                            grade, color = "Clinically Acceptable", "orange"
                        else:
                            grade, color = "Standard Not Met", "red"
                        
                        results["bl_ratio"] = {
                            "value": round(bl_ratio, 2),
                            "isthmus_width": round(isthmus_w, 2),
                            "intercuspal_distance": round(icd, 2),
                            "grade": grade,
                            "unit": "ratio",
                            "color": color,
                            "description": "Isthmus width to intercuspal distance ratio"
                        }
        except Exception as e:
            pass
    
    if "bl_ratio" not in results:
        results["bl_ratio"] = {"value": None, "grade": "N/A", "unit": "ratio", "color": "gray"}
    
    # ========================================================================
    # METRIC 3: AXIAL WALL HEIGHT
    # ========================================================================
    gingival_mask = (labels == 3)
    
    if np.any(pulpal_mask) and np.any(gingival_mask):
        avg_pulpal_z = np.median(vertices[pulpal_mask, 2])
        avg_gingival_z = np.median(vertices[gingival_mask, 2])
        height = float(abs(avg_pulpal_z - avg_gingival_z))
        
        if 1.0 <= height <= 1.5:
            grade, color = "Excellent", "green"
        elif 0.8 <= height <= 2.0:
            grade, color = "Clinically Acceptable", "orange"
        else:
            grade, color = "Standard Not Met", "red"
        
        results["axial_height"] = {
            "value": round(height, 2),
            "grade": grade,
            "unit": "mm",
            "color": color,
            "description": "Height of axial wall"
        }
    else:
        results["axial_height"] = {"value": None, "grade": "N/A", "unit": "mm", "color": "gray"}
    
    # ========================================================================
    # METRIC 4: WALL TAPER (Convergence Angle)
    # ========================================================================
    fl_mask_taper = (labels == 4)
    floor_mask_taper = np.isin(labels, [2, 3, 5, 6])
    
    fl_v_taper = vertices[fl_mask_taper]
    floor_v_taper = vertices[floor_mask_taper]
    
    if len(fl_v_taper) >= 20 and len(floor_v_taper) >= 20:
        try:
            floor_center_taper = floor_v_taper.mean(axis=0)
            centered_floor_taper = floor_v_taper[:, :2] - floor_center_taper[:2]
            _, _, vh_taper = np.linalg.svd(centered_floor_taper)
            md_axis_taper = vh_taper[0]
            bl_axis_taper = np.array([-md_axis_taper[1], md_axis_taper[0]])
            
            fl_indices_taper = np.where(fl_mask_taper)[0]
            side_vectors_taper = vertices[fl_indices_taper, :2] - floor_center_taper[:2]
            side_idx_taper = np.dot(side_vectors_taper, bl_axis_taper) > 0
            wall_groups = [fl_indices_taper[side_idx_taper], fl_indices_taper[~side_idx_taper]]
            
            angles = []
            vertical = np.array([0, 0, 1])
            
            for indices in wall_groups:
                if len(indices) < 5:
                    continue
                wall_normals = normals[indices]
                avg_normal = np.mean(wall_normals, axis=0)
                avg_normal /= np.linalg.norm(avg_normal)
                angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(avg_normal, vertical), -1.0, 1.0))))
                angles.append(angle_deg)
            
            if len(angles) >= 2:
                divergent = any(a > 93.0 for a in angles)
                if divergent:
                    grade, color = "Standard Not Met", "red"
                elif all(85 <= a <= 93 for a in angles):
                    grade, color = "Excellent", "green"
                else:
                    grade, color = "Clinically Acceptable", "orange"
                
                results["wall_taper"] = {
                    "value": round(np.mean(angles), 1),
                    "angle1": round(angles[0], 1),
                    "angle2": round(angles[1], 1),
                    "grade": grade,
                    "unit": "°",
                    "color": color,
                    "description": "Wall convergence angles"
                }
        except Exception as e:
            pass
    
    if "wall_taper" not in results:
        results["wall_taper"] = {"value": None, "grade": "N/A", "unit": "°", "color": "gray"}
    
    # ========================================================================
    # METRIC 5: MARGINAL RIDGE WIDTH
    # ========================================================================
    intact_v_ridge = vertices[labels == 1]
    cavity_v_ridge = vertices[labels > 1]
    
    if len(intact_v_ridge) >= 20 and len(cavity_v_ridge) >= 20:
        try:
            tooth_center = np.median(intact_v_ridge, axis=0)
            cavity_center = np.median(cavity_v_ridge, axis=0)
            
            md_axis_ridge = tooth_center[:2] - cavity_center[:2]
            md_axis_ridge /= np.linalg.norm(md_axis_ridge)
            
            def measure_ridge_robust(direction, cavity_v, intact_v, cavity_center):
                proj_cav = np.dot(cavity_v[:, :2] - cavity_center[:2], direction)
                limit_proj = np.percentile(proj_cav, 99)
                p_cavity = cavity_v[np.abs(proj_cav - limit_proj).argmin()]
                
                perp_dir = np.array([-direction[1], direction[0]])
                corridor_mask = np.abs(np.dot(intact_v[:, :2] - p_cavity[:2], perp_dir)) < 0.15
                corridor_points = intact_v[corridor_mask]
                
                if len(corridor_points) == 0:
                    return p_cavity, p_cavity, 0.0
                
                proj_int = np.dot(corridor_points[:, :2] - p_cavity[:2], direction)
                p_tooth = corridor_points[np.argmax(proj_int)]
                dist = np.linalg.norm(p_cavity[:2] - p_tooth[:2])
                return p_cavity, p_tooth, dist
            
            _, _, r1 = measure_ridge_robust(md_axis_ridge, cavity_v_ridge, intact_v_ridge, cavity_center)
            _, _, r2 = measure_ridge_robust(-md_axis_ridge, cavity_v_ridge, intact_v_ridge, cavity_center)
            
            valid_ridges = [r for r in [r1, r2] if r > 0.8]
            ridge_val = float(min(valid_ridges)) if valid_ridges else 0.0
            
            if ridge_val >= 1.6:
                grade, color = "Excellent", "green"
            elif ridge_val >= 1.2:
                grade, color = "Clinically Acceptable", "orange"
            else:
                grade, color = "Standard Not Met", "red"
            
            results["marginal_ridge"] = {
                "value": round(ridge_val, 2),
                "grade": grade,
                "unit": "mm",
                "color": color,
                "description": "Minimum marginal ridge width"
            }
        except Exception as e:
            pass
    
    if "marginal_ridge" not in results:
        results["marginal_ridge"] = {"value": None, "grade": "N/A", "unit": "mm", "color": "gray"}
    
    # ========================================================================
    # METRIC 6: WALL SMOOTHNESS (Surface Roughness)
    # ========================================================================
    fl_mask_smooth = (labels == 4)
    fl_indices_smooth = np.where(fl_mask_smooth)[0]
    
    if len(fl_indices_smooth) >= 20:
        try:
            # Create neighbor structure
            tree = cKDTree(vertices)
            roughness_sum = 0
            count = 0
            
            # Sample subset for speed
            sample_size = min(len(fl_indices_smooth), 500)
            sampled_indices = np.random.choice(fl_indices_smooth, sample_size, replace=False)
            
            for idx in sampled_indices:
                _, neighbor_indices = tree.query(vertices[idx], k=10)
                wall_neighbors = [n for n in neighbor_indices if n < len(labels) and labels[n] == 4]
                
                if len(wall_neighbors) > 2:
                    local_normals = normals[wall_neighbors]
                    main_normal = normals[idx]
                    cos_sim = np.dot(local_normals, main_normal)
                    angles = np.arccos(np.clip(cos_sim, -1.0, 1.0))
                    roughness_sum += np.std(angles) + np.mean(angles)
                    count += 1
            
            if count > 0:
                avg_roughness = float(roughness_sum / count)
                
                if avg_roughness < 0.08:
                    grade, color = "Excellent", "green"
                elif avg_roughness < 0.15:
                    grade, color = "Clinically Acceptable", "orange"
                else:
                    grade, color = "Standard Not Met", "red"
                
                results["wall_smoothness"] = {
                    "value": round(avg_roughness, 4),
                    "grade": grade,
                    "unit": "index",
                    "color": color,
                    "description": "Wall surface roughness index"
                }
        except Exception as e:
            pass
    
    if "wall_smoothness" not in results:
        results["wall_smoothness"] = {"value": None, "grade": "N/A", "unit": "index", "color": "gray"}
    
    # ========================================================================
    # METRIC 7: UNDERCUT DETECTION
    # ========================================================================
    wall_mask_undercut = np.isin(labels, [4, 5, 6])
    
    if np.any(wall_mask_undercut):
        insertion_axis = np.array([0, 0, 1])
        wall_normals_undercut = normals[wall_mask_undercut]
        cos_angles_undercut = np.dot(wall_normals_undercut, insertion_axis)
        angles_deg_undercut = np.degrees(np.arccos(np.clip(cos_angles_undercut, -1.0, 1.0)))
        
        undercut_threshold = 92.0
        is_undercut = angles_deg_undercut > undercut_threshold
        undercut_ratio = float((np.sum(is_undercut) / len(angles_deg_undercut)) * 100)
        
        if undercut_ratio < 0.5:
            grade, color = "Excellent", "green"
        elif undercut_ratio < 3.0:
            grade, color = "Clinically Acceptable", "orange"
        else:
            grade, color = "Standard Not Met", "red"
        
        results["undercuts"] = {
            "value": round(undercut_ratio, 1),
            "grade": grade,
            "unit": "%",
            "color": color,
            "description": "Percentage of wall area with undercuts"
        }
    else:
        results["undercuts"] = {"value": None, "grade": "N/A", "unit": "%", "color": "gray"}
    
    # ========================================================================
    # METRIC 8: CUSP UNDERMINING
    # ========================================================================
    intact_v_cusp = vertices[labels == 1]
    fl_v_cusp = vertices[labels == 4]
    
    if len(intact_v_cusp) >= 50 and len(fl_v_cusp) >= 15:
        try:
            center_cusp = np.median(intact_v_cusp, axis=0)
            
            # Define quadrants: MB, DB, DL, ML
            quad_masks_cusp = [
                (intact_v_cusp[:, 0] >= center_cusp[0]) & (intact_v_cusp[:, 1] >= center_cusp[1]),
                (intact_v_cusp[:, 0] < center_cusp[0])  & (intact_v_cusp[:, 1] >= center_cusp[1]),
                (intact_v_cusp[:, 0] < center_cusp[0])  & (intact_v_cusp[:, 1] < center_cusp[1]),
                (intact_v_cusp[:, 0] >= center_cusp[0]) & (intact_v_cusp[:, 1] < center_cusp[1])
            ]
            
            cusp_distances = []
            for mask in quad_masks_cusp:
                q_pts = intact_v_cusp[mask]
                if len(q_pts) < 5:
                    continue
                
                dist_to_c = np.linalg.norm(q_pts[:, :2] - center_cusp[:2], axis=1)
                valid_q_pts = q_pts[(dist_to_c > 3.0) & (dist_to_c < 7.5)]
                
                if len(valid_q_pts) > 0:
                    tip = valid_q_pts[np.argmax(valid_q_pts[:, 2])]
                    dists = np.linalg.norm(fl_v_cusp - tip, axis=1)
                    min_dist = float(np.min(dists))
                    
                    if min_dist < 6.0:
                        cusp_distances.append(min_dist)
            
            if cusp_distances:
                min_cusp_val = float(min(cusp_distances))
                
                if min_cusp_val >= 2.0:
                    grade, color = "Excellent", "green"
                elif min_cusp_val >= 1.5:
                    grade, color = "Clinically Acceptable", "orange"
                else:
                    grade, color = "Standard Not Met", "red"
                
                results["cusp_undermining"] = {
                    "value": round(min_cusp_val, 2),
                    "grade": grade,
                    "unit": "mm",
                    "color": color,
                    "description": "Minimum distance from cusp to cavity wall"
                }
        except Exception as e:
            pass
    
    if "cusp_undermining" not in results:
        results["cusp_undermining"] = {"value": None, "grade": "N/A", "unit": "mm", "color": "gray"}
    
    # ========================================================================
    # METRIC 9: PULPAL FLOOR FLATNESS
    # ========================================================================
    pulpal_v_flat = vertices[pulpal_mask]
    
    if len(pulpal_v_flat) >= 30:
        try:
            # Fit ideal plane
            A = np.column_stack((pulpal_v_flat[:, 0], pulpal_v_flat[:, 1], np.ones(len(pulpal_v_flat))))
            Z = pulpal_v_flat[:, 2]
            coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            a, b, c = coeffs
            
            ideal_Z = a * pulpal_v_flat[:, 0] + b * pulpal_v_flat[:, 1] + c
            deviations = np.abs(pulpal_v_flat[:, 2] - ideal_Z)
            flatness_score = float(np.std(deviations))
            max_dev = float(np.max(deviations))
            
            if flatness_score < 0.12:
                grade, color = "Excellent", "green"
            elif flatness_score < 0.25:
                grade, color = "Clinically Acceptable", "orange"
            else:
                grade, color = "Standard Not Met", "red"
            
            results["floor_flatness"] = {
                "value": round(flatness_score, 4),
                "max_deviation": round(max_dev, 3),
                "grade": grade,
                "unit": "mm",
                "color": color,
                "description": "Pulpal floor flatness score"
            }
        except Exception as e:
            pass
    
    if "floor_flatness" not in results:
        results["floor_flatness"] = {"value": None, "grade": "N/A", "unit": "mm", "color": "gray"}
    
    
    return results