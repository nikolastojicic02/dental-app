



from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
import uuid
import torch
import numpy as np
import trimesh
import __main__

class Config:
    DETECTION_FEATURE_DIM = 256
    SEGMENTATION_FEATURE_DIM = 256
    NUM_CLASSES = 7
    FIXED_BOX_SIZE = [12.0, 14.0, 12.0]
    NUM_POINTS_DETECTION = 4096
    NUM_POINTS_SEGMENTATION = 8192
    CLASS_NAMES = ["Background", "Intact Tooth", "Pulpal Floor", "Gingival Floor", 
                   "F-L Walls", "Axial Wall", "Distal Wall"]

__main__.Config = Config

from metrics_engine import calculate_all_metrics
from models_arch import PointNetPlusBBoxDetector, PointNetPlusPlusSegmentation
from inference_utils import (
    load_stl_for_inference, rotation_6d_to_matrix, transform_to_local,
    farthest_point_sampling, apply_knn_smoothing, refine_segmentation
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
MODEL_DIR = "weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(UPLOAD_DIR, exist_ok=True)

det_model = None
seg_model = None
sessions = {}

@app.on_event("startup")
async def load_models():
    global det_model, seg_model
    print(f"--- Loading Models on {DEVICE} ---")
    
    det_model = PointNetPlusBBoxDetector(feature_dim=256).to(DEVICE)
    det_path = os.path.join(MODEL_DIR, "best_detection_model.pth")
    det_ckpt = torch.load(det_path, map_location=DEVICE, weights_only=False)
    det_model.load_state_dict(det_ckpt['model_state_dict'])
    det_model.eval()
    
    seg_model = PointNetPlusPlusSegmentation(num_classes=7).to(DEVICE)
    seg_path = os.path.join(MODEL_DIR, "best_seg_model.pth")
    seg_ckpt = torch.load(seg_path, map_location=DEVICE, weights_only=False)
    seg_model.load_state_dict(seg_ckpt['model_state_dict'])
    seg_model.eval()
    
    print("--- All Models Loaded Successfully ---")

# ============================================================================
# STAGE 0: UPLOAD ONLY (PREVIEW STL)
# ============================================================================

@app.post("/stage0/upload")
async def stage0_upload(file: UploadFile = File(...)):
    """Stage 0: Upload STL and return mesh for preview"""
    if not file.filename.endswith('.stl'):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    session_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{session_id}.stl")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        mesh = trimesh.load(temp_path)
        
        sessions[session_id] = {
            "stl_path": temp_path,
            "filename": file.filename,
            "original_mesh": mesh
        }
        
        return {
            "status": "success",
            "session_id": session_id,
            "stage": "upload",
            "data": {
                "mesh_vertices": mesh.vertices.tolist(),
                "mesh_faces": mesh.faces.tolist(),
                "filename": file.filename
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================================
# STAGE 1: DETECTION
# ============================================================================

@app.post("/stage1/detect/{session_id}")
async def stage1_detect(session_id: str):
    """Stage 1: Detect bounding box on uploaded STL"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    temp_path = session["stl_path"]

    try:
        features, centroid, original_mesh = load_stl_for_inference(temp_path, num_points=4096)
        input_det = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_center_local, pred_r6d = det_model(input_det)
            pred_center_local = pred_center_local.cpu().numpy()[0]
            pred_r6d = pred_r6d.cpu().numpy()[0]
        
        pred_center_world = pred_center_local + centroid
        pred_orientation = rotation_6d_to_matrix(pred_r6d)
        
        # Update session
        session["pred_center"] = pred_center_world
        session["pred_orientation"] = pred_orientation
        session["centroid"] = centroid
        
        # Calculate bbox corners
        half_size = np.array(Config.FIXED_BOX_SIZE) / 2
        corners_local = np.array([
            [-half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], -half_size[1], -half_size[2]],
            [+half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], +half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], -half_size[1], +half_size[2]],
            [+half_size[0], +half_size[1], +half_size[2]],
            [-half_size[0], +half_size[1], +half_size[2]]
        ])
        bbox_corners = (pred_orientation @ corners_local.T).T + pred_center_world
        
        return {
            "status": "success",
            "stage": "detection",
            "data": {
                "mesh_vertices": original_mesh.vertices.tolist(),
                "mesh_faces": original_mesh.faces.tolist(),
                "bbox_center": pred_center_world.tolist(),
                "bbox_corners": bbox_corners.tolist(),
                "bbox_size": Config.FIXED_BOX_SIZE,
                "jaw_type": "lower" if "lower" in session["filename"].lower() else "upper"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================================
# STAGE 2: CROPPING
# ============================================================================

@app.post("/stage2/crop/{session_id}")
async def stage2_crop(session_id: str):
    """Stage 2: Crop tooth using detected bbox"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    original_mesh = session["original_mesh"]
    pred_center_world = session["pred_center"]
    pred_orientation = session["pred_orientation"]
    
    try:
        vertices_local = transform_to_local(
            original_mesh.vertices, 
            pred_center_world, 
            pred_orientation
        )
        
        half_size = np.array([6.0, 7.0, 6.0])
        mask = np.all(np.abs(vertices_local) <= half_size, axis=1)
        
        if not np.any(mask):
            return {"status": "error", "message": "No vertices inside bbox"}
        
        needs_flip = "upper" in session["filename"].lower()
        if needs_flip:
            flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            vertices_local = vertices_local @ flip_matrix.T
        
        cropped_vertices = vertices_local[mask]
        cropped_faces_indices = np.where(mask)[0]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(cropped_faces_indices)}
        
        valid_faces = []
        for face in original_mesh.faces:
            if all(v in old_to_new for v in face):
                valid_faces.append([old_to_new[v] for v in face])
        
        cropped_mesh = trimesh.Trimesh(
            vertices=cropped_vertices,
            faces=np.array(valid_faces) if valid_faces else np.array([]),
            process=True
        )
        cropped_mesh.vertex_normals
        
        session["cropped_mesh"] = cropped_mesh
        session["cropped_vertices"] = cropped_vertices
        session["cropped_normals"] = cropped_mesh.vertex_normals[:len(cropped_vertices)]
        
        return {
            "status": "success",
            "stage": "cropping",
            "data": {
                "mesh_vertices": cropped_mesh.vertices.tolist(),
                "mesh_faces": cropped_mesh.faces.tolist(),
                "original_vertices": len(original_mesh.vertices),
                "cropped_vertices": len(cropped_mesh.vertices),
                "retention_pct": (len(cropped_mesh.vertices) / len(original_mesh.vertices)) * 100
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# STAGE 3: SEGMENTATION (REPLACE YOUR EXISTING stage3_segment FUNCTION)
# ============================================================================

@app.post("/stage3/segment/{session_id}")
async def stage3_segment(session_id: str):
    """Stage 3: Segment tooth parts"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    cropped_mesh = session["cropped_mesh"]
    
    try:
        # Get vertices and normals from the FULL MESH (not sampled yet)
        full_vertices = np.array(cropped_mesh.vertices, dtype=np.float32)
        full_normals = np.array(cropped_mesh.vertex_normals, dtype=np.float32)
        
        print(f"Full mesh: {len(full_vertices)} vertices")
        
        # Sample 8192 points for model inference
        indices = farthest_point_sampling(full_vertices, 8192)
        sampled_v = full_vertices[indices]
        sampled_n = full_normals[indices]
        
        # Normalize for model
        seg_centroid = sampled_v.mean(axis=0)
        sampled_v_norm = sampled_v - seg_centroid
        seg_features = np.hstack((sampled_v_norm, sampled_n)).astype(np.float32)
        
        # Run segmentation model
        input_seg = torch.from_numpy(seg_features).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = seg_model(input_seg)
            labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Post-processing on SAMPLED points
        refined_labels = apply_knn_smoothing(sampled_v, labels, k=20)
        final_labels = refine_segmentation(sampled_v, refined_labels)
        
        print(f"Sampled predictions: {len(final_labels)} labels")
        
        # Class distribution
        unique, counts = np.unique(final_labels, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        
        # Color mapping
        color_map = {
            0: [254, 254, 254], 1: [127, 127, 127], 2: [1, 0, 254],
            3: [0, 254, 254], 4: [254, 0, 0], 5: [0, 254, 0], 6: [254, 254, 0]
        }
        
        # CRITICAL FIX: Map labels from sampled points to FULL MESH using KNN
        from scipy.spatial import cKDTree
        
        # Build KD-tree on sampled points
        tree = cKDTree(sampled_v)
        
        # For each vertex in the full mesh, find closest sampled point
        distances, closest_indices = tree.query(full_vertices, k=1)
        
        # Transfer labels from sampled to full mesh
        full_labels = final_labels[closest_indices]
        
        # Generate colors for FULL mesh
        full_vertex_colors = np.array([color_map[label] for label in full_labels])
        
        # DEBUG: Check label distribution on FULL mesh
        full_unique, full_counts = np.unique(full_labels, return_counts=True)
        print(f"✓ Mapped to full mesh: {len(full_vertex_colors)} colors")
        print(f"Full mesh label distribution:")
        for cls, count in zip(full_unique, full_counts):
            pct = (count / len(full_labels)) * 100
            print(f"  Class {cls} ({Config.CLASS_NAMES[cls]}): {count} vertices ({pct:.1f}%)")
        
        # Save for metrics stage
        session["sampled_vertices"] = sampled_v
        session["final_labels"] = final_labels
        session["sampled_normals"] = sampled_n
        session["full_labels"] = full_labels  # NEW: Save full mesh labels
        
        return {
            "status": "success",
            "stage": "segmentation",
            "data": {
                "mesh_vertices": cropped_mesh.vertices.tolist(),
                "mesh_faces": cropped_mesh.faces.tolist(),
                "vertex_colors": full_vertex_colors.tolist(),  # FULL mesh colors
                "labels": final_labels.tolist(),  # Sampled labels (for reference)
                "class_distribution": class_distribution,
                "class_names": Config.CLASS_NAMES
            }
        }
    except Exception as e:
        import traceback
        print(f"ERROR in stage3_segment: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

# ============================================================================
# STAGE 4: METRICS
# ============================================================================

@app.post("/stage4/metrics/{session_id}")
async def stage4_metrics(session_id: str):
    """Stage 4: Calculate all metrics with visualizations"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Check if segmentation was done
    if "sampled_vertices" not in session:
        return {"status": "error", "message": "Please run segmentation first"}
    
    vertices = session["sampled_vertices"]
    labels = session["final_labels"]
    normals = session["sampled_normals"]
    cropped_mesh = session["cropped_mesh"]
    
    try:
        # Calculate metrics
        metrics = calculate_all_metrics(vertices, labels, normals)
        
        # Generate visualization data for each metric
        visualizations = generate_metric_visualizations(vertices, labels, normals, metrics, cropped_mesh)
        
        return {
            "status": "success",
            "stage": "metrics",
            "filename": session["filename"],
            "data": {
                "metrics": metrics,
                "visualizations": visualizations
            }
        }
    except Exception as e:
        import traceback
        print(f"ERROR in stage4_metrics: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

def generate_metric_visualizations(vertices, labels, normals, metrics, cropped_mesh):
    """
    Generates complete visualizations for all metrics - matching frontend implementation
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    import numpy as np
    
    # Original color map
    color_map = {
        0: [254, 254, 254], 1: [127, 127, 127], 2: [1, 0, 254],
        3: [0, 254, 254],   4: [254, 0, 0],     5: [0, 254, 0], 6: [254, 254, 0]
    }
    
    # Map labels to full mesh
    tree = cKDTree(vertices)
    _, closest_indices = tree.query(cropped_mesh.vertices, k=1)
    full_labels = labels[closest_indices]
    
    # Generate colors for full mesh
    full_vertex_colors = np.array([color_map[lbl] for lbl in full_labels])
    
    visualizations = {}
    
    # ============================================================================
    # 1. PULPAL DEPTH
    # ============================================================================
    if metrics.get("pulpal_depth", {}).get("value") is not None:
        intact_mask = (labels == 1)
        pulpal_mask = (labels == 2)
        
        top_idx = np.argmax(vertices[intact_mask, 2])
        highest_point = vertices[intact_mask][top_idx]
        avg_pulpal_z = np.median(vertices[pulpal_mask, 2])
        
        depth_color_name = metrics["pulpal_depth"]["color"]
        hex_colors = {"green": "#008000", "orange": "#ffa500", "red": "#ff0000"}
        arrow_color = hex_colors.get(depth_color_name, "#ffa500")

        visualizations["pulpal_depth"] = {
            "mesh_vertices": cropped_mesh.vertices.tolist(),
            "mesh_faces": cropped_mesh.faces.tolist(),
            "vertex_colors": full_vertex_colors.tolist(),
            "measurement": {
                "start": highest_point.tolist(),
                "end": [float(highest_point[0]), float(highest_point[1]), float(avg_pulpal_z)],
                "color": arrow_color,
                "value_text": f"{metrics['pulpal_depth']['value']:.2f} mm"
            }
        }
    
    # ============================================================================
    # 2. BL RATIO
    # ============================================================================
 # ============================================================================
    # 2. BL RATIO - ZAKRIVLJENA RAZDELNA RAVAN
    # ============================================================================
    if metrics.get("bl_ratio", {}).get("value") is not None:
        fl_mask = (labels == 4)
        floor_mask = np.isin(labels, [2, 3, 5, 6])
        intact_mask = (labels == 1)
        
        fl_v = vertices[fl_mask]
        floor_v = vertices[floor_mask]
        intact_v = vertices[intact_mask]
        
        if len(fl_v) >= 20 and len(floor_v) >= 50 and len(intact_v) >= 20:
            # --- TRANSFORMACIJA U LOKALNI KOORDINATNI SISTEM ---
            center = floor_v.mean(axis=0)
            centered_floor = floor_v - center
            
            # SVD na podnim tačkama da dobijemo glavne ose
            _, _, vh = np.linalg.svd(centered_floor[:, :2])
            md_axis = vh[0]  # Mezio-distalna osa (glavna osa preparacije)
            bl_axis = np.array([-md_axis[1], md_axis[0]])  # Bukko-lingvalna osa (normalna na md)
            
            # Projektujemo podne tačke na ove ose
            x_floor = np.dot(centered_floor[:, :2], md_axis)
            y_floor = np.dot(centered_floor[:, :2], bl_axis)
            
            # --- FITOVANJE KRIVE (Središnja linija preparacije) ---
            # Polinom 2. stepena opisuje kako pod "vrluda"
            poly_coeffs = np.polyfit(x_floor, y_floor, deg=2)
            poly_func = np.poly1d(poly_coeffs)
            
            # --- PODELA ZIDOVA NA OSNOVU ZAKRIVLJENE LINIJE ---
            centered_wall = fl_v - center
            x_wall = np.dot(centered_wall[:, :2], md_axis)
            y_wall = np.dot(centered_wall[:, :2], bl_axis)
            
            # Rastojanje od krive - tačke iznad krive su bukalne, ispod lingvalne
            dist_from_curve = y_wall - poly_func(x_wall)
            
            # Threshold za podelu (0.5mm)
            side_a = fl_v[dist_from_curve > 0.5]  # Bukalna strana
            side_b = fl_v[dist_from_curve < -0.5]  # Lingvalna strana
            
            if len(side_a) > 0 and len(side_b) > 0:
                # Pronalaženje isthmusa (najuži prostor između strana)
                dists = cdist(side_a, side_b)
                m_idx = np.unravel_index(np.argmin(dists), dists.shape)
                is_p1, is_p2 = side_a[m_idx[0]], side_b[m_idx[1]]
                
                # --- ICD LINIJA (kao pre) ---
                mid_p = (is_p1 + is_p2) / 2
                dir_vec = (is_p2 - is_p1)[:2]
                dir_vec /= np.linalg.norm(dir_vec)
                norm_vec = np.array([-dir_vec[1], dir_vec[0]])
                
                corridor_mask = np.abs(np.dot(intact_v[:, :2] - mid_p[:2], norm_vec)) < 2.5
                corridor_pts = intact_v[corridor_mask]
                
                icd_line = None
                if len(corridor_pts) >= 2:
                    projs = np.dot(corridor_pts[:, :2] - mid_p[:2], dir_vec)
                    side1, side2 = corridor_pts[projs > 0], corridor_pts[projs <= 0]
                    
                    if len(side1) > 0 and len(side2) > 0:
                        cusp1 = side1[np.argmax(side1[:, 2])]
                        cusp2 = side2[np.argmax(side2[:, 2])]
                        icd_line = {
                            "start": cusp1.tolist(),
                            "end": cusp2.tolist(),
                            "value_text": f"{metrics['bl_ratio'].get('intercuspal_distance', 0):.2f} mm"
                        }
                
                visualizations["bl_ratio"] = {
                    "mesh_vertices": cropped_mesh.vertices.tolist(),
                    "mesh_faces": cropped_mesh.faces.tolist(),
                    "vertex_colors": full_vertex_colors.tolist(),
                    "isthmus_line": {
                        "start": is_p1.tolist(),
                        "end": is_p2.tolist(),
                        "value_text": f"{metrics['bl_ratio'].get('isthmus_width', 0):.2f} mm"
                    },
                    "icd_line": icd_line
                }
    # ============================================================================
    # 3. AXIAL HEIGHT
    # ============================================================================
    if metrics.get("axial_height", {}).get("value") is not None:
        pulpal_mask = (labels == 2)
        gingival_mask = (labels == 3)
        
        pulpal_v = vertices[pulpal_mask]
        gingival_v = vertices[gingival_mask]
        
        gingival_center = gingival_v.mean(axis=0)
        height_value = metrics["axial_height"]["value"]
        
        ax_color = metrics["axial_height"]["color"]
        hex_colors = {"green": "#00ff00", "orange": "#ffa500", "red": "#ff0000"}
        arrow_color = hex_colors.get(ax_color, "#ffa500")
        
        visualizations["axial_height"] = {
            "mesh_vertices": cropped_mesh.vertices.tolist(),
            "mesh_faces": cropped_mesh.faces.tolist(),
            "vertex_colors": full_vertex_colors.tolist(),
            "gingival_center": gingival_center.tolist(),
            "height_value": float(height_value),
            "color": arrow_color
        }
    
    # ============================================================================
    # 4. WALL TAPER
    # ============================================================================
    if metrics.get("wall_taper", {}).get("value") is not None:
        fl_mask_taper = (labels == 4)
        floor_mask_taper = np.isin(labels, [2, 3, 5, 6])
        
        fl_v_taper = vertices[fl_mask_taper]
        floor_v_taper = vertices[floor_mask_taper]
        
        if len(fl_v_taper) >= 20 and len(floor_v_taper) >= 20:
            floor_center_taper = floor_v_taper.mean(axis=0)
            centered_floor_taper = floor_v_taper[:, :2] - floor_center_taper[:2]
            _, _, vh_taper = np.linalg.svd(centered_floor_taper)
            md_axis_taper = vh_taper[0]
            bl_axis_taper = np.array([-md_axis_taper[1], md_axis_taper[0]])
            
            fl_indices_taper = np.where(fl_mask_taper)[0]
            side_vectors_taper = vertices[fl_indices_taper, :2] - floor_center_taper[:2]
            side_idx_taper = np.dot(side_vectors_taper, bl_axis_taper) > 0
            wall_groups = [fl_indices_taper[side_idx_taper], fl_indices_taper[~side_idx_taper]]
            
            measurement_lines = []
            vertical = np.array([0, 0, 1])
            
            taper_color = metrics["wall_taper"]["color"]
            hex_colors = {"green": "#008000", "orange": "#ffa500", "red": "#ff0000"}
            line_color = hex_colors.get(taper_color, "#ffa500")
            
            angles = []
            if "angle1" in metrics["wall_taper"]:
                angles.append(metrics["wall_taper"]["angle1"])
            if "angle2" in metrics["wall_taper"]:
                angles.append(metrics["wall_taper"]["angle2"])
            
            for idx, indices in enumerate(wall_groups):
                if len(indices) < 5:
                    continue
                pts = vertices[indices]
                wall_normals = normals[indices]
                avg_normal = np.mean(wall_normals, axis=0)
                avg_normal /= np.linalg.norm(avg_normal)
                
                z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
                center_xy = pts.mean(axis=0)[:2]
                offset_vector = -avg_normal[:2] * (z_max - z_min) * 0.3
                p_start = np.array([center_xy[0] - offset_vector[0], center_xy[1] - offset_vector[1], z_min])
                p_end = np.array([center_xy[0] + offset_vector[0], center_xy[1] + offset_vector[1], z_max])
                
                angle_val = angles[idx] if idx < len(angles) else metrics["wall_taper"]["value"]
                
                measurement_lines.append({
                    "start": p_start.tolist(),
                    "end": p_end.tolist(),
                    "color": line_color,
                    "value_text": f"{angle_val:.1f}°"
                })
            
            visualizations["wall_taper"] = {
                "mesh_vertices": cropped_mesh.vertices.tolist(),
                "mesh_faces": cropped_mesh.faces.tolist(),
                "vertex_colors": full_vertex_colors.tolist(),
                "measurement_lines": measurement_lines
            }
    
    # ============================================================================
    # 5. MARGINAL RIDGE
    # ============================================================================
    if metrics.get("marginal_ridge", {}).get("value") is not None:
        intact_v_ridge = vertices[labels == 1]
        cavity_v_ridge = vertices[labels > 1]
        
        if len(intact_v_ridge) >= 20 and len(cavity_v_ridge) >= 20:
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
            
            p1_c, p1_t, r1 = measure_ridge_robust(md_axis_ridge, cavity_v_ridge, intact_v_ridge, cavity_center)
            p2_c, p2_t, r2 = measure_ridge_robust(-md_axis_ridge, cavity_v_ridge, intact_v_ridge, cavity_center)
            
            measurement_lines = []
            for pc, pt, val in [(p1_c, p1_t, r1), (p2_c, p2_t, r2)]:
                if val > 0.8:
                    if val >= 1.6:
                        l_col = "#00ff00"
                    elif val >= 1.2:
                        l_col = "#ffa500"
                    else:
                        l_col = "#ff0000"
                    
                    measurement_lines.append({
                        "start": pc.tolist(),
                        "end": pt.tolist(),
                        "color": l_col,
                        "value_text": f"{val:.2f} mm"
                    })
            
            visualizations["marginal_ridge"] = {
                "mesh_vertices": cropped_mesh.vertices.tolist(),
                "mesh_faces": cropped_mesh.faces.tolist(),
                "vertex_colors": full_vertex_colors.tolist(),
                "measurement_lines": measurement_lines
            }

    # ============================================================================
    # 6. WALL SMOOTHNESS - INTERPOLACIJA NA FULL MESH
    # ============================================================================
    if metrics.get("wall_smoothness", {}).get("value") is not None:
        fl_mask_smooth = (labels == 4)
        fl_indices_smooth = np.where(fl_mask_smooth)[0]
        
        roughness_values = np.zeros(len(vertices))
        
        if len(fl_indices_smooth) >= 20:
            print(f"[SMOOTHNESS] Processing {len(fl_indices_smooth)} wall vertices")
            
            # Rekonstruiši faces
            sampled_faces = []
            tree_sampled = cKDTree(vertices)
            
            for face in cropped_mesh.faces:
                face_verts = cropped_mesh.vertices[face]
                _, sampled_idx = tree_sampled.query(face_verts, k=1)
                if len(set(sampled_idx)) == 3:
                    sampled_faces.append(sampled_idx)
            
            if len(sampled_faces) > 0:
                import trimesh
                sampled_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=np.array(sampled_faces),
                    process=False
                )
                
                print(f"[SMOOTHNESS] Created sampled mesh: {len(sampled_mesh.vertices)} verts, {len(sampled_mesh.faces)} faces")
                
                processed = 0
                for idx in fl_indices_smooth:
                    neighbors = sampled_mesh.vertex_neighbors[idx]
                    wall_neighbors = [n for n in neighbors if n < len(labels) and labels[n] == 4]
                    
                    if len(wall_neighbors) > 2:
                        local_normals = normals[wall_neighbors]
                        main_normal = normals[idx]
                        cos_sim = np.dot(local_normals, main_normal)
                        angles = np.arccos(np.clip(cos_sim, -1.0, 1.0))
                        roughness_values[idx] = np.std(angles) + np.mean(angles)
                        processed += 1
                
                print(f"[SMOOTHNESS] Processed: {processed}/{len(fl_indices_smooth)}")
                
                # Interpolacija missing
                missing_count = len(fl_indices_smooth) - processed
                if missing_count > 0:
                    print(f"[SMOOTHNESS] Interpolating {missing_count} missing vertices...")
                    missing_mask = (fl_mask_smooth) & (roughness_values == 0)
                    missing_indices = np.where(missing_mask)[0]
                    processed_mask = (fl_mask_smooth) & (roughness_values > 0)
                    processed_indices = np.where(processed_mask)[0]
                    
                    if len(processed_indices) > 0:
                        tree_processed = cKDTree(vertices[processed_indices])
                        for idx in missing_indices:
                            _, neighbors = tree_processed.query(vertices[idx], k=min(5, len(processed_indices)))
                            neighbor_idx = processed_indices[neighbors]
                            roughness_values[idx] = np.mean(roughness_values[neighbor_idx])
                        print(f"[SMOOTHNESS] ✓ Interpolated {len(missing_indices)} vertices")
                
                # Skaliranje
                non_zero = np.sum(roughness_values > 0)
                if non_zero > 0:
                    non_zero_vals = roughness_values[roughness_values > 0]
                    min_r = non_zero_vals.min()
                    max_r = non_zero_vals.max()
                    mean_r = non_zero_vals.mean()
                    median_r = np.median(non_zero_vals)
                    
                    print(f"[SMOOTHNESS] RAW stats:")
                    print(f"  Min: {min_r:.4f}, Max: {max_r:.4f}, Mean: {mean_r:.4f}, Median: {median_r:.4f}")
                    
                    p5 = np.percentile(non_zero_vals, 5)
                    p95 = np.percentile(non_zero_vals, 95)
                    print(f"  P5: {p5:.4f}, P95: {p95:.4f}")
                    
                    roughness_values_clipped = np.clip(roughness_values, p5, p95)
                    mask_nonzero = roughness_values > 0
                    roughness_values[mask_nonzero] = (
                        (roughness_values_clipped[mask_nonzero] - p5) / (p95 - p5)
                    ) * 0.2
                    
                    new_min = roughness_values[roughness_values > 0].min()
                    new_max = roughness_values[roughness_values > 0].max()
                    new_mean = roughness_values[roughness_values > 0].mean()
                    print(f"[SMOOTHNESS] ✓ Scaled range: {new_min:.4f} - {new_max:.4f} (mean: {new_mean:.4f})")
            else:
                print("[SMOOTHNESS] ⚠️ No valid sampled faces created!")
        
        # Mapiranje na full mesh
        roughness_mapped = roughness_values[closest_indices]
        full_labels_for_filter = labels[closest_indices]
        
        # ⚠️ KRITIČNO: Interpoliraj roughness ZA SVE WALL VERTICES U FULL MESH
        full_wall_mask = (full_labels_for_filter == 4)
        full_wall_indices = np.where(full_wall_mask)[0]
        
        print(f"[SMOOTHNESS] Full mesh wall vertices: {len(full_wall_indices)}")
        print(f"[SMOOTHNESS] Non-zero roughness in full mesh: {np.sum(roughness_mapped > 0)}")
        
        # Ako ima wall vertices sa roughness = 0, interpoliraj ih
        missing_in_full = np.sum((full_wall_mask) & (roughness_mapped == 0))
        if missing_in_full > 0:
            print(f"[SMOOTHNESS] Interpolating {missing_in_full} missing vertices in full mesh...")
            
            # Vertices sa roughness > 0
            has_roughness_mask = (full_wall_mask) & (roughness_mapped > 0)
            has_roughness_indices = np.where(has_roughness_mask)[0]
            
            if len(has_roughness_indices) > 0:
                tree_full = cKDTree(cropped_mesh.vertices[has_roughness_indices])
                
                # Za sve wall vertices bez roughness
                missing_full_indices = np.where((full_wall_mask) & (roughness_mapped == 0))[0]
                
                for idx in missing_full_indices:
                    # Nađi 3 najbliža vertices sa roughness
                    _, neighbors = tree_full.query(cropped_mesh.vertices[idx], k=min(3, len(has_roughness_indices)))
                    neighbor_idx = has_roughness_indices[neighbors]
                    # Weighted average po distanci
                    dists = np.linalg.norm(cropped_mesh.vertices[idx] - cropped_mesh.vertices[neighbor_idx], axis=1)
                    weights = 1.0 / (dists + 1e-6)
                    weights /= weights.sum()
                    roughness_mapped[idx] = np.sum(roughness_mapped[neighbor_idx] * weights)
                
                print(f"[SMOOTHNESS] ✓ Interpolated {len(missing_full_indices)} vertices in full mesh")
        
        # Sad postavi 0 za non-wall
        roughness_mapped[full_labels_for_filter != 4] = 0.0
        
        # Filtriranje faces
        filtered_faces = []
        for face in cropped_mesh.faces:
            if all(full_labels_for_filter[v_idx] == 4 for v_idx in face):
                # Svi vertices moraju biti wall I bar jedan mora imati roughness
                if any(roughness_mapped[v_idx] > 0.0001 for v_idx in face):
                    filtered_faces.append(face.tolist())
        
        print(f"[SMOOTHNESS] Filtered faces: {len(filtered_faces)}/{len(cropped_mesh.faces)}")
        
        visualizations["wall_smoothness"] = {
            "mesh_vertices": cropped_mesh.vertices.tolist(),
            "mesh_faces": cropped_mesh.faces.tolist(),
            "vertex_colors": full_vertex_colors.tolist(),
            "heatmap_vertices": cropped_mesh.vertices.tolist(),
            "heatmap_faces": filtered_faces,
            "heatmap_intensity": roughness_mapped.tolist(),
            "ghost_vertices": cropped_mesh.vertices.tolist(),
            "ghost_faces": cropped_mesh.faces.tolist(),
            "ghost_colors": full_vertex_colors.tolist()
        }
    

    # 7. UNDERCUTS - COLAB IDENTIČNA VERZIJA
    # ============================================================================
    if metrics.get("undercuts", {}).get("value") is not None:
        wall_mask = np.isin(labels, [4, 5, 6])
        wall_indices = np.where(wall_mask)[0]
        
        if len(wall_indices) >= 20:
            insertion_axis = np.array([0, 0, 1])
            
            # Kreiraj undercut map (0 = safe, >92 = angle vrednost)
            undercut_map = np.zeros(len(vertices))
            
            wall_normals = normals[wall_mask]
            cos_angles = np.dot(wall_normals, insertion_axis)
            angles_deg = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
            
            # KLJUČNO: Postavi angle SAMO za undercuts (>92°)
            undercut_threshold = 92.0
            is_undercut = angles_deg > undercut_threshold
            
            # Samo undercut vertices dobijaju angle vrednost
            undercut_map[wall_indices[is_undercut]] = angles_deg[is_undercut]
            
            print(f"[UNDERCUTS] Total wall vertices: {len(wall_indices)}")
            print(f"[UNDERCUTS] Undercut vertices: {np.sum(is_undercut)}")
            print(f"[UNDERCUTS] Undercut ratio: {(np.sum(is_undercut) / len(wall_indices)) * 100:.1f}%")
            
            # Mapiranje na full mesh
            undercut_mapped = undercut_map[closest_indices]
            full_labels_for_filter = labels[closest_indices]
            
            # Interpolacija missing undercuts (kao kod smoothness)
            full_wall_mask = np.isin(full_labels_for_filter, [4, 5, 6])
            full_wall_indices = np.where(full_wall_mask)[0]
            
            # Interpoliraj missing vertices
            has_undercut_mask = (full_wall_mask) & (undercut_mapped > 0)
            has_undercut_indices = np.where(has_undercut_mask)[0]
            
            if len(has_undercut_indices) > 0:
                missing_undercut = np.sum((full_wall_mask) & (undercut_mapped == 0))
                if missing_undercut > 0:
                    print(f"[UNDERCUTS] Interpolating {missing_undercut} missing vertices...")
                    
                    tree_undercut = cKDTree(cropped_mesh.vertices[has_undercut_indices])
                    missing_indices = np.where((full_wall_mask) & (undercut_mapped == 0))[0]
                    
                    for idx in missing_indices:
                        # Proveri da li treba interpolirati (ako su susedi undercuts)
                        _, neighbors = tree_undercut.query(cropped_mesh.vertices[idx], k=min(3, len(has_undercut_indices)))
                        neighbor_idx = has_undercut_indices[neighbors]
                        dists = np.linalg.norm(cropped_mesh.vertices[idx] - cropped_mesh.vertices[neighbor_idx], axis=1)
                        
                        # Interpoliraj samo ako je najbliži undercut vertex blizu
                        if dists[0] < 0.5:  # Threshold za blizinu
                            weights = 1.0 / (dists + 1e-6)
                            weights /= weights.sum()
                            undercut_mapped[idx] = np.sum(undercut_mapped[neighbor_idx] * weights)
                    
                    print(f"[UNDERCUTS] ✓ Interpolated {len(missing_indices)} vertices")
            
            # Postavi 0 za non-wall
            undercut_mapped[~full_wall_mask] = 0.0
            
            # Filtriranje faces - samo wall faces
            filtered_faces = []
            for face in cropped_mesh.faces:
                # Svi vertices moraju biti wall
                if all(np.isin(full_labels_for_filter[v_idx], [4, 5, 6]) for v_idx in face):
                    filtered_faces.append(face.tolist())
            
            print(f"[UNDERCUTS] Filtered wall faces: {len(filtered_faces)}/{len(cropped_mesh.faces)}")
            
            visualizations["undercuts"] = {
                "mesh_vertices": cropped_mesh.vertices.tolist(),
                "mesh_faces": cropped_mesh.faces.tolist(),
                "vertex_colors": full_vertex_colors.tolist(),
                "heatmap_vertices": cropped_mesh.vertices.tolist(),
                "heatmap_faces": filtered_faces,
                "heatmap_intensity": undercut_mapped.tolist(),
                "ghost_vertices": cropped_mesh.vertices.tolist(),
                "ghost_faces": cropped_mesh.faces.tolist(),
                "ghost_colors": full_vertex_colors.tolist()
            }
        
    # ============================================================================
    # 8. CUSP UNDERMINING
    # ============================================================================
    if metrics.get("cusp_undermining", {}).get("value") is not None:
        intact_v_cusp = vertices[labels == 1]
        fl_v_cusp = vertices[labels == 4]
        
        if len(intact_v_cusp) >= 50 and len(fl_v_cusp) >= 15:
            center_cusp = np.median(intact_v_cusp, axis=0)
            
            quad_masks_cusp = [
                (intact_v_cusp[:, 0] >= center_cusp[0]) & (intact_v_cusp[:, 1] >= center_cusp[1]),
                (intact_v_cusp[:, 0] < center_cusp[0])  & (intact_v_cusp[:, 1] >= center_cusp[1]),
                (intact_v_cusp[:, 0] < center_cusp[0])  & (intact_v_cusp[:, 1] < center_cusp[1]),
                (intact_v_cusp[:, 0] >= center_cusp[0]) & (intact_v_cusp[:, 1] < center_cusp[1])
            ]
            
            measurement_lines = []
            for mask in quad_masks_cusp:
                q_pts = intact_v_cusp[mask]
                if len(q_pts) < 5:
                    continue
                
                dist_to_c = np.linalg.norm(q_pts[:, :2] - center_cusp[:2], axis=1)
                valid_q_pts = q_pts[(dist_to_c > 3.5) & (dist_to_c < 7.5)]
                
                if len(valid_q_pts) > 0:
                    tip = valid_q_pts[np.argmax(valid_q_pts[:, 2])]
                    dists = np.linalg.norm(fl_v_cusp - tip, axis=1)
                    min_idx = np.argmin(dists)
                    val = float(dists[min_idx])
                    
                    if val < 6.0:
                        if val >= 2.0:
                            l_col = "#00ff00"
                        elif val >= 1.5:
                            l_col = "#ffa500"
                        else:
                            l_col = "#ff0000"
                        
                        measurement_lines.append({
                            "start": tip.tolist(),
                            "end": fl_v_cusp[min_idx].tolist(),
                            "color": l_col,
                            "value_text": f"{val:.2f} mm"
                        })
            
            visualizations["cusp_undermining"] = {
                "mesh_vertices": cropped_mesh.vertices.tolist(),
                "mesh_faces": cropped_mesh.faces.tolist(),
                "vertex_colors": full_vertex_colors.tolist(),
                "measurement_lines": measurement_lines
            }
    
    # ============================================================================
    # 9. FLOOR FLATNESS
    # ============================================================================
    if metrics.get("floor_flatness", {}).get("value") is not None:
        pulpal_mask = (labels == 2)
        pulpal_v = vertices[pulpal_mask]
        
        if len(pulpal_v) >= 30:
            # Fit plane
            A = np.column_stack((pulpal_v[:, 0], pulpal_v[:, 1], np.ones(len(pulpal_v))))
            Z = pulpal_v[:, 2]
            coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
            a, b, c = coeffs
            
            ideal_Z = a * pulpal_v[:, 0] + b * pulpal_v[:, 1] + c
            deviations = np.abs(pulpal_v[:, 2] - ideal_Z)
            
            heatmap_intensity = np.zeros(len(vertices))
            heatmap_intensity[pulpal_mask] = deviations
            
            # Find max deviation point
            max_dev_idx = np.argmax(deviations)
            max_point = pulpal_v[max_dev_idx]
            plane_point = np.array([max_point[0], max_point[1], ideal_Z[max_dev_idx]])
            
            # Plane visualization
            plane_center = pulpal_v.mean(axis=0)
            plane_size = np.max(np.linalg.norm(pulpal_v[:, :2] - plane_center[:2], axis=1)) * 2
            
            visualizations["floor_flatness"] = {
                "mesh_vertices": cropped_mesh.vertices.tolist(),
                "mesh_faces": cropped_mesh.faces.tolist(),
                "vertex_colors": full_vertex_colors.tolist(),
                "heatmap_vertices": cropped_mesh.vertices.tolist(),
                "heatmap_faces": cropped_mesh.faces.tolist(),
                "heatmap_intensity": heatmap_intensity[closest_indices].tolist(),
                "ghost_vertices": cropped_mesh.vertices.tolist(),
                "ghost_faces": cropped_mesh.faces.tolist(),
                "ghost_colors": full_vertex_colors.tolist(),
                "ideal_plane": {
                    "center": [float(plane_center[0]), float(plane_center[1]), float(a * plane_center[0] + b * plane_center[1] + c)],
                    "size": float(plane_size)
                },
                "max_deviation_line": {
                    "start": max_point.tolist(),
                    "end": plane_point.tolist(),
                    "value_text": f"{metrics['floor_flatness'].get('max_deviation', 0):.3f} mm"
                }
            }
    
    
    return visualizations
# ============================================================================
# CLEANUP
# ============================================================================

@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up session data"""
    if session_id in sessions:
        stl_path = sessions[session_id].get("stl_path")
        if stl_path and os.path.exists(stl_path):
            os.remove(stl_path)
        del sessions[session_id]
        return {"status": "success", "message": "Session cleaned up"}
    return {"status": "error", "message": "Session not found"}




# ============================================================================
# SERVIRANJE FRONTA 
# ============================================================================
# Putanja do "dist" foldera (gde React napravi build)
# U Dockeru će struktura biti takva da je frontend folder pored backend foldera
frontend_dist_path = os.path.join(os.getcwd(), "..", "frontend", "dist")

# Ako folder postoji (na serveru), montiraj ga
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static")

    # Catch-all ruta za React Router (ako koristiš stranice na frontu)
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        # Ako ruta ne počinje sa "stage" ili "cleanup" (tvoj API), pošalji index.html
        if not full_path.startswith(("stage", "cleanup", "docs", "redoc", "openapi.json")):
            return FileResponse(os.path.join(frontend_dist_path, "index.html"))













