"""Web GUI Dashboard & Interactive Simulator for LiDAR-Camera Projection QA (Topic A).

Cung cấp giao diện trực quan cao cấp (Ultra-premium Dark Theme), hỗ trợ:
1. Mô phỏng trực tiếp Extrinsic Calibration Drift với thanh trượt mượt mà (Yaw, Pitch, Roll, Translation).
2. Tính toán thời gian thực các chỉ số: Pixel Shift, Point Retention (Near/Far), Edge Alignment.
3. Chế độ kiểm tra Failure Case & Phân tích nguyên nhân hình học (Geometry Layer).
4. Khám phá số liệu Benchmark và so sánh đa tập dữ liệu (KITTI vs nuScenes).
"""
from __future__ import annotations

import base64
import copy
import io
import sys
from pathlib import Path

# Thêm root repo vào sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from starter.datasets import dataset_type, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import (
    cam_to_image,
    draw_box2d,
    overlay_points,
    perturb_extrinsic,
    project_velo_to_image,
    velo_to_cam,
)

app = Flask(__name__, static_folder="static", static_url_path="")

# Cache dữ liệu frame trong RAM để phản hồi cực nhanh (<20ms)
FRAME_CACHE: dict[str, dict] = {}


def get_cached_frame(dataset_name: str, frame_id: str) -> dict:
    key = f"{dataset_name}:{frame_id}"
    if key in FRAME_CACHE:
        return FRAME_CACHE[key]

    if dataset_name == "kitti":
        root = ROOT / "data" / "kitti_mini"
    elif dataset_name == "nusc":
        root = ROOT / "data" / "nuscenes_mini_subset"
    else:
        root = ROOT / "data" / "synthetic"

    data = load_frame(str(root), frame_id)
    pts = data["points"][:, :3]
    calib = data["calib"]
    img = data["image"]
    labels = [l for l in data.get("labels", []) if getattr(l, "type", "") != "DontCare"]

    # Tiền tính baseline unperturbed
    uv0, d0, m0 = project_velo_to_image(pts, calib, img.shape)

    # Tiền tính Canny edge map cho ảnh để tính edge alignment score nhanh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dist_map = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

    # Gán nhãn cho từng điểm xem thuộc object nào ở baseline
    N = len(pts)
    point_obj_ids = np.full(N, -1, dtype=np.int32)
    near_obj_ids = set()
    far_obj_ids = set()

    for idx, obj in enumerate(labels):
        dist = float(np.linalg.norm(obj.location)) if hasattr(obj, "location") else float(np.median(d0))
        if dist <= 15.0:
            near_obj_ids.add(idx)
        elif dist > 20.0:
            far_obj_ids.add(idx)

        b = obj.bbox
        in_b = (uv0[:, 0] >= b[0]) & (uv0[:, 0] <= b[2]) & (uv0[:, 1] >= b[1]) & (uv0[:, 1] <= b[3])
        full_m = np.zeros(N, dtype=bool)
        full_m[m0] = in_b
        point_obj_ids[full_m] = idx

    cached = {
        "dataset": dataset_name,
        "frame_id": frame_id,
        "points": pts,
        "calib": calib,
        "image": img,
        "labels": labels,
        "dist_map": dist_map,
        "uv_base": uv0,
        "mask_base": m0,
        "point_obj_ids": point_obj_ids,
        "near_obj_ids": near_obj_ids,
        "far_obj_ids": far_obj_ids,
    }
    FRAME_CACHE[key] = cached
    return cached


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/frames", methods=["GET"])
def get_available_frames():
    return jsonify({
        "kitti": ["000011", "000001", "000004", "000007", "000021", "000049"],
        "synthetic": ["000000", "000001", "000002"],
        "nusc": ["scene-0103_010", "scene-0103_000", "scene-1094_000"],
    })


@app.route("/api/project", methods=["POST"])
def live_project():
    req = request.get_json() or {}
    dataset_name = req.get("dataset", "kitti")
    frame_id = req.get("frame_id", "000011")

    yaw = float(req.get("yaw", 0.0))
    pitch = float(req.get("pitch", 0.0))
    roll = float(req.get("roll", 0.0))
    tx = float(req.get("tx", 0.0))
    ty = float(req.get("ty", 0.0))
    tz = float(req.get("tz", 0.0))

    show_boxes = bool(req.get("show_boxes", True))
    show_points = bool(req.get("show_points", True))
    point_radius = int(req.get("point_radius", 2))
    max_depth = float(req.get("max_depth", 50.0))

    data = get_cached_frame(dataset_name, frame_id)
    pts = data["points"]
    calib = data["calib"]
    img = data["image"]
    labels = data["labels"]
    dist_map = data["dist_map"]
    N = len(pts)

    # 1. Perturb extrinsic
    calib_pert = perturb_extrinsic(calib, roll_deg=roll, pitch_deg=pitch,
                                   yaw_deg=yaw, t_xyz_m=(tx, ty, tz))

    # 2. Project
    uv_pert, depth_pert, mask_pert = project_velo_to_image(pts, calib_pert, img.shape)

    # 3. Tính Metrics
    uv_base = data["uv_base"]
    mask_base = data["mask_base"]

    # Pixel shift
    both_valid = mask_base & mask_pert
    if np.any(both_valid):
        uv_base_full = np.zeros((N, 2), dtype=np.float32)
        uv_base_full[mask_base] = uv_base
        uv_pert_full = np.zeros((N, 2), dtype=np.float32)
        uv_pert_full[mask_pert] = uv_pert
        diff = uv_pert_full[both_valid] - uv_base_full[both_valid]
        shifts = np.linalg.norm(diff, axis=1)
        mean_shift = float(np.mean(shifts))
        max_shift = float(np.max(shifts))
        p95_shift = float(np.percentile(shifts, 95))
    else:
        mean_shift = 0.0
        max_shift = 0.0
        p95_shift = 0.0

    # True Point Retention per Object
    point_obj_ids = data["point_obj_ids"]
    obj_stats = []
    total_obj_base = 0
    total_obj_retained = 0
    near_base = 0
    near_retained = 0
    far_base = 0
    far_retained = 0

    # Ánh xạ điểm trong perturbed state vào bounding boxes
    in_box_pert_per_obj = {}
    if len(uv_pert) > 0:
        u_p = uv_pert[:, 0]
        v_p = uv_pert[:, 1]
        for idx, obj in enumerate(labels):
            b = obj.bbox
            in_b = (u_p >= b[0]) & (u_p <= b[2]) & (v_p >= b[1]) & (v_p <= b[3])
            full_m = np.zeros(N, dtype=bool)
            full_m[mask_pert] = in_b
            in_box_pert_per_obj[idx] = full_m

    for idx, obj in enumerate(labels):
        base_mask = (point_obj_ids == idx)
        n_base = int(np.sum(base_mask))
        if n_base == 0:
            continue
        total_obj_base += n_base
        pert_m = in_box_pert_per_obj.get(idx, np.zeros(N, dtype=bool))
        n_retained = int(np.sum(base_mask & pert_m))
        total_obj_retained += n_retained
        ret_rate = (n_retained / n_base * 100.0)

        dist = float(np.linalg.norm(obj.location)) if hasattr(obj, "location") else 0.0
        if idx in data["near_obj_ids"]:
            near_base += n_base
            near_retained += n_retained
        elif idx in data["far_obj_ids"]:
            far_base += n_base
            far_retained += n_retained

        obj_stats.append({
            "id": idx,
            "type": getattr(obj, "type", "Object"),
            "distance": round(dist, 1),
            "bbox": [round(float(v), 1) for v in obj.bbox],
            "base_points": n_base,
            "retained_points": n_retained,
            "retention_rate": round(ret_rate, 1),
        })

    retention_all = (total_obj_retained / total_obj_base * 100.0) if total_obj_base > 0 else 100.0
    retention_near = (near_retained / near_base * 100.0) if near_base > 0 else 100.0
    retention_far = (far_retained / far_base * 100.0) if far_base > 0 else 100.0

    # Edge alignment score
    if len(uv_pert) > 0:
        H, W = img.shape[:2]
        u_int = np.clip(np.round(uv_pert[:, 0]).astype(int), 0, W - 1)
        v_int = np.clip(np.round(uv_pert[:, 1]).astype(int), 0, H - 1)
        edge_score = float(np.mean(np.exp(-dist_map[v_int, u_int] / 5.0)))
    else:
        edge_score = 0.0

    # Phân loại trạng thái hệ thống
    if abs(yaw) <= 0.3 and abs(pitch) <= 0.3 and mean_shift < 6.0:
        status = "OPTIMAL"
        status_color = "#00e676"  # Emerald
        status_msg = "Calibration chuẩn xác. Mọi đối tượng bám khớp hoàn hảo."
    elif abs(yaw) <= 1.0 and abs(pitch) <= 1.0 and retention_all > 70.0:
        status = "WARNING"
        status_color = "#ffb300"  # Amber
        status_msg = "Phát hiện sai lệch nhẹ. Đối tượng ở xa (>20m) bắt đầu suy giảm điểm."
    else:
        status = "CRITICAL"
        status_color = "#ff3366"  # Crimson
        status_msg = "CẢNH BÁO LỆCH NGHIÊM TRỌNG! Điểm vật thể xa trượt hoàn toàn khỏi 2D box."

    # 4. Vẽ Overlay
    rendered = img.copy()
    if show_points and len(uv_pert) > 0:
        rendered = overlay_points(rendered, uv_pert, depth_pert, max_depth=max_depth, radius=point_radius)

    if show_boxes:
        for item in obj_stats:
            bbox = item["bbox"]
            r_rate = item["retention_rate"]
            # Đổi màu box theo retention: xanh nếu tốt, vàng nếu giảm, đỏ nếu mất điểm
            if r_rate >= 80.0:
                color = (0, 230, 118)  # Xanh
            elif r_rate >= 40.0:
                color = (0, 179, 255)  # Vàng
            else:
                color = (102, 51, 255)  # Đỏ
            label_text = f"{item['type']} ({item['distance']}m) {r_rate:.0f}%"
            rendered = draw_box2d(rendered, bbox, color=color, label=label_text)

    # Encode JPEG base64
    _, buf = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpg_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "image": jpg_b64,
        "metrics": {
            "mean_pixel_shift": round(mean_shift, 2),
            "max_pixel_shift": round(max_shift, 2),
            "p95_pixel_shift": round(p95_shift, 2),
            "retention_all_pct": round(retention_all, 1),
            "retention_near_pct": round(retention_near, 1),
            "retention_far_pct": round(retention_far, 1),
            "edge_alignment_score": round(edge_score, 4),
            "points_inside_fov": len(uv_pert),
            "status": status,
            "status_color": status_color,
            "status_msg": status_msg,
        },
        "objects": obj_stats,
    })


@app.route("/api/benchmark-data", methods=["GET"])
def get_benchmark_data():
    csv_sweep = ROOT / "results" / "calibration_drift_sweep.csv"
    csv_cross = ROOT / "results" / "cross_dataset_comparison.csv"

    sweep_data = pd.read_csv(csv_sweep).to_dict(orient="records") if csv_sweep.exists() else []
    cross_data = pd.read_csv(csv_cross).to_dict(orient="records") if csv_cross.exists() else []

    return jsonify({
        "sweep": sweep_data,
        "cross_dataset": cross_data,
    })


if __name__ == "__main__":
    port = 5050
    print("\n=======================================================")
    print("[SERVER] LiDAR-Camera Projection QA Interactive GUI is running!")
    print(f"[URL] Mo trinh duyet tai: http://127.0.0.1:{port}")
    print("=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
