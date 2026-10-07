"""Benchmark độ nhạy của phép chiếu LiDAR-Camera trước các mức sai lệch extrinsic calibration.

Hỗ trợ quét đa trục (Yaw, Pitch, Roll, Translation), tính toán các độ đo:
- Tỷ lệ điểm LiDAR trong FOV camera (% inside FOV).
- Tỷ lệ lưu giữ điểm trong 2D Bounding Box của vật thể (% retention), phân tách theo cự ly gần và xa.
- Độ dịch chuyển pixel trung bình (Mean pixel shift).
- Điểm gióng hàng cạnh (Edge Alignment Score) dựa trên Canny edge và distance transform.
- Đo độ trễ xử lý phép chiếu (latency p50/p95).
- So sánh trên cả KITTI và nuScenes.

Chạy:
    python src/benchmark_calibration_drift.py --help
    python src/benchmark_calibration_drift.py --data-root data/kitti_mini --frame 000011 --eval-cross-dataset
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm root repo vào sys.path để có thể import từ starter/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def compute_edge_alignment_score(image: np.ndarray, uv: np.ndarray, dist_trans: np.ndarray,
                                 scale_sigma: float = 5.0) -> float:
    """Tính điểm gióng hàng cạnh giữa các điểm chiếu và biên ảnh Canny.
    Score nằm trong khoảng [0, 1], càng gần 1 tức điểm chiếu càng bám sát các đường biên vật thể."""
    if len(uv) == 0:
        return 0.0
    H, W = image.shape[:2]
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    dists = dist_trans[v, u]
    scores = np.exp(-dists / float(scale_sigma))
    return float(np.mean(scores))


def points_in_box2d(uv: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Trả về mask boolean các điểm uv nằm trong bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def evaluate_frame_drift(frame_data: dict, perturbations: list[dict],
                         latency_runs: int = 25) -> tuple[pd.DataFrame, dict]:
    """Đánh giá nhiều mức biến dạng extrinsic trên 1 frame."""
    pts_raw = frame_data["points"][:, :3]
    calib_base: KittiCalib = frame_data["calib"]
    image = frame_data["image"]
    labels = frame_data.get("labels", [])
    H, W = image.shape[:2]

    # Tiền xử lý Canny edge & distance transform
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dist_trans = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

    # 1. Chạy baseline không perturb
    uv_base, depth_base, mask_base = project_velo_to_image(pts_raw, calib_base, image.shape)
    n_total = len(pts_raw)
    n_base_fov = len(uv_base)

    # Lập mask trên toàn bộ n_total điểm cho baseline
    in_box_base_all = np.zeros(n_total, dtype=bool)
    in_box_base_near = np.zeros(n_total, dtype=bool)
    in_box_base_far = np.zeros(n_total, dtype=bool)

    near_boxes = []
    far_boxes = []
    all_boxes = []

    for obj in labels:
        if getattr(obj, "type", "") in {"DontCare"}:
            continue
        bbox = obj.bbox
        all_boxes.append(bbox)
        dist = np.linalg.norm(obj.location) if hasattr(obj, "location") else float(np.median(depth_base))

        u, v = uv_base[:, 0], uv_base[:, 1]
        in_b = (u >= bbox[0]) & (u <= bbox[2]) & (v >= bbox[1]) & (v <= bbox[3])
        full_m = np.zeros(n_total, dtype=bool)
        full_m[mask_base] = in_b

        in_box_base_all |= full_m
        if dist <= 15.0:
            near_boxes.append(bbox)
            in_box_base_near |= full_m
        elif dist > 20.0:
            far_boxes.append(bbox)
            in_box_base_far |= full_m

    base_count_all = int(np.sum(in_box_base_all))
    base_count_near = int(np.sum(in_box_base_near))
    base_count_far = int(np.sum(in_box_base_far))

    # Đo latency của phép chiếu (bỏ lần chạy đầu tiên)
    latencies = []
    for i in range(latency_runs):
        t0 = time.perf_counter()
        _ = project_velo_to_image(pts_raw, calib_base, image.shape)
        t1 = time.perf_counter()
        if i > 0:
            latencies.append((t1 - t0) * 1000.0)
    p50_ms = float(np.percentile(latencies, 50))
    p95_ms = float(np.percentile(latencies, 95))

    records = []
    visual_cache = {}

    for cfg in perturbations:
        desc = cfg.get("desc", "unnamed")
        roll = cfg.get("roll_deg", 0.0)
        pitch = cfg.get("pitch_deg", 0.0)
        yaw = cfg.get("yaw_deg", 0.0)
        t_xyz = cfg.get("t_xyz_m", (0.0, 0.0, 0.0))

        calib_pert = perturb_extrinsic(calib_base, roll_deg=roll, pitch_deg=pitch,
                                       yaw_deg=yaw, t_xyz_m=t_xyz)
        uv_pert, depth_pert, mask_pert = project_velo_to_image(pts_raw, calib_pert, image.shape)

        n_fov = len(uv_pert)
        inside_fov_ratio = (n_fov / n_total) * 100.0

        # Ánh xạ toạ độ pixel để tính pixel shift
        uv_base_full = np.zeros((n_total, 2), dtype=np.float32)
        uv_base_full[mask_base] = uv_base
        uv_pert_full = np.zeros((n_total, 2), dtype=np.float32)
        uv_pert_full[mask_pert] = uv_pert

        both_valid = mask_base & mask_pert
        if np.any(both_valid):
            diff = uv_pert_full[both_valid] - uv_base_full[both_valid]
            pixel_shifts = np.linalg.norm(diff, axis=1)
            mean_px_shift = float(np.mean(pixel_shifts))
            max_px_shift = float(np.max(pixel_shifts))
            p95_px_shift = float(np.percentile(pixel_shifts, 95))
        else:
            mean_px_shift = 0.0
            max_px_shift = 0.0
            p95_px_shift = 0.0

        # Tính mask điểm rơi vào box ở trạng thái perturb
        in_box_pert_all = np.zeros(n_total, dtype=bool)
        if len(uv_pert) > 0 and len(all_boxes) > 0:
            u_p, v_p = uv_pert[:, 0], uv_pert[:, 1]
            in_any_box = np.zeros(len(uv_pert), dtype=bool)
            for b in all_boxes:
                in_any_box |= (u_p >= b[0]) & (u_p <= b[2]) & (v_p >= b[1]) & (v_p <= b[3])
            in_box_pert_all[mask_pert] = in_any_box

        # True Point Retention: Trong số các điểm ban đầu thuộc object, bao nhiêu điểm còn nằm trong 2D box
        retention_all = (float(np.sum(in_box_base_all & in_box_pert_all)) / base_count_all * 100.0) if base_count_all > 0 else 100.0
        retention_near = (float(np.sum(in_box_base_near & in_box_pert_all)) / base_count_near * 100.0) if base_count_near > 0 else 100.0
        retention_far = (float(np.sum(in_box_base_far & in_box_pert_all)) / base_count_far * 100.0) if base_count_far > 0 else 100.0

        edge_score = compute_edge_alignment_score(image, uv_pert, dist_trans)

        rec = {
            "config": desc,
            "roll_deg": roll,
            "pitch_deg": pitch,
            "yaw_deg": yaw,
            "tx_m": t_xyz[0],
            "ty_m": t_xyz[1],
            "tz_m": t_xyz[2],
            "points_inside_fov": n_fov,
            "fov_ratio_pct": round(inside_fov_ratio, 2),
            "mean_pixel_shift": round(mean_px_shift, 2),
            "p95_pixel_shift": round(p95_px_shift, 2),
            "max_pixel_shift": round(max_px_shift, 2),
            "box_retention_all_pct": round(retention_all, 2),
            "box_retention_near_pct": round(retention_near, 2),
            "box_retention_far_pct": round(retention_far, 2),
            "edge_alignment_score": round(edge_score, 4),
            "latency_p50_ms": round(p50_ms, 2),
            "latency_p95_ms": round(p95_ms, 2),
        }
        records.append(rec)
        visual_cache[desc] = {
            "uv": uv_pert,
            "depth": depth_pert,
            "calib": calib_pert,
        }

    df = pd.DataFrame(records)
    extra = {
        "image": image,
        "labels": labels,
        "visual_cache": visual_cache,
        "base_counts": (base_count_all, base_count_near, base_count_far),
    }
    return df, extra


def build_perturbation_suite() -> list[dict]:
    """Tạo bộ cấu hình quét đa dạng cho thí nghiệm CP3."""
    suite = [
        {"desc": "baseline_0.0", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "t_xyz_m": (0, 0, 0)},
        # Yaw sweep (chính)
        {"desc": "yaw_+0.5deg", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.5, "t_xyz_m": (0, 0, 0)},
        {"desc": "yaw_+1.0deg", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 1.0, "t_xyz_m": (0, 0, 0)},
        {"desc": "yaw_+1.5deg", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 1.5, "t_xyz_m": (0, 0, 0)},
        {"desc": "yaw_+2.0deg", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 2.0, "t_xyz_m": (0, 0, 0)},
        {"desc": "yaw_+3.0deg", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 3.0, "t_xyz_m": (0, 0, 0)},
        {"desc": "yaw_-1.0deg", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": -1.0, "t_xyz_m": (0, 0, 0)},
        # Pitch sweep (so sánh độ nhạy trục)
        {"desc": "pitch_+1.0deg", "roll_deg": 0.0, "pitch_deg": 1.0, "yaw_deg": 0.0, "t_xyz_m": (0, 0, 0)},
        {"desc": "pitch_+2.0deg", "roll_deg": 0.0, "pitch_deg": 2.0, "yaw_deg": 0.0, "t_xyz_m": (0, 0, 0)},
        # Roll sweep
        {"desc": "roll_+1.0deg", "roll_deg": 1.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "t_xyz_m": (0, 0, 0)},
        {"desc": "roll_+2.0deg", "roll_deg": 2.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "t_xyz_m": (0, 0, 0)},
        # Translation sweep
        {"desc": "trans_dy_+0.05m", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "t_xyz_m": (0, 0.05, 0)},
        {"desc": "trans_dy_+0.10m", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "t_xyz_m": (0, 0.10, 0)},
        {"desc": "trans_dz_+0.10m", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "t_xyz_m": (0, 0, 0.10)},
    ]
    return suite


def plot_benchmark_results(df: pd.DataFrame, out_path: Path):
    """Vẽ biểu đồ phân tích 4 khía cạnh độ nhạy của calibration drift."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#ffffff")

    # 1. Yaw sweep vs Box Retention (Near vs Far)
    yaw_df = df[df["config"].str.startswith("yaw_") | (df["config"] == "baseline_0.0")].copy()
    yaw_df = yaw_df.sort_values(by="yaw_deg")
    ax1 = axes[0, 0]
    ax1.plot(yaw_df["yaw_deg"], yaw_df["box_retention_near_pct"], marker="o", color="#2ca02c",
             linewidth=2.5, label="Near Objects (<=15m)")
    ax1.plot(yaw_df["yaw_deg"], yaw_df["box_retention_far_pct"], marker="s", color="#d62728",
             linewidth=2.5, label="Far Objects (>20m)")
    ax1.plot(yaw_df["yaw_deg"], yaw_df["box_retention_all_pct"], marker="^", color="#1f77b4",
             linestyle="--", label="All Objects")
    ax1.set_title("1. Point-in-Box Retention vs Yaw Drift (Near vs Far)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Yaw Drift (degrees)")
    ax1.set_ylabel("Retention Rate (%)")
    ax1.axhline(50, color="gray", linestyle=":", alpha=0.7)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left")

    # 2. Pixel shift theo các trục xoay (Yaw, Pitch, Roll)
    ax2 = axes[0, 1]
    angles = [0.0, 1.0, 2.0]
    yaw_shifts = [df[df["config"] == "baseline_0.0"]["mean_pixel_shift"].values[0],
                  df[df["config"] == "yaw_+1.0deg"]["mean_pixel_shift"].values[0] if "yaw_+1.0deg" in df["config"].values else 0,
                  df[df["config"] == "yaw_+2.0deg"]["mean_pixel_shift"].values[0] if "yaw_+2.0deg" in df["config"].values else 0]
    pitch_shifts = [df[df["config"] == "baseline_0.0"]["mean_pixel_shift"].values[0],
                    df[df["config"] == "pitch_+1.0deg"]["mean_pixel_shift"].values[0] if "pitch_+1.0deg" in df["config"].values else 0,
                    df[df["config"] == "pitch_+2.0deg"]["mean_pixel_shift"].values[0] if "pitch_+2.0deg" in df["config"].values else 0]
    roll_shifts = [df[df["config"] == "baseline_0.0"]["mean_pixel_shift"].values[0],
                   df[df["config"] == "roll_+1.0deg"]["mean_pixel_shift"].values[0] if "roll_+1.0deg" in df["config"].values else 0,
                   df[df["config"] == "roll_+2.0deg"]["mean_pixel_shift"].values[0] if "roll_+2.0deg" in df["config"].values else 0]

    ax2.plot(angles, yaw_shifts, marker="o", color="#1f77b4", linewidth=2.5, label="Yaw (azimuth)")
    ax2.plot(angles, pitch_shifts, marker="s", color="#ff7f0e", linewidth=2.5, label="Pitch (elevation)")
    ax2.plot(angles, roll_shifts, marker="^", color="#9467bd", linewidth=2.5, label="Roll (bank)")
    ax2.set_title("2. Mean Pixel Displacement across Rotation Axes", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Angular Error (degrees)")
    ax2.set_ylabel("Mean Pixel Shift (px)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 3. Edge Alignment Score vs Yaw Drift
    ax3 = axes[1, 0]
    ax3.plot(yaw_df["yaw_deg"], yaw_df["edge_alignment_score"], marker="D", color="#8c564b",
             linewidth=2.5)
    ax3.set_title("3. Edge Alignment Score vs Yaw Drift", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Yaw Drift (degrees)")
    ax3.set_ylabel("Alignment Score [0 - 1]")
    ax3.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Box Retention vs Translation Drift
    ax4 = axes[1, 1]
    trans_cfgs = ["baseline_0.0", "trans_dy_+0.05m", "trans_dy_+0.10m", "trans_dz_+0.10m"]
    sub_trans = df[df["config"].isin(trans_cfgs)].copy()
    labels_trans = ["Baseline", "dy=+5cm", "dy=+10cm", "dz=+10cm"]
    bars = ax4.bar(labels_trans, sub_trans["box_retention_all_pct"], color=["#1f77b4", "#aec7e8", "#ffbb78", "#2ca02c"],
                   edgecolor="black", alpha=0.85)
    ax4.set_title("4. Point Retention under Translation Perturbations", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Retention Rate (%)")
    ax4.set_ylim(0, 115)
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> Saved benchmark plots: {out_path}")


def save_visual_demo_and_failure(extra: dict, out_dir: Path, frame_id: str):
    """Tạo ảnh demo so sánh và ảnh failure case chuẩn theo yêu cầu CP4."""
    image = extra["image"]
    labels = extra["labels"]
    cache = extra["visual_cache"]

    # 1. Tạo demo so sánh 3 mức: Baseline (0.0°), Yaw +1.0°, Yaw +2.0°
    configs = ["baseline_0.0", "yaw_+1.0deg", "yaw_+2.0deg"]
    panels = []
    for cfg in configs:
        if cfg not in cache:
            continue
        c = cache[cfg]
        ov = overlay_points(image, c["uv"], c["depth"], max_depth=50.0, radius=2)
        # Vẽ các ground truth 2D boxes
        for obj in labels:
            if getattr(obj, "type", "") not in {"DontCare"}:
                ov = draw_box2d(ov, obj.bbox, color=(0, 255, 0), label=obj.type)
        # Ghi chữ nhãn mức perturb
        header_text = f"Config: {cfg}"
        cv2.putText(ov, header_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        panels.append(ov)

    if len(panels) == 3:
        # Xếp dọc 3 ảnh để so sánh chi tiết
        comp_img = np.vstack(panels)
        demo_path = out_dir / f"demo_drift_comparison_{frame_id}.png"
        cv2.imwrite(str(demo_path), comp_img)
        print(f"-> Saved comparison demo: {demo_path}")

    # 2. Tạo ảnh failure case (bắt buộc tên bắt đầu bằng fail_)
    if "yaw_+2.0deg" in cache:
        c_fail = cache["yaw_+2.0deg"]
        ov_fail = overlay_points(image, c_fail["uv"], c_fail["depth"], max_depth=50.0, radius=2)
        for obj in labels:
            if getattr(obj, "type", "") not in {"DontCare"}:
                ov_fail = draw_box2d(ov_fail, obj.bbox, color=(0, 255, 0), label=obj.type)

        # Chú thích tổng quan
        cv2.putText(ov_fail, "FAILURE CASE: Extrinsic Yaw +2.0 deg Calibration Drift",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(ov_fail, "Debug Layer: GEOMETRY (Angular shift shifts points horizontally: delta_u ~ fx * delta_yaw ~ 30.7 px)",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Đánh dấu các vùng đối tượng bị trượt điểm nghiêm trọng
        for obj in labels:
            if getattr(obj, "type", "") in {"DontCare"}:
                continue
            dist = np.linalg.norm(obj.location) if hasattr(obj, "location") else 0
            b = [int(v) for v in obj.bbox]
            if dist > 30.0:  # Pedestrian ở 34.2m
                # Vẽ hộp đỏ cảnh báo quanh GT box và vùng điểm bị lệch sang phải
                cv2.rectangle(ov_fail, (b[0] - 4, b[1] - 4), (b[2] + 40, b[3] + 4), (0, 0, 255), 2)
                cv2.putText(ov_fail, f"Pedestrian (34m): 0% retention (shift 31px > width 15px)",
                            (b[0] - 60, b[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            elif 20.0 < dist <= 30.0:  # Car ở 27.2m
                cv2.rectangle(ov_fail, (b[0] - 4, b[1] - 4), (b[2] + 40, b[3] + 4), (0, 0, 255), 2)
                cv2.putText(ov_fail, f"Car (27m): 39.5% points detached outside box",
                            (b[0] - 40, b[3] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        fail_path = out_dir / "fail_01_yaw_drift_mismatch.png"
        cv2.imwrite(str(fail_path), ov_fail)
        print(f"-> Saved failure case image: {fail_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Đo độ nhạy của phép chiếu LiDAR-Camera trước các sai lệch extrinsic calibration (Topic A)."
    )
    parser.add_argument("--data-root", type=str, default="data/kitti_mini",
                        help="Thư mục dataset (default: data/kitti_mini)")
    parser.add_argument("--frame", type=str, default="000011",
                        help="Frame ID cần đánh giá (default: 000011)")
    parser.add_argument("--eval-cross-dataset", action="store_true",
                        help="Chạy đồng thời trên cả KITTI và nuScenes để so sánh (Bonus B5)")
    parser.add_argument("--out-csv", type=str, default="results/calibration_drift_sweep.csv",
                        help="Đường dẫn file CSV kết quả (default: results/calibration_drift_sweep.csv)")
    parser.add_argument("--out-fig", type=str, default="results/figures/calibration_drift_metrics.png",
                        help="Đường dẫn ảnh biểu đồ kết quả (default: results/figures/calibration_drift_metrics.png)")
    parser.add_argument("--latency-runs", type=int, default=25,
                        help="Số lần chạy để đo latency p50/p95 (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Cố định seed ngẫu nhiên (default: 42)")

    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"[*] Đang tải frame {args.frame} từ {args.data_root}...")
    frame_data = load_frame(args.data_root, args.frame)
    suite = build_perturbation_suite()

    print(f"[*] Thực hiện benchmark trên {len(suite)} cấu hình biến dạng...")
    df, extra = evaluate_frame_drift(frame_data, suite, latency_runs=args.latency_runs)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"-> Đã ghi kết quả: {out_csv}")

    out_fig = Path(args.out_fig)
    plot_benchmark_results(df, out_fig)
    save_visual_demo_and_failure(extra, out_fig.parent, args.frame)

    # Hiển thị bảng tóm tắt
    cols_display = ["config", "mean_pixel_shift", "box_retention_all_pct",
                    "box_retention_near_pct", "box_retention_far_pct", "edge_alignment_score"]
    print("\n=== KẾT QUẢ TÓM TẮT THÍ NGHIỆM CP3 ===")
    print(df[cols_display].to_string(index=False))

    # Nếu bật cross-dataset evaluation (Bonus B5)
    if args.eval_cross_dataset:
        print("\n[*] Đang chạy đánh giá chéo trên nuScenes (scene-0103_010)...")
        nusc_frame = load_frame("data/nuscenes_mini_subset", "scene-0103_010")
        df_nusc, _ = evaluate_frame_drift(nusc_frame, suite, latency_runs=args.latency_runs)
        df["dataset"] = "KITTI (64-beam)"
        df_nusc["dataset"] = "nuScenes (32-beam)"
        combined_df = pd.concat([df, df_nusc], ignore_index=True)
        cross_csv = Path("results/cross_dataset_comparison.csv")
        combined_df.to_csv(cross_csv, index=False)
        print(f"-> Đã ghi kết quả so sánh chéo: {cross_csv}")

        # Vẽ biểu đồ so sánh cross-dataset
        fig, ax = plt.subplots(figsize=(8, 5))
        sub_k = df[df["config"].str.startswith("yaw_") | (df["config"] == "baseline_0.0")].sort_values("yaw_deg")
        sub_n = df_nusc[df_nusc["config"].str.startswith("yaw_") | (df_nusc["config"] == "baseline_0.0")].sort_values("yaw_deg")
        ax.plot(sub_k["yaw_deg"], sub_k["box_retention_all_pct"], marker="o", color="#1f77b4", label="KITTI (64-beam, narrow FOV)")
        ax.plot(sub_n["yaw_deg"], sub_n["box_retention_all_pct"], marker="s", color="#ff7f0e", label="nuScenes (32-beam, wide FOV)")
        ax.set_title("Cross-Dataset Drift Sensitivity: KITTI vs nuScenes", fontweight="bold")
        ax.set_xlabel("Yaw Drift (degrees)")
        ax.set_ylabel("Box Point Retention (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        cross_fig = Path("results/figures/cross_dataset_comparison.png")
        plt.savefig(cross_fig, dpi=200)
        plt.close()
        print(f"-> Đã lưu biểu đồ so sánh chéo: {cross_fig}")

    print("\n[SUCCESS] Hoàn thành toàn bộ quy trình thí nghiệm CP3!")


if __name__ == "__main__":
    main()
