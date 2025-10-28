#!/usr/bin/env python3
"""
Single-sample visualisation for the MTF pipeline (normalized axis version).

- LR/SR/HR 各シリーズから 1 個の ROI を抽出
- 可視化は全て「正規化周波数 f/f_Nyquist（0..1）」で出力
- 本線(main_mtf.py)の処理・結果には一切影響しません（関数は既存を呼ぶだけ）

出力先: <out_dir>/sample_viz/<series_id>/<LR|SR|HR>/...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cv2  # type: ignore

# ==== ここを相対 import に変更（パッケージ実行前提） ====
from .dicom_io import load_series, adjust_pixel_spacing_for_sr, apply_spacing_override
from .roi_selector import ROISelector, ROI
from .utils import ensure_dir, seed_everything, setup_logging, stratified_sample
from .mtf_core import compute_mtf_for_roi
# ======================================================

YELLOW = (0, 255, 255)  # BGR (OpenCV)

def _to_normalized_freq(freq_cmm: np.ndarray, col_spacing_mm: float) -> np.ndarray:
    f_nyq = 1.0 / (2.0 * float(col_spacing_mm))
    if f_nyq <= 0:
        return np.zeros_like(freq_cmm)
    f_norm = freq_cmm / f_nyq
    return np.clip(f_norm, 0.0, 1.0)

def _poly_from_center_wh_angle(center_rc, w, h, angle_edge_deg):
    r0, c0 = float(center_rc[0]), float(center_rc[1])
    theta = np.deg2rad(angle_edge_deg)
    u = np.array([np.sin(theta), np.cos(theta)])   # 接線（dr, dc）
    v = np.array([-np.cos(theta), np.sin(theta)])  # 法線（dr, dc）
    dr_u = (h / 2.0) * u[0]; dc_u = (h / 2.0) * u[1]
    dr_v = (w / 2.0) * v[0]; dc_v = (w / 2.0) * v[1]
    corners_rc = np.array([
        [r0 - dr_u - dr_v, c0 - dc_u - dc_v],
        [r0 - dr_u + dr_v, c0 - dc_u + dc_v],
        [r0 + dr_u + dr_v, c0 + dc_u + dc_v],
        [r0 + dr_u - dr_v, c0 + dc_u - dc_v],
    ], dtype=np.float32)
    corners_xy = corners_rc[:, ::-1]
    return corners_xy.astype(np.int32)

def _draw_full_overlay(img_gray, center_rc, w, h, angle_edge_deg):
    vis = cv2.cvtColor(
        cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    poly = _poly_from_center_wh_angle(center_rc, w, h, angle_edge_deg)
    cv2.polylines(vis, [poly], isClosed=True, color=YELLOW, thickness=2)
    theta_n = np.deg2rad(angle_edge_deg - 90.0)
    c0, r0 = float(center_rc[1]), float(center_rc[0])
    length = int(max(w, h) * 3)
    x1, y1 = int(c0 - length * np.cos(theta_n)), int(r0 - length * np.sin(theta_n))
    x2, y2 = int(c0 + length * np.cos(theta_n)), int(r0 + length * np.sin(theta_n))
    cv2.line(vis, (x1, y1), (x2, y2), YELLOW, 3, lineType=cv2.LINE_AA)
    return vis

def _draw_context_overlay(img_gray, center_rc, w, h, angle_edge_deg):
    patch_side = int(np.ceil(np.hypot(w, h))) + 4
    margin = int(patch_side * 1.5)
    H, W = img_gray.shape
    r0, c0 = center_rc
    top  = max(0, r0 - margin); bot = min(H, r0 + margin)
    left = max(0, c0 - margin); right = min(W, c0 + margin)
    ctx_gray = img_gray[top:bot, left:right].copy()
    ctx_vis = cv2.cvtColor(
        cv2.normalize(ctx_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    poly = _poly_from_center_wh_angle(center_rc, w, h, angle_edge_deg)
    poly[:, 0] -= left; poly[:, 1] -= top
    cv2.polylines(ctx_vis, [poly], isClosed=True, color=YELLOW, thickness=2)
    theta_n = np.deg2rad(angle_edge_deg - 90.0)
    c_ctx, r_ctx = float(c0 - left), float(r0 - top)
    length = int(max(w, h) * 3)
    x1, y1 = int(c_ctx - length * np.cos(theta_n)), int(r_ctx - length * np.sin(theta_n))
    x2, y2 = int(c_ctx + length * np.cos(theta_n)), int(r_ctx + length * np.sin(theta_n))
    cv2.line(ctx_vis, (x1, y1), (x2, y2), YELLOW, 3, lineType=cv2.LINE_AA)
    return ctx_vis

def _gaussian1d(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return arr.astype(np.float64)
    radius = int(np.ceil(3 * sigma))
    xk = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(xk ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    return np.convolve(arr.astype(np.float64), k, mode="same")

def _save_plot(y, x, path, xlabel, ylabel, title=None):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

class _ROIWithCenter(ROI):
    center_row: int
    center_col: int

class ROISelectorWithCenter(ROISelector):
    def select_rois_with_centers(self, slices, series_name: str, num_rois: int) -> List[_ROIWithCenter]:
        if num_rois <= 0:
            return []
        candidates: List[_ROIWithCenter] = []
        n_slices = len(slices)
        if n_slices == 0:
            return []
        slice_bin_edges = [0.0, 1.0/3.0, 2.0/3.0, 1.0+1e-6]
        max_candidates = num_rois * 10
        rng = np.random.default_rng()

        for sl in slices:
            if len(candidates) >= max_candidates:
                break
            img = sl.pixel_array
            p2, p98 = np.percentile(img, [2, 98])
            if p98 - p2 < 1e-3:
                img_8 = np.zeros_like(img, dtype=np.uint8)
            else:
                img_8 = np.clip(((img - p2) / (p98 - p2)) * 255.0, 0, 255).astype(np.uint8)

            edges = cv2.Canny(img_8, 50, 150, apertureSize=3)
            sobelx = cv2.Sobel(img_8, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_8, cv2.CV_64F, 0, 1, ksize=3)
            _, grad_angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

            coords = np.column_stack(np.where(edges > 0))
            rng.shuffle(coords)

            for (r, c) in coords:
                angle_edge = (float(grad_angle[r, c]) + 90.0) % 180.0
                diff_h = min(abs(angle_edge - 0.0), abs(angle_edge - 180.0))
                diff_v = abs(angle_edge - 90.0)
                min_diff = min(diff_h, diff_v)
                if min_diff < self.angle_min or min_diff > self.angle_max:
                    continue

                patch_side = int(np.ceil(np.hypot(self.roi_width, self.roi_height))) + 4
                half = patch_side // 2
                H, W = img.shape
                if not (half <= c < W - half and half <= r < H - half):
                    continue
                patch = cv2.getRectSubPix(img, (patch_side, patch_side), (float(c), float(r)))
                rot_angle = 90.0 - angle_edge
                M = cv2.getRotationMatrix2D((patch_side/2.0, patch_side/2.0), rot_angle, 1.0)
                rotated = cv2.warpAffine(patch, M, (patch_side, patch_side),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                roi_img = cv2.getRectSubPix(rotated, (self.roi_width, self.roi_height),
                                            (patch_side/2.0, patch_side/2.0))
                if roi_img is None:
                    continue

                third = self.roi_width // 3
                if third < 1:
                    continue
                left = roi_img[:, :third]
                right = roi_img[:, -third:]
                delta_hu = abs(float(np.mean(right)) - float(np.mean(left)))
                if delta_hu < self.delta_hu_threshold:
                    continue

                relative_pos = sl.index / max(1, n_slices - 1)
                axial_bin = 0
                for idx, (lo, hi) in enumerate(zip(slice_bin_edges[:-1], slice_bin_edges[1:])):
                    if lo <= relative_pos < hi:
                        axial_bin = idx; break
                orientation_category = 0 if diff_h <= diff_v else 1
                group = axial_bin * 2 + orientation_category

                roi = _ROIWithCenter(
                    series_name=series_name,
                    slice_index=sl.index,
                    slice_filename=sl.filename,
                    roi_image=roi_img.astype(np.float32),
                    orientation_deg=angle_edge,
                    delta_hu=delta_hu,
                    group=group,
                )
                roi.center_row = int(r)
                roi.center_col = int(c)
                candidates.append(roi)
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

        if not candidates:
            logging.warning(f"No ROI candidates for {series_name}")
            return []

        indices = list(range(len(candidates)))
        groups = [roi.group for roi in candidates]
        selected = stratified_sample(indices, groups, num_rois)
        if len(selected) < num_rois:
            remain = list(set(indices) - set(selected))
            if len(remain) > 0:
                extra = rng.choice(remain, size=min(num_rois - len(selected), len(remain)), replace=False)
                selected.extend(extra.tolist())
        selected = selected[:num_rois]
        return [candidates[i] for i in selected]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lr_dir", type=str, required=True)
    p.add_argument("--sr_dir", type=str, required=True)
    p.add_argument("--hr_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--series_id", type=str, default="sample")
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_cuda", type=int, choices=[0,1], default=1)
    p.add_argument("--lr_spacing", type=float, nargs=2, metavar=("ROW","COL"))
    p.add_argument("--hr_spacing", type=float, nargs=2, metavar=("ROW","COL"))
    p.add_argument("--sr_scale", type=float, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)

    out_root = Path(args.out_dir) / "sample_viz" / args.series_id
    ensure_dir(out_root)

    series_dirs = {"LR": Path(args.lr_dir), "SR": Path(args.sr_dir), "HR": Path(args.hr_dir)}
    series_slices: Dict[str, List] = {}
    for name, d in series_dirs.items():
        logging.info(f"[viz] Loading series {name} from {d}")
        series_slices[name] = load_series(d)

    if getattr(args, "lr_spacing", None):
        apply_spacing_override(series_slices.get("LR", []), args.lr_spacing)
        apply_spacing_override(series_slices.get("SR", []), args.lr_spacing)
    if getattr(args, "hr_spacing", None):
        apply_spacing_override(series_slices.get("HR", []), args.hr_spacing)
    if args.sr_scale is not None and args.sr_scale > 0.0:
        adjust_pixel_spacing_for_sr(series_slices.get("SR", []), args.sr_scale)

    slices_lookup: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for name, slices in series_slices.items():
        tbl: Dict[int, Tuple[float, float]] = {}
        for s in slices:
            if not getattr(s, "pixel_spacing", None):
                s.pixel_spacing = (1.0, 1.0)
            row_sp, col_sp = float(s.pixel_spacing[0]), float(s.pixel_spacing[1])
            tbl[s.index] = (row_sp, col_sp)
        slices_lookup[name] = tbl

    selector = ROISelectorWithCenter(
        roi_width=30, roi_height=100,
        angle_min=5.0, angle_max=15.0, delta_hu_threshold=200.0
    )

    chosen: Dict[str, _ROIWithCenter] = {}
    for name, slices in series_slices.items():
        rois = selector.select_rois_with_centers(slices, name, num_rois=max(1, args.sample_index+1))
        if not rois:
            logging.warning(f"[viz] No ROI found in {name}")
            continue
        idx = min(args.sample_index, len(rois)-1)
        chosen[name] = rois[idx]
        logging.info(f"[viz] {name}: using ROI idx={idx} on slice={rois[idx].slice_index}")

    triplet = []
    for name in ["LR", "SR", "HR"]:
        if name not in chosen:
            continue
        r = chosen[name]
        sl = series_slices[name][r.slice_index]
        series_out = out_root / name
        ensure_dir(series_out)

        full_overlay = _draw_full_overlay(sl.pixel_array, (r.center_row, r.center_col),
                                          selector.roi_width, selector.roi_height, r.orientation_deg)
        import cv2
        cv2.imwrite(str(series_out / "roi_overlay.png"), full_overlay)

        ctx = _draw_context_overlay(sl.pixel_array, (r.center_row, r.center_col),
                                    selector.roi_width, selector.roi_height, r.orientation_deg)
        cv2.imwrite(str(series_out / "context_overlay.png"), ctx)

        crop = r.roi_image
        crop_vis = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(series_out / "roi_crop.png"), crop_vis)

        esf_raw = np.mean(crop, axis=0).astype(np.float64)
        x0 = np.arange(len(esf_raw))
        esf_smooth = _gaussian1d(esf_raw, sigma=1.0)
        lsf = np.gradient(esf_smooth)
        _save_plot(esf_raw, x0, series_out / "esf_raw.png", "Pixel", "ESF", f"ESF Raw ({name})")
        _save_plot(esf_smooth, x0, series_out / "esf_smooth.png", "Pixel", "ESF", f"ESF Smooth ({name})")
        _save_plot(lsf, x0, series_out / "lsf.png", "Pixel", "LSF", f"LSF ({name})")

        row_sp, col_sp = slices_lookup[name][r.slice_index]
        freq, mtf, mtf50, mtf10 = compute_mtf_for_roi(
            crop, pixel_spacing=(row_sp, col_sp),
            oversample_factor=4, zero_pad_factor=4, use_cuda=bool(args.use_cuda)
        )
        f_norm = _to_normalized_freq(freq, col_sp)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(f_norm, mtf)
        ax.set_xlabel("Normalized Spatial Frequency (f / f_Nyquist)")
        ax.set_ylabel("MTF")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.2)
        fig.tight_layout()
        fig.savefig(series_out / "mtf_single.png", dpi=180)
        plt.close(fig)

        triplet.append((name, freq, mtf, r.slice_index))

    if triplet:
        fig, ax = plt.subplots(figsize=(7,6))
        for name, f_cmm, m, sl_idx in triplet:
            _, col_sp_ = slices_lookup[name][sl_idx]
            f_norm = _to_normalized_freq(f_cmm, col_sp_)
            ls = "-" if name == "LR" else "--" if name == "SR" else ":"
            ax.plot(f_norm, m, linestyle=ls, linewidth=2.0, label=name)
        ax.set_xlabel("Normalized Spatial Frequency (f / f_Nyquist)")
        ax.set_ylabel("MTF")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_root / "mtf_triplet_overlaid.png", dpi=200)
        plt.close(fig)

    logging.info(f"[viz] Saved visualisations under: {out_root}")

if __name__ == "__main__":
    main()
