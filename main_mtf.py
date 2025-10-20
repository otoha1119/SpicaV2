#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 既存プロジェクト構成に合わせた import（パッケージ名は環境に応じて調整）
from MTF.dicom_io import load_series, adjust_pixel_spacing_for_sr, apply_spacing_override
from MTF.roi_selector import ROISelector, ROI
from MTF.mtf_core import compute_mtf_for_roi, compute_auc
from MTF.plotting import plot_mean_mtf
from MTF.utils import ensure_dir, seed_everything, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MTF curves from CT series.")
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--sr_dir", type=str, required=True)
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--num_rois", type=int, default=150)
    parser.add_argument("--draw_nyquist", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_cuda", type=int, choices=[0, 1], default=1)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--sr_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x_norm", type=int, default=1)

    parser.add_argument("--lr_spacing", type=float, nargs=2, metavar=("ROW", "COL"))
    parser.add_argument("--hr_spacing", type=float, nargs=2, metavar=("ROW", "COL"))

    # 例画像出力
    parser.add_argument("--export_examples", type=int, choices=[0, 1], default=0)
    parser.add_argument("--examples_per_series", type=int, default=1)

    # --- ROI geometry & angle gates（新規） ---
    parser.add_argument("--roi_width", type=int, default=40)
    parser.add_argument("--roi_height", type=int, default=100)
    parser.add_argument("--angle_min", type=float, default=6.0)
    parser.add_argument("--angle_max", type=float, default=12.0)

    # --- 一本＆上下貫通（新規） ---
    parser.add_argument("--require_single_edge", type=int, choices=[0, 1], default=1)
    parser.add_argument("--edge_vertical_tol_deg", type=float, default=2.5)
    parser.add_argument("--min_vertical_span_ratio", type=float, default=0.92)
    parser.add_argument("--center_tolerance_ratio", type=float, default=0.08)

    # --- ESF 安定化（新規） ---
    parser.add_argument("--esf_frac", type=float, default=0.6)
    parser.add_argument("--esf_reduce", type=str, choices=["mean", "median", "trimmed"], default="median")
    parser.add_argument("--trim_alpha", type=float, default=0.1)

    return parser.parse_args()


def compute_metrics_for_rois(
    rois: List[ROI],
    slices_lookup: Dict[int, Tuple[float, float]],
    use_cuda: bool,
    global_f_max: float,
    row_fraction: float,
    reduce_mode: str,
    trim_alpha: float,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float], pd.DataFrame]:
    """
    各 ROI の MTF を計算し、(freq, mtf)曲線 / Nyquist / 指標の表を返す。
    """
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    nyquist_list: List[float] = []
    records = []
    for idx, roi in enumerate(rois):
        row_spacing, col_spacing = slices_lookup.get(roi.slice_index, (1.0, 1.0))
        try:
            freq, mtf, mtf50, mtf10 = compute_mtf_for_roi(
                roi.roi_image,
                pixel_spacing=(row_spacing, col_spacing),
                oversample_factor=4,
                zero_pad_factor=4,
                use_cuda=use_cuda,
                row_fraction=row_fraction,
                reduce_mode=reduce_mode,
                trim_alpha=trim_alpha,
            )
        except Exception as e:
            logging.warning(f"Failed to compute MTF for ROI idx={idx} slice={roi.slice_index}: {e}")
            continue

        nyquist = 0.5 / col_spacing if col_spacing != 0 else 0.0
        nyquist_list.append(nyquist)
        auc = compute_auc(freq, mtf, min(global_f_max, nyquist))
        curves.append((freq, mtf))
        records.append(
            {
                "series": roi.series_name,
                "slice_index": roi.slice_index,
                "roi_idx": idx,
                "orientation_deg": roi.orientation_deg,
                "delta_hu": roi.delta_hu,
                "mtf50": mtf50,
                "mtf10": mtf10,
                "auc": auc,
                "pixel_spacing_mm": col_spacing,
                "nyquist": nyquist,
            }
        )
    df = pd.DataFrame.from_records(records)
    return curves, nyquist_list, df


# --- 例画像 (ROI→ESF→LSF→MTF) 出力 ---
def _export_roi_esf_lsf_mtf(
    roi: ROI,
    pixel_spacing: Tuple[float, float],
    out_dir: Path,
    series_name: str,
    idx_in_series: int,
    draw_nyquist: bool,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    img = np.asarray(roi.roi_image, dtype=float)
    roi_dir = out_dir / "examples" / series_name
    roi_dir.mkdir(parents=True, exist_ok=True)

    # ROI 画像（法線方向をオレンジの直線で端から端まで）
    roi_png = roi_dir / f"roi_{idx_in_series:03d}.png"
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    h, w = img.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    theta = np.deg2rad(roi.orientation_deg + 90.0)  # edge normal
    dx, dy = np.cos(theta), np.sin(theta)
    # 端点（直線）
    t1x, t1y = cx - dx * (w * 0.5), cy - dy * (h * 0.5)
    t2x, t2y = cx + dx * (w * 0.5), cy + dy * (h * 0.5)
    ax.plot([t1x, t2x], [t1y, t2y], '-', color='orange', linewidth=2.0)
    fig.tight_layout(pad=0)
    fig.savefig(roi_png, dpi=200)
    plt.close(fig)

    # ESF（法線方向の投影でビン平均）
    y_idx, x_idx = np.indices(img.shape)
    x0 = x_idx - cx
    y0 = y_idx - cy
    t = x0 * dx + y0 * dy  # projection to normal [px]
    row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    px_mm = (row_mm + col_mm) * 0.5
    dt_px = 0.25
    t_edges = np.linspace(np.min(t), np.max(t), int(np.ceil((np.max(t) - np.min(t)) / dt_px)) + 2)
    t_centers_px = 0.5 * (t_edges[:-1] + t_edges[1:])
    bin_idx = np.clip(np.searchsorted(t_edges, t, side="right") - 1, 0, len(t_centers_px) - 1)
    sums = np.bincount(bin_idx.ravel(), weights=img.ravel(), minlength=len(t_centers_px))
    counts = np.bincount(bin_idx.ravel(), minlength=len(t_centers_px))
    esf = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
    if np.isfinite(esf).any():
        k = 5
        kernel = np.ones(k, dtype=float) / k
        esf = np.convolve(np.nan_to_num(esf, nan=np.nanmean(esf)), kernel, mode="same")
    t_centers_mm = t_centers_px * px_mm

    esf_png = roi_dir / f"esf_{idx_in_series:03d}.png"
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(t_centers_mm, esf, color="black", linewidth=1.8)
    ax.set_xlabel("Distance along edge normal [mm]")
    ax.set_ylabel("ESF (arb. units)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(esf_png, dpi=200)
    plt.close(fig)

    # LSF
    dt_mm = float(np.median(np.diff(t_centers_mm))) if len(t_centers_mm) >= 3 else px_mm * dt_px
    lsf = np.gradient(esf, dt_mm)
    lsf_png = roi_dir / f"lsf_{idx_in_series:03d}.png"
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(t_centers_mm, lsf, color="black", linewidth=1.8)
    ax.set_xlabel("Distance along edge normal [mm]")
    ax.set_ylabel("LSF (derivative of ESF)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(lsf_png, dpi=200)
    plt.close(fig)

    # MTF
    lsf = np.nan_to_num(lsf, nan=0.0)
    spec = np.fft.rfft(lsf)
    freq = np.fft.rfftfreq(lsf.size, d=dt_mm)
    mtf = np.abs(spec)
    if mtf.size > 0 and mtf[0] != 0:
        mtf = mtf / mtf[0]
    nyq = 0.5 / col_mm if col_mm > 0 else None

    mtf_png = roi_dir / f"mtf_{idx_in_series:03d}.png"
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(freq, mtf, color="black", linewidth=1.8)
    if draw_nyquist and nyq is not None:
        ax.axvline(nyq, color="0.6", linestyle="--", linewidth=1.0)
    ax.set_xlim(0, max(1e-6, np.nanmax(freq)))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Spatial Frequency [cycles/mm]")
    ax.set_ylabel("MTF")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(mtf_png, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load
    series_dirs = {"LR": Path(args.lr_dir), "SR": Path(args.sr_dir), "HR": Path(args.hr_dir)}
    series_slices: Dict[str, List] = {}
    for series_name, d in series_dirs.items():
        logging.info(f"Loading series {series_name} from {d}")
        series_slices[series_name] = load_series(d)
        if not series_slices[series_name]:
            logging.warning(f"No DICOM slices found in {d}")

    # SR spacing補正 & 明示上書き
    if args.lr_spacing:
        apply_spacing_override(series_slices.get("LR", []), args.lr_spacing)
        apply_spacing_override(series_slices.get("SR", []), args.lr_spacing)
    if args.hr_spacing:
        apply_spacing_override(series_slices.get("HR", []), args.hr_spacing)
    if args.sr_scale and args.sr_scale > 0:
        adjust_pixel_spacing_for_sr(series_slices.get("SR", []), args.sr_scale)

    # lookup map
    slices_lookup_map: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for series_name, slices in series_slices.items():
        lookup = {}
        for s in slices:
            lookup[s.index] = s.pixel_spacing
        slices_lookup_map[series_name] = lookup

    # ROI selection（新既定：厳格な一本＆上下貫通チェック）
    selector = ROISelector(
        roi_width=args.roi_width,
        roi_height=args.roi_height,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        delta_hu_threshold=200.0,
        edge_vertical_tol_deg=args.edge_vertical_tol_deg,
        min_vertical_span_ratio=args.min_vertical_span_ratio,
        require_single_edge=bool(args.require_single_edge),
        center_tolerance_ratio=args.center_tolerance_ratio,
    )
    series_rois: Dict[str, List[ROI]] = {}
    for name, slices in series_slices.items():
        logging.info(f"Selecting ROIs for series {name}")
        rois = selector.select_rois(slices, name, args.num_rois)
        series_rois[name] = rois
        logging.info(f"Selected {len(rois)} ROIs for {name}")

    # 例保存
    if args.export_examples == 1:
        ex_n = int(args.examples_per_series)
        for series_name, rois in series_rois.items():
            if not rois:
                continue
            lookup = slices_lookup_map.get(series_name, {})
            for j, roi in enumerate(rois[:ex_n]):
                pixsp = lookup.get(roi.slice_index, (1.0, 1.0))
                _export_roi_esf_lsf_mtf(
                    roi=roi,
                    pixel_spacing=pixsp,
                    out_dir=out_dir,
                    series_name=series_name,
                    idx_in_series=j,
                    draw_nyquist=bool(args.draw_nyquist),
                )
        logging.info(f"Saved example ROI/ESF/LSF/MTF under {out_dir / 'examples'}")

    # f_max
    all_nyquists: List[float] = []
    for name, slices in series_slices.items():
        for s in slices:
            col_spacing = s.pixel_spacing[1]
            if col_spacing > 0:
                all_nyquists.append(0.5 / col_spacing)
    global_f_max = (min(all_nyquists) * 0.95) if all_nyquists else 0.0
    logging.info(f"Global f_max for AUC: {global_f_max:.3f} cycles/mm")

    # 計算
    series_curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    series_nyquists: Dict[str, List[float]] = {}
    metrics_frames: List[pd.DataFrame] = []
    for name, rois in series_rois.items():
        if not rois:
            continue
        curves, nyquists, df_metrics = compute_metrics_for_rois(
            rois,
            slices_lookup_map[name],
            use_cuda=bool(args.use_cuda),
            global_f_max=global_f_max,
            row_fraction=float(args.esf_frac),
            reduce_mode=str(args.esf_reduce),
            trim_alpha=float(args.trim_alpha),
        )
        series_curves[name] = curves
        series_nyquists[name] = nyquists
        metrics_frames.append(df_metrics)
        (out_dir / f"{name.lower()}_roi_metrics.csv").write_text(df_metrics.to_csv(index=False))

    # プロット保存
    fig_path = out_dir / ("mtf_mean_norm.png" if args.x_norm == 1 else "mtf_mean_phys.png")
    plot_mean_mtf(series_curves, series_nyquists, str(fig_path), draw_nyquist=bool(args.draw_nyquist), normalize_x=bool(args.x_norm))
    logging.info(f"Saved mean MTF figure to: {fig_path}")


if __name__ == "__main__":
    main()
