#!/usr/bin/env python3
"""
Entry point for the MTF extraction CLI.

This script orchestrates loading DICOM series, selecting natural
high-contrast ROIs, computing MTF curves for each ROI, aggregating
statistics, and generating summary plots.  It exposes a command
line interface to specify directories for LR, SR and HR series,
control the number of ROIs, toggle CUDA usage, and adjust SR
scaling. Optionally exports per-ROI demonstration figures
(ROI image, ESF, LSF, MTF) for presentation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from MTF.dicom_io import load_series
from MTF.roi_selector import ROISelector, ROI
from MTF.mtf_core import compute_mtf_for_roi, compute_auc
from MTF.plotting import plot_mean_mtf
from MTF.utils import ensure_dir, seed_everything, setup_logging
from MTF.dicom_io import adjust_pixel_spacing_for_sr, apply_spacing_override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MTF curves from CT series.")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory with LR DICOM files")
    parser.add_argument("--sr_dir", type=str, required=True, help="Directory with SR DICOM files")
    parser.add_argument("--hr_dir", type=str, required=True, help="Directory with HR DICOM files")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num_rois", type=int, default=150, help="Number of ROIs per series")
    parser.add_argument("--draw_nyquist", type=int, choices=[0, 1], default=1, help="Draw Nyquist lines in plot")
    parser.add_argument("--use_cuda", type=int, choices=[0, 1], default=1, help="Use CUDA for FFT if available")
    parser.add_argument("--gpu_ids", type=str, default="0", help="CUDA device IDs (ignored, set via wrapper)")
    parser.add_argument("--sr_scale", type=float, default=None, help="Scale factor to adjust SR PixelSpacing (optional)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--x_norm", type=int, default=1, help="1: normalized x (f/fNyquist), 0: physical cycles/mm")
    parser.add_argument(
        "--lr_spacing",
        type=float,
        nargs=2,
        metavar=("ROW", "COL"),
        help="Override LR PixelSpacing (mm), e.g., --lr_spacing 0.3184 0.3184",
    )
    parser.add_argument(
        "--hr_spacing",
        type=float,
        nargs=2,
        metavar=("ROW", "COL"),
        help="Override HR PixelSpacing (mm), e.g., --hr_spacing 0.136719 0.136719",
    )
    # ROI configuration
    parser.add_argument(
        "--roi_width",
        type=int,
        default=40,
        help="Width of the rotated ROI in pixels (default: 40). A more square ROI reduces averaging over unrelated regions.",
    )
    parser.add_argument(
        "--roi_height",
        type=int,
        default=40,
        help="Height of the rotated ROI in pixels (default: 40). Smaller values avoid elongated ROIs that may not contain the edge across the full height.",
    )
    parser.add_argument(
        "--esf_frac",
        type=float,
        default=0.5,
        help="Fraction (0–1) of the ROI height used to compute the ESF. Only the central portion of the ROI will be used. Default is 0.5 (central 50%).",
    )
    # --- NEW: export examples (ROI/ESF/LSF/MTF) ---
    parser.add_argument("--export_examples", type=int, choices=[0, 1], default=0,
                        help="Save example ROI/ESF/LSF/MTF PNGs (0=off, 1=on)")
    parser.add_argument("--examples_per_series", type=int, default=1,
                        help="How many example ROIs to export per series when --export_examples=1")

    return parser.parse_args()


def compute_metrics_for_rois(
    rois: List[ROI],
    slices_lookup: Dict[int, Tuple[float, float]],
    use_cuda: bool,
    global_f_max: float,
    row_fraction: float = 0.5,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float], pd.DataFrame]:
    """
    Compute MTF metrics for each ROI in a series.

    The function iterates over the supplied ROIs, looks up the
    corresponding pixel spacing for each ROI's slice, and computes
    the MTF curve using :func:`compute_mtf_for_roi`.  The Nyquist
    frequency for each ROI is also computed from the in‐plane pixel
    spacing.  A pandas DataFrame is returned with per‐ROI metrics.

    Parameters
    ----------
    rois : list of ROI
        The ROIs to process.
    slices_lookup : dict
        Mapping from slice index to pixel spacing (row_spacing, col_spacing).
    use_cuda : bool
        Whether to attempt CUDA for FFTs (falls back to CPU if unavailable).
    global_f_max : float
        Global maximum frequency used for AUC computation.
    row_fraction : float, optional
        Fraction of the ROI height to use when computing the ESF.  This
        value is passed through to :func:`compute_mtf_for_roi`.

    Returns
    -------
    curves : list of (freq, mtf) tuples
        The MTF curves for each ROI.
    nyquists : list of float
        Nyquist frequencies for each ROI.
    metrics_df : pandas.DataFrame
        A table of ROI metrics including MTF50, MTF10 and AUC.
    """
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    nyquist_list: List[float] = []
    records = []
    for idx, roi in enumerate(rois):
        # Lookup pixel spacing for this ROI's slice index
        row_spacing, col_spacing = slices_lookup.get(roi.slice_index, (1.0, 1.0))
        try:
            freq, mtf, mtf50, mtf10 = compute_mtf_for_roi(
                roi.roi_image,
                pixel_spacing=(row_spacing, col_spacing),
                oversample_factor=4,
                zero_pad_factor=4,
                use_cuda=use_cuda,
                row_fraction=row_fraction,
            )
        except Exception as e:
            logging.warning(f"Failed to compute MTF for ROI idx={idx} slice={roi.slice_index}: {e}")
            continue
        # Nyquist frequency based on pixel spacing (col_spacing)
        nyquist = 0.5 / col_spacing if col_spacing != 0 else 0.0
        nyquist_list.append(nyquist)
        # Compute AUC up to global_f_max
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


# --- NEW: single-ROI export (ROI -> ESF -> LSF -> MTF) ---
def _export_roi_esf_lsf_mtf(
    roi: ROI,
    pixel_spacing: Tuple[float, float],
    out_dir: Path,
    series_name: str,
    idx_in_series: int,
    draw_nyquist: bool,
) -> None:
    """
    Make & save 4 PNGs per ROI: roi_XXX.png, esf_XXX.png, lsf_XXX.png, mtf_XXX.png
    Dependencies: numpy, matplotlib only.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    img = np.asarray(roi.roi_image, dtype=float)

    # --- ROI image with normal indicator ---
    roi_dir = out_dir / "examples" / series_name
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_png = roi_dir / f"roi_{idx_in_series:03d}.png"

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    h, w = img.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    theta = np.deg2rad(roi.orientation_deg + 90.0)  # edge normal
    dx, dy = np.cos(theta), np.sin(theta)
    # Draw a solid line along the normal direction through the ROI centre.  The line
    # spans the full width of the ROI to clearly indicate the sampling direction.
    # A slightly shorter length is used to avoid drawing outside the axes.
    half_len = (w - 1) / 2.0
    x_start = cx - dx * half_len
    y_start = cy - dy * half_len
    x_end = cx + dx * half_len
    y_end = cy + dy * half_len
    ax.plot([x_start, x_end], [y_start, y_end], color="orange", linewidth=1.5)
    fig.tight_layout(pad=0)
    fig.savefig(roi_png, dpi=200)
    plt.close(fig)

    # --- ESF along the normal direction ---
    y_idx, x_idx = np.indices(img.shape)
    x0 = x_idx - cx
    y0 = y_idx - cy
    t = x0 * dx + y0 * dy  # projection to normal [px]

    row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    px_mm = (row_mm + col_mm) * 0.5  # approximate
    dt_px = 0.25  # oversampling-like bin
    t_min, t_max = float(np.min(t)), float(np.max(t))
    nbins = int(np.ceil((t_max - t_min) / dt_px)) + 1
    t_edges = np.linspace(t_min, t_max, nbins + 1)
    t_centers_px = 0.5 * (t_edges[:-1] + t_edges[1:])
    bin_idx = np.clip(np.searchsorted(t_edges, t, side="right") - 1, 0, nbins - 1)
    sums = np.bincount(bin_idx.ravel(), weights=img.ravel(), minlength=nbins)
    counts = np.bincount(bin_idx.ravel(), minlength=nbins)
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

    # --- LSF = d/dx ESF ---
    if len(t_centers_mm) >= 3:
        dt_mm = float(np.median(np.diff(t_centers_mm)))
    else:
        dt_mm = px_mm * dt_px
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

    # --- MTF = |rFFT(LSF)| normalized by DC ---
    lsf = np.nan_to_num(lsf, nan=0.0)
    spec = np.fft.rfft(lsf)
    freq = np.fft.rfftfreq(lsf.size, d=dt_mm)  # cycles/mm
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

    # Load all series first
    series_dirs = {"LR": Path(args.lr_dir), "SR": Path(args.sr_dir), "HR": Path(args.hr_dir)}
    series_slices: Dict[str, List] = {}
    for series_name, d in series_dirs.items():
        logging.info(f"Loading series {series_name} from {d}")
        series_slices[series_name] = load_series(d)
        if not series_slices[series_name]:
            logging.warning(f"No DICOM slices found in {d}")

    # 1) copy DICOM PixelSpacing -> s.pixel_spacing when present
    for series_name, slices in series_slices.items():
        for s in slices:
            if hasattr(s, "PixelSpacing") and s.PixelSpacing is not None:
                try:
                    s.pixel_spacing = (float(s.PixelSpacing[0]), float(s.PixelSpacing[1]))
                except Exception:
                    pass  # let overrides handle

    # 2) explicit overrides (LR/HR), and seed SR from LR if needed
    if getattr(args, "lr_spacing", None):
        apply_spacing_override(series_slices.get("LR", []), args.lr_spacing)
    if getattr(args, "hr_spacing", None):
        apply_spacing_override(series_slices.get("HR", []), args.hr_spacing)
    if getattr(args, "lr_spacing", None):
        apply_spacing_override(series_slices.get("SR", []), args.lr_spacing)

    # 3) SR scale (divide in-plane spacing by scale)
    if args.sr_scale is not None and args.sr_scale > 0.0:
        logging.info(f"Applying SR scale factor {args.sr_scale} to pixel spacing")
        adjust_pixel_spacing_for_sr(series_slices.get("SR", []), args.sr_scale)

    # 3.5) final sync / fallback
    for series_name, slices in series_slices.items():
        for s in slices:
            src = getattr(s, "PixelSpacing", None)
            if src is not None:
                try:
                    s.pixel_spacing = (float(src[0]), float(src[1]))
                    continue
                except Exception:
                    pass
            ps = getattr(s, "pixel_spacing", None)
            if (ps is None) or (len(ps) == 2 and (ps[0] == 1.0 and ps[1] == 1.0)):
                if series_name == "LR" and getattr(args, "lr_spacing", None):
                    s.pixel_spacing = (float(args.lr_spacing[0]), float(args.lr_spacing[1]))
                elif series_name == "HR" and getattr(args, "hr_spacing", None):
                    s.pixel_spacing = (float(args.hr_spacing[0]), float(args.hr_spacing[1]))
                elif series_name == "SR":
                    if getattr(args, "lr_spacing", None) and args.sr_scale:
                        s.pixel_spacing = (
                            float(args.lr_spacing[0]) / float(args.sr_scale),
                            float(args.lr_spacing[1]) / float(args.sr_scale),
                        )

    # 4) build per-slice spacing lookup (used everywhere below)
    slices_lookup_map: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for series_name, slices in series_slices.items():
        lookup = {}
        for s in slices:
            lookup[s.index] = s.pixel_spacing
        slices_lookup_map[series_name] = lookup

    # quick sanity log
    def _med_col(sp_list):
        cols = [float(sp[1]) for sp in sp_list if sp and sp[1] > 0]
        return (np.median(cols) if cols else float("nan"))
    lr_med = _med_col([s.pixel_spacing for s in series_slices.get("LR", [])])
    sr_med = _med_col([s.pixel_spacing for s in series_slices.get("SR", [])])
    hr_med = _med_col([s.pixel_spacing for s in series_slices.get("HR", [])])
    logging.info(f"[Check] median PixelSpacing col (mm): LR={lr_med:.6f}, SR={sr_med:.6f}, HR={hr_med:.6f}")

    # ROI selection
    # Instantiate ROISelector using user‑specified dimensions.  A more
    # square ROI (width≈height) mitigates the issue of edges that do
    # not traverse the full height of very tall patches and therefore
    # improves the stability of the ESF.
    selector = ROISelector(
        roi_width=int(getattr(args, "roi_width", 40)),
        roi_height=int(getattr(args, "roi_height", 40)),
        angle_min=5.0,
        angle_max=15.0,
        delta_hu_threshold=200.0,
    )
    series_rois: Dict[str, List[ROI]] = {}
    for name, slices in series_slices.items():
        logging.info(f"Selecting ROIs for series {name}")
        rois = selector.select_rois(slices, name, args.num_rois)
        series_rois[name] = rois
        logging.info(f"Selected {len(rois)} ROIs for {name}")

    # --- NEW: optional export of example ROI/ESF/LSF/MTF before heavy metrics ---
    if getattr(args, "export_examples", 0) == 1:
        ex_n = int(getattr(args, "examples_per_series", 1))
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

    # Determine global f_max (min of all Nyquist frequencies across series)
    all_nyquists: List[float] = []
    for name, slices in series_slices.items():
        for s in slices:
            col_spacing = s.pixel_spacing[1]
            if col_spacing > 0:
                all_nyquists.append(0.5 / col_spacing)
    global_f_max = min(all_nyquists) * 0.95 if all_nyquists else 0.0
    if global_f_max <= 0.0:
        global_f_max = 0.0
    logging.info(f"Global f_max for AUC computation: {global_f_max:.3f} cycles/mm")

    # Compute metrics
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
            row_fraction=float(getattr(args, "esf_frac", 0.5)),
        )
        series_curves[name] = curves
        series_nyquists[name] = nyquists
        metrics_frames.append(df_metrics)
        # Save ROI metrics
        roi_csv = out_dir / f"{name.lower()}_roi_metrics.csv"
        df_metrics.to_csv(roi_csv, index=False)
        logging.info(f"Saved ROI metrics for {name} to {roi_csv}")

    # Save summary metrics
    if metrics_frames:
        summary_records = []
        for df in metrics_frames:
            series_name = df['series'].iloc[0]
            summary_records.append(
                {
                    'series': series_name,
                    'n_rois': len(df),
                    'mean_mtf50': df['mtf50'].mean(),
                    'sd_mtf50': df['mtf50'].std(ddof=0) if len(df) > 1 else 0.0,
                    'mean_mtf10': df['mtf10'].mean(),
                    'sd_mtf10': df['mtf10'].std(ddof=0) if len(df) > 1 else 0.0,
                    'mean_auc': df['auc'].mean(),
                    'sd_auc': df['auc'].std(ddof=0) if len(df) > 1 else 0.0,
                    'mean_pixel_spacing_mm': df['pixel_spacing_mm'].mean(),
                    'sd_pixel_spacing_mm': df['pixel_spacing_mm'].std(ddof=0) if len(df) > 1 else 0.0,
                    'mean_nyquist': df['nyquist'].mean(),
                    'sd_nyquist': df['nyquist'].std(ddof=0) if len(df) > 1 else 0.0,
                }
            )
        summary_df = pd.DataFrame(summary_records)
        summary_path = out_dir / "summary_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Saved summary metrics to {summary_path}")

    # Plot
    plot_path = out_dir / "mtf_overlaid.png"
    plot_mean_mtf(
        series_mtf=series_curves,
        series_nyquist=series_nyquists,
        out_path=str(plot_path),
        draw_nyquist=bool(args.draw_nyquist),
        normalize_x=bool(args.x_norm),
    )
    logging.info(f"Saved MTF plot to {plot_path}")


if __name__ == "__main__":
    main()
