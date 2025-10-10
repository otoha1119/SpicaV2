#!/usr/bin/env python3
"""
Entry point for the MTF extraction CLI.

This script orchestrates loading DICOM series, selecting natural
high‑contrast ROIs, computing MTF curves for each ROI, aggregating
statistics, and generating summary plots.  It exposes a command
line interface to specify directories for LR, SR and HR series,
control the number of ROIs, toggle CUDA usage, and adjust SR
scaling.
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
    return parser.parse_args()


def adjust_pixel_spacing_for_sr(slices: List, scale: float) -> None:
    """Apply scaling factor to SR slice pixel spacing in place."""
    if scale is None:
        return
    for s in slices:
        # Pixel spacing is tuple (row_spacing, col_spacing)
        row_spacing, col_spacing = s.pixel_spacing
        # We assume scaling is uniform; only adjust along in‑plane dimensions
        s.pixel_spacing = (row_spacing / scale, col_spacing / scale)


def compute_metrics_for_rois(
    rois: List[ROI],
    slices_lookup: Dict[int, Tuple[float, float]],
    use_cuda: bool,
    global_f_max: float,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float], pd.DataFrame]:
    """
    Compute MTF metrics for each ROI in a series.

    Parameters
    ----------
    rois : list of ROI
        ROIs selected for the series.
    slices_lookup : dict
        Mapping from slice index to pixel spacing (row_spacing, col_spacing) in mm.
    use_cuda : bool
        Whether to use CUDA FFT where available.
    global_f_max : float
        Maximum frequency for AUC computation (shared across all series).

    Returns
    -------
    curves : list of (freq, mtf) tuples
        Frequency and MTF arrays for each ROI.
    nyquists : list of float
        Nyquist frequency for each ROI (cycles/mm).
    metrics_df : pandas.DataFrame
        Dataframe containing ROI metrics (mtf50, mtf10, auc).
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


def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    # Load series
    series_dirs = {"LR": Path(args.lr_dir), "SR": Path(args.sr_dir), "HR": Path(args.hr_dir)}
    series_slices: Dict[str, List] = {}
    for name, d in series_dirs.items():
        logging.info(f"Loading series {name} from {d}")
        series_slices[name] = load_series(d)
        if not series_slices[name]:
            logging.warning(f"No DICOM slices found in {d}")
    # Adjust SR pixel spacing if scale factor provided
    if args.sr_scale is not None and args.sr_scale > 0.0:
        logging.info(f"Applying SR scale factor {args.sr_scale} to pixel spacing")
        adjust_pixel_spacing_for_sr(series_slices.get("SR", []), args.sr_scale)
    # Build slice lookup table for pixel spacing per slice (col spacing used for Nyquist)
    slices_lookup_map: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for name, slices in series_slices.items():
        lookup = {}
        for s in slices:
            lookup[s.index] = s.pixel_spacing
        slices_lookup_map[name] = lookup
    # ROI selection
    selector = ROISelector(
        roi_width=30,
        roi_height=100,
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
    )
    logging.info(f"Saved MTF plot to {plot_path}")


if __name__ == "__main__":
    main()