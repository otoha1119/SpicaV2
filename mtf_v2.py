#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, wilcoxon

from MTF.dicom_io import DicomSlice, adjust_pixel_spacing_for_sr, apply_spacing_override, load_series
from MTF.mtf_core import compute_mtf_for_roi
from MTF.plotting import plot_mean_mtf
from MTF.roi_selector import ROI, ROISelector
from MTF.utils import ensure_dir, seed_everything, setup_logging


@dataclass(frozen=True)
class CropSpec:
    z_start_1based: int
    z_end_1based: int
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MTF v2 (heart-crop): Wasserstein + MTF50 + classic MTF graph")
    p.add_argument("--lr_dir", required=True, type=str)
    p.add_argument("--sr_dir", required=True, type=str)
    p.add_argument("--hr_dir", required=True, type=str)
    p.add_argument("--out_dir", default="/workspace/results", type=str)

    p.add_argument("--num_rois", default=4000, type=int, help="ROIs per case for MTF50/MTF curve")
    p.add_argument("--draw_nyquist", default=0, choices=[0, 1], type=int)
    p.add_argument("--use_cuda", default=1, choices=[0, 1], type=int)
    p.add_argument("--gpu_ids", default="0", type=str)
    p.add_argument("--sr_scale", default=None, type=float)
    p.add_argument("--x_norm", default=1, choices=[0, 1], type=int)
    p.add_argument("--lr_spacing", nargs=2, type=float, metavar=("ROW", "COL"))
    p.add_argument("--hr_spacing", nargs=2, type=float, metavar=("ROW", "COL"))
    p.add_argument("--seed", default=42, type=int)

    # Heart crop defaults (user-specified)
    p.add_argument("--hr_z", nargs=2, type=int, default=[1, 550], metavar=("START", "END"))
    p.add_argument("--hr_xyxy", nargs=4, type=int, default=[90, 190, 850, 830], metavar=("X1", "Y1", "X2", "Y2"))

    p.add_argument("--sr_z", nargs=2, type=int, default=[1, 210], metavar=("START", "END"))
    p.add_argument("--sr_xyxy", nargs=4, type=int, default=[60, 124, 800, 710], metavar=("X1", "Y1", "X2", "Y2"))

    p.add_argument("--lr_z", nargs=2, type=int, default=[1, 210], metavar=("START", "END"))
    p.add_argument("--lr_xyxy", nargs=4, type=int, default=[30, 62, 400, 355], metavar=("X1", "Y1", "X2", "Y2"))
    return p.parse_args()


def _discover_case_dirs(series_root: Path) -> Dict[str, Path]:
    subdirs = sorted([p for p in series_root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if subdirs:
        return {p.name: p for p in subdirs}
    return {series_root.name: series_root}


def _crop_slices(slices: Sequence[DicomSlice], spec: CropSpec) -> List[DicomSlice]:
    z0 = max(0, spec.z_start_1based - 1)
    z1 = min(len(slices) - 1, spec.z_end_1based - 1)
    if len(slices) == 0 or z1 < z0:
        return []

    cropped: List[DicomSlice] = []
    for i in range(z0, z1 + 1):
        sl = slices[i]
        h, w = sl.pixel_array.shape
        x1 = max(0, min(spec.x1, w))
        x2 = max(0, min(spec.x2, w))
        y1 = max(0, min(spec.y1, h))
        y2 = max(0, min(spec.y2, h))
        if x2 <= x1 or y2 <= y1:
            continue

        roi_img = sl.pixel_array[y1:y2, x1:x2]
        if roi_img.size == 0:
            continue

        cropped.append(
            DicomSlice(
                index=sl.index,
                filename=sl.filename,
                pixel_array=roi_img,
                pixel_spacing=sl.pixel_spacing,
                slice_thickness=sl.slice_thickness,
                convolution_kernel=sl.convolution_kernel,
            )
        )
    return cropped


def _apply_spacing(args: argparse.Namespace, series_slices: Dict[str, List[DicomSlice]]) -> None:
    if args.lr_spacing:
        apply_spacing_override(series_slices.get("LR", []), args.lr_spacing)
        apply_spacing_override(series_slices.get("SR", []), args.lr_spacing)
    if args.hr_spacing:
        apply_spacing_override(series_slices.get("HR", []), args.hr_spacing)
    if args.sr_scale is not None and args.sr_scale > 0:
        adjust_pixel_spacing_for_sr(series_slices.get("SR", []), args.sr_scale)


def _flatten_hu(cropped_slices: Sequence[DicomSlice]) -> np.ndarray:
    if not cropped_slices:
        return np.array([], dtype=np.float32)
    arr = np.concatenate([s.pixel_array.astype(np.float32).ravel() for s in cropped_slices])
    return arr[np.isfinite(arr)]


def _extract_rois_from_crop(
    cropped_slices: Sequence[DicomSlice],
    series_name: str,
    num_rois: int,
) -> List[ROI]:
    selector = ROISelector(
        roi_width=30,
        roi_height=100,
        angle_min=5.0,
        angle_max=15.0,
        delta_hu_threshold=200.0,
    )
    return selector.select_rois(list(cropped_slices), series_name=series_name, num_rois=num_rois)


def _mtf50_and_curves(rois: Sequence[ROI], lookup: Dict[int, Tuple[float, float]], use_cuda: bool):
    mtf50_list: List[float] = []
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    nyqs: List[float] = []
    for roi in rois:
        row_spacing, col_spacing = lookup.get(roi.slice_index, (1.0, 1.0))
        try:
            freq, mtf, mtf50, _ = compute_mtf_for_roi(
                roi.roi_image,
                pixel_spacing=(row_spacing, col_spacing),
                oversample_factor=4,
                zero_pad_factor=4,
                use_cuda=use_cuda,
            )
        except Exception:
            continue
        mtf50_list.append(float(mtf50))
        curves.append((freq, mtf))
        nyqs.append(0.5 / col_spacing if col_spacing > 0 else 0.0)

    mtf50_case = float(np.median(mtf50_list)) if mtf50_list else np.nan
    return mtf50_case, mtf50_list, curves, nyqs


def _wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[mask], yy[mask]
    if len(xx) < 2:
        return np.nan, np.nan, int(len(xx))
    try:
        res = wilcoxon(xx, yy, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue), int(len(xx))
    except ValueError:
        return np.nan, np.nan, int(len(xx))


def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)

    if args.use_cuda == 1 and args.gpu_ids != "-1":
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    out_dir = Path(args.out_dir) / "mtf_v2"
    ensure_dir(out_dir)

    crop_specs = {
        "LR": CropSpec(args.lr_z[0], args.lr_z[1], *args.lr_xyxy),
        "SR": CropSpec(args.sr_z[0], args.sr_z[1], *args.sr_xyxy),
        "HR": CropSpec(args.hr_z[0], args.hr_z[1], *args.hr_xyxy),
    }

    roots = {
        "LR": Path(args.lr_dir),
        "SR": Path(args.sr_dir),
        "HR": Path(args.hr_dir),
    }

    case_map = {k: _discover_case_dirs(v) for k, v in roots.items()}
    common_case_ids = sorted(set(case_map["LR"]).intersection(case_map["SR"]).intersection(case_map["HR"]))
    if not common_case_ids:
        # Fallback: treat each provided root as a single paired case.
        common_case_ids = ["single_case"]

    all_case_records: List[dict] = []
    global_curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {"LR": [], "SR": [], "HR": []}
    global_nyq: Dict[str, List[float]] = {"LR": [], "SR": [], "HR": []}

    for case_id in common_case_ids:
        logging.info("Processing case: %s", case_id)
        loaded: Dict[str, List[DicomSlice]] = {}
        for name in ["LR", "SR", "HR"]:
            case_dir = case_map[name].get(case_id, roots[name])
            loaded[name] = load_series(case_dir)

        _apply_spacing(args, loaded)

        cropped: Dict[str, List[DicomSlice]] = {
            name: _crop_slices(loaded[name], crop_specs[name]) for name in ["LR", "SR", "HR"]
        }

        hu_lr = _flatten_hu(cropped["LR"])
        hu_sr = _flatten_hu(cropped["SR"])
        hu_hr = _flatten_hu(cropped["HR"])

        wd_hr_sr = float(wasserstein_distance(hu_hr, hu_sr)) if (hu_hr.size and hu_sr.size) else np.nan
        wd_hr_lr = float(wasserstein_distance(hu_hr, hu_lr)) if (hu_hr.size and hu_lr.size) else np.nan

        mtf50_case = {}
        n_rois = {}
        for name in ["LR", "SR", "HR"]:
            rois = _extract_rois_from_crop(cropped[name], name, args.num_rois)
            lookup = {s.index: s.pixel_spacing for s in loaded[name]}
            mtf50_med, mtf50_list, curves, nyqs = _mtf50_and_curves(rois, lookup, use_cuda=bool(args.use_cuda))
            mtf50_case[name] = mtf50_med
            n_rois[name] = len(mtf50_list)
            global_curves[name].extend(curves)
            global_nyq[name].extend(nyqs)

        all_case_records.append(
            {
                "case_id": case_id,
                "wd_hr_sr": wd_hr_sr,
                "wd_hr_lr": wd_hr_lr,
                "mtf50_hr": mtf50_case["HR"],
                "mtf50_sr": mtf50_case["SR"],
                "mtf50_lr": mtf50_case["LR"],
                "n_rois_hr": n_rois["HR"],
                "n_rois_sr": n_rois["SR"],
                "n_rois_lr": n_rois["LR"],
            }
        )

    per_case_df = pd.DataFrame(all_case_records)
    per_case_csv = out_dir / "per_case_metrics.csv"
    per_case_df.to_csv(per_case_csv, index=False)

    stat_rows = []
    w_stat, w_p, w_n = _wilcoxon_paired(per_case_df["wd_hr_sr"], per_case_df["wd_hr_lr"])
    stat_rows.append(
        {
            "metric": "wasserstein",
            "comparison": "HR-SR vs HR-LR",
            "statistic": w_stat,
            "pvalue": w_p,
            "n_pairs": w_n,
        }
    )

    for a, b, label in [
        ("mtf50_hr", "mtf50_sr", "HR vs SR"),
        ("mtf50_hr", "mtf50_lr", "HR vs LR"),
        ("mtf50_sr", "mtf50_lr", "SR vs LR"),
    ]:
        s, p, n = _wilcoxon_paired(per_case_df[a], per_case_df[b])
        stat_rows.append({"metric": "mtf50", "comparison": label, "statistic": s, "pvalue": p, "n_pairs": n})

    stat_df = pd.DataFrame(stat_rows)
    stat_csv = out_dir / "stats_wilcoxon.csv"
    stat_df.to_csv(stat_csv, index=False)

    plot_path = out_dir / "mtf_overlaid_heart_crop.png"
    plot_mean_mtf(
        series_mtf=global_curves,
        series_nyquist=global_nyq,
        out_path=str(plot_path),
        draw_nyquist=bool(args.draw_nyquist),
        normalize_x=bool(args.x_norm),
    )

    summary_txt = out_dir / "summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("MTF v2 summary (heart-crop based)\n")
        f.write("================================\n")
        f.write(f"Cases: {len(per_case_df)}\n")
        if len(per_case_df) > 0:
            f.write(f"Median WD(HR,SR): {np.nanmedian(per_case_df['wd_hr_sr']):.6f}\n")
            f.write(f"Median WD(HR,LR): {np.nanmedian(per_case_df['wd_hr_lr']):.6f}\n")
            f.write(f"Median MTF50 HR: {np.nanmedian(per_case_df['mtf50_hr']):.6f}\n")
            f.write(f"Median MTF50 SR: {np.nanmedian(per_case_df['mtf50_sr']):.6f}\n")
            f.write(f"Median MTF50 LR: {np.nanmedian(per_case_df['mtf50_lr']):.6f}\n")
        f.write("\nWilcoxon results\n")
        for _, r in stat_df.iterrows():
            f.write(
                f"- {r['metric']} | {r['comparison']} | stat={r['statistic']} | p={r['pvalue']} | n={int(r['n_pairs'])}\n"
            )

    logging.info("Saved: %s", per_case_csv)
    logging.info("Saved: %s", stat_csv)
    logging.info("Saved: %s", plot_path)
    logging.info("Saved: %s", summary_txt)


if __name__ == "__main__":
    main()
