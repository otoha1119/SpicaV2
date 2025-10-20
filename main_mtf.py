#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# パッケージ名は環境に合わせてください（ここでは MTF.* 前提）
from MTF.dicom_io import load_series, adjust_pixel_spacing_for_sr, apply_spacing_override
from MTF.roi_selector import ROISelector, ROI
from MTF.mtf_core import compute_mtf_for_roi, compute_auc
from MTF.plotting import plot_mean_mtf
from MTF.utils import ensure_dir, seed_everything, setup_logging


# ============================== CLI ============================== #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute MTF curves from CT series.")

    # 入力
    p.add_argument("--lr_dir", type=str, required=True)
    p.add_argument("--sr_dir", type=str, required=True)
    p.add_argument("--hr_dir", type=str, required=True)

    # 出力
    p.add_argument("--out_dir", type=str, default="outputs")

    # ROI/選別
    p.add_argument("--num_rois", type=int, default=150)
    p.add_argument("--roi_width", type=int, default=40)    # 法線方向の幅（短辺）
    p.add_argument("--roi_height", type=int, default=100)  # エッジ方向の高さ（長辺）
    p.add_argument("--angle_min", type=float, default=6.0)
    p.add_argument("--angle_max", type=float, default=12.0)

    # 一本＆上下貫通など（roi_selector側に属性がある場合のみ適用）
    p.add_argument("--require_single_edge", type=int, choices=[0, 1], default=1)
    p.add_argument("--edge_vertical_tol_deg", type=float, default=2.5)
    p.add_argument("--min_vertical_span_ratio", type=float, default=0.92)
    p.add_argument("--center_tolerance_ratio", type=float, default=0.08)

    # SR倍率/ピクセルピッチ上書き
    p.add_argument("--sr_scale", type=float, default=None)
    p.add_argument("--lr_spacing", type=float, nargs=2, metavar=("ROW", "COL"))
    p.add_argument("--hr_spacing", type=float, nargs=2, metavar=("ROW", "COL"))

    # 例の画像出力（回転前の見え方）
    p.add_argument("--export_examples", type=int, choices=[0, 1], default=0)
    p.add_argument("--examples_per_series", type=int, default=1)

    # 図の出力
    p.add_argument("--export_overlaid", type=int, choices=[0, 1], default=1,
                   help="Save overlaid MTF (per-ROI). 1=on (default)")
    p.add_argument("--export_mean", type=int, choices=[0, 1], default=0,
                   help="Save mean MTF. 0=off (default), 1=on")
    p.add_argument("--x_norm", type=int, choices=[0, 1], default=1,
                   help="1: normalize x by Nyquist, 0: physical cycles/mm")
    p.add_argument("--draw_nyquist", type=int, choices=[0, 1], default=1)

    # 速度/再現性
    p.add_argument("--use_cuda", type=int, choices=[0, 1], default=1)
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)

    # ESF縮約（mtf_coreが対応している場合のみ自動で渡す）
    p.add_argument("--esf_frac", type=float, default=1.0)
    p.add_argument("--esf_reduce", type=str, default="mean")
    p.add_argument("--trim_alpha", type=float, default=0.1)

    return p.parse_args()


# ====================== 計算ヘルパ（後方互換対応） ====================== #
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
    mtf_core.compute_mtf_for_roi のシグネチャに自動追従し、
    未対応の引数は渡さない。戻り値が (freq, mtf) だけでも
    MTF50/MTF10 を線形補間で推定してメトリクスを埋める。
    """
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    nyquists: List[float] = []
    records = []

    # 受け付ける引数だけ抽出（失敗時は安全セットにフォールバック）
    try:
        import inspect
        sig = inspect.signature(compute_mtf_for_roi)
        supported = set(sig.parameters.keys())
    except Exception:
        supported = {"roi", "pixel_spacing", "oversample_factor", "zero_pad_factor", "use_cuda"}

    base_kwargs = {
        "oversample_factor": 4,
        "zero_pad_factor": 4,
        "use_cuda": use_cuda,
        "row_fraction": row_fraction,
        "reduce_mode": reduce_mode,
        "trim_alpha": trim_alpha,
    }
    safe_kwargs = {k: v for k, v in base_kwargs.items() if k in supported}

    def _interp_threshold(freq: np.ndarray, mtf: np.ndarray, thr: float) -> float:
        try:
            f = np.asarray(freq, float)
            m = np.asarray(mtf, float)
            for i in range(1, len(m)):
                if (m[i-1] >= thr and m[i] <= thr) or (m[i-1] <= thr and m[i] >= thr):
                    f0, f1 = f[i-1], f[i]
                    m0, m1 = m[i-1], m[i]
                    if abs(m1 - m0) < 1e-12:
                        return float(f0)
                    t = (thr - m0) / (m1 - m0)
                    return float(f0 + t * (f1 - f0))
            return float("nan")
        except Exception:
            return float("nan")

    def _unpack_result(res):
        if isinstance(res, (list, tuple)):
            if len(res) >= 4:
                freq, mtf, mtf50, mtf10 = res[0], res[1], res[2], res[3]
            elif len(res) == 3:
                freq, mtf, mtf50 = res[0], res[1], res[2]
                mtf10 = _interp_threshold(freq, mtf, 0.1)
            elif len(res) == 2:
                freq, mtf = res[0], res[1]
                mtf50 = _interp_threshold(freq, mtf, 0.5)
                mtf10 = _interp_threshold(freq, mtf, 0.1)
            else:
                raise ValueError("Unexpected return format from compute_mtf_for_roi")
        else:
            raise ValueError("compute_mtf_for_roi returned non-sequence")
        return freq, mtf, float(mtf50), float(mtf10)

    for idx, roi in enumerate(rois):
        row_spacing, col_spacing = slices_lookup.get(roi.slice_index, (1.0, 1.0))
        try:
            res = compute_mtf_for_roi(
                roi.roi_image,
                pixel_spacing=(row_spacing, col_spacing),
                **safe_kwargs,
            )
            freq, mtf, mtf50, mtf10 = _unpack_result(res)
        except Exception as e:
            logging.warning(f"Failed to compute MTF for ROI idx={idx} slice={roi.slice_index}: {e}")
            continue

        nyq = 0.5 / col_spacing if col_spacing != 0 else 0.0
        nyquists.append(nyq)
        auc = compute_auc(freq, mtf, min(global_f_max, nyq))

        curves.append((freq, mtf))
        records.append(
            {
                "series": roi.series_name,
                "slice_index": roi.slice_index,
                "roi_idx": idx,
                "orientation_deg": getattr(roi, "orientation_deg", np.nan),
                "delta_hu": getattr(roi, "delta_hu", np.nan),
                "mtf50": mtf50,
                "mtf10": mtf10,
                "auc": auc,
                "pixel_spacing_mm": col_spacing,
                "nyquist": nyq,
            }
        )

    df = pd.DataFrame.from_records(records)
    return curves, nyquists, df


# ====================== 例の可視化（回転前、向き修正） ====================== #
def _export_example_raw_patch(
    raw_img: np.ndarray,            # 回転前スライス
    roi: ROI,                       # center_row/center_col/edge_angle_deg を利用
    roi_size: Tuple[int, int],      # (W, H) = (法線幅, エッジ高さ=長辺)
    out_dir: Path,
    series_name: str,
    idx_in_series: int,
) -> None:
    """
    回転前スライス上に、
      (1) 回転後の軸整列長方形の逆写像（黄色：長辺=エッジ方向）
      (2) エッジ法線のガイド線（オレンジ）
    を重畳して保存する。
    """
    import cv2
    img = np.asarray(raw_img, dtype=float)
    h, w = img.shape
    W, H = int(roi_size[0]), int(roi_size[1])   # W: 法線側の短辺 / H: エッジ側の長辺
    c, r = float(roi.center_col), float(roi.center_row)
    ang = float(roi.edge_angle_deg)             # 想定：法線角（0°=水平, 90°=垂直）

    # 角度が接線角で渡ってきた場合に +90° して法線角へ補正（ロバスト表示用）
    try:
        win = 15
        x0 = max(0, int(c) - win); x1 = min(w - 1, int(c) + win)
        y0 = max(0, int(r) - win); y1 = min(h - 1, int(r) + win)
        patch_local = img[y0:y1+1, x0:x1+1].astype(np.float32)
        gx = cv2.Sobel(patch_local, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch_local, cv2.CV_64F, 0, 1, ksize=3)
        ang_local = (np.degrees(np.arctan2(gy.mean(), gx.mean())) % 180.0)  # 法線角
        def ang_diff(a, b):
            d = (a - b + 90.0) % 180.0 - 90.0
            return abs(d)
        if ang_diff(ang, ang_local) > ang_diff((ang + 90.0) % 180.0, ang_local):
            ang = (ang + 90.0) % 180.0
    except Exception:
        pass

    # 基底ベクトル：法線 n（オレンジ線と同向）／接線 t（エッジ方向）
    ang_rad = np.deg2rad(ang)
    n = np.array([np.cos(ang_rad), np.sin(ang_rad)], dtype=np.float32)    # normal
    t = np.array([-np.sin(ang_rad), np.cos(ang_rad)], dtype=np.float32)   # tangent (edge dir)

    # ★長方形の長辺（H）を接線 t に、短辺（W）を法線 n に
    dx = W / 2.0  # 法線方向の半幅（短辺）
    dy = H / 2.0  # 接線方向の半長（長辺）
    center = np.array([c, r], dtype=np.float32)
    corners = np.stack([
        center + (-dx)*n + (-dy)*t,
        center + (+dx)*n + (-dy)*t,
        center + (+dx)*n + (+dy)*t,
        center + (-dx)*n + (+dy)*t,
    ], axis=0).astype(np.float32)
    poly = corners  # (4,2)

    # bbox パッチ
    x_min = max(0, int(np.floor(poly[:, 0].min()) - 10))
    x_max = min(w - 1, int(np.ceil (poly[:, 0].max()) + 10))
    y_min = max(0, int(np.floor(poly[:, 1].min()) - 10))
    y_max = min(h - 1, int(np.ceil (poly[:, 1].max()) + 10))
    patch = img[y_min:y_max + 1, x_min:x_max + 1]

    # ポリゴンをパッチ座標へ
    poly_patch = poly.copy()
    poly_patch[:, 0] -= x_min
    poly_patch[:, 1] -= y_min
    poly_closed = np.vstack([poly_patch, poly_patch[:1]])

    # オレンジ線（法線）
    nx, ny = np.cos(ang_rad), np.sin(ang_rad)
    cx, cy = c - x_min, r - y_min
    L = max(W, H) * 1.2
    p1 = (cx - nx * L, cy - ny * L)
    p2 = (cx + nx * L, cy + ny * L)

    # 保存
    out_dir = out_dir / "examples_raw" / series_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    ax.imshow(patch, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], '-', color='yellow', lw=1.5)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='orange', lw=2.0)
    fig.tight_layout(pad=0)
    fig.savefig(out_dir / f"roi_raw_{idx_in_series:03d}.png", dpi=200)
    plt.close(fig)


# ====================== Overlaid 図の保存 ====================== #
def _save_overlaid_mtf(
    series_curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    series_nyquists: Dict[str, List[float]],
    out_path: Path,
    draw_nyquist: bool,
    normalize_x: bool,
) -> None:
    """
    各シリーズの各ROIの MTF をすべて重ね描きして保存する。
    normalize_x=True のときは各ROI毎に Nyquist=1 へ正規化。
    """
    if not any(len(v) for v in series_curves.values()):
        logging.warning("No MTF curves to plot for overlaid figure.")
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for series_name, curves in series_curves.items():
        if not curves:
            continue
        nyqs = series_nyquists.get(series_name, [])
        # 全曲線を薄く
        for i, (freq, mtf) in enumerate(curves):
            if normalize_x:
                if i < len(nyqs) and nyqs[i] and np.isfinite(nyqs[i]) and nyqs[i] > 0:
                    x = np.asarray(freq, float) / float(nyqs[i])
                else:
                    x = np.asarray(freq, float)
                ax.set_xlabel("Normalized Spatial Frequency (cycles/Nyquist)")
            else:
                x = np.asarray(freq, float)
                ax.set_xlabel("Spatial Frequency [cycles/mm]")
            y = np.asarray(mtf, float)
            ax.plot(x, y, linewidth=0.7, alpha=0.25, label=series_name)

        # 代表線を1本だけ濃く（凡例用）
        for i, (freq, mtf) in enumerate(curves):
            if len(freq) > 1:
                if normalize_x and i < len(nyqs) and nyqs[i] > 0:
                    x_rep = np.asarray(freq, float) / float(nyqs[i])
                else:
                    x_rep = np.asarray(freq, float)
                ax.plot(x_rep, np.asarray(mtf, float), linewidth=1.5, alpha=0.9, label=f"{series_name} (rep)")
                break

        if draw_nyquist and normalize_x:
            ax.axvline(1.0, color="0.6", linestyle="--", linewidth=1.0)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("MTF")
    ax.grid(True, alpha=0.3)

    # 重複ラベル整理
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="best", framealpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logging.info(f"Saved overlaid MTF figure to: {out_path}")


# ============================== main ============================== #
def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 読み込み
    series_dirs = {"LR": Path(args.lr_dir), "SR": Path(args.sr_dir), "HR": Path(args.hr_dir)}
    series_slices: Dict[str, List] = {}
    for series_name, d in series_dirs.items():
        logging.info(f"Loading series {series_name} from {d}")
        series_slices[series_name] = load_series(d)
        if not series_slices[series_name]:
            logging.warning(f"No DICOM slices found in {d}")

    # spacing 補正/上書き
    if args.lr_spacing:
        apply_spacing_override(series_slices.get("LR", []), args.lr_spacing)
        apply_spacing_override(series_slices.get("SR", []), args.lr_spacing)
    if args.hr_spacing:
        apply_spacing_override(series_slices.get("HR", []), args.hr_spacing)
    if args.sr_scale and args.sr_scale > 0:
        adjust_pixel_spacing_for_sr(series_slices.get("SR", []), args.sr_scale)

    # 各シリーズの slice_index → (row_spacing, col_spacing) の辞書
    slices_lookup_map: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for series_name, slices in series_slices.items():
        lookup = {}
        for s in slices:
            lookup[s.index] = s.pixel_spacing
        slices_lookup_map[series_name] = lookup

    # ROI 選択器（互換性重視：必須だけ __init__ に渡し、追加は hasattr で注入）
    selector = ROISelector(
        roi_width=args.roi_width,
        roi_height=args.roi_height,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
    )
    _optional = {
        "delta_hu_threshold": 200.0,
        "edge_vertical_tol_deg": getattr(args, "edge_vertical_tol_deg", 3.0),
        "min_vertical_span_ratio": getattr(args, "min_vertical_span_ratio", 0.92),
        "require_single_edge": bool(getattr(args, "require_single_edge", 1)),
        "center_tolerance_ratio": getattr(args, "center_tolerance_ratio", 0.08),
    }
    for key, val in _optional.items():
        if hasattr(selector, key):
            try:
                setattr(selector, key, val)
            except Exception:
                logging.warning(f"ROISelector has attr '{key}' but failed to set value={val!r}")

    # ROI 抽出
    series_rois: Dict[str, List[ROI]] = {}
    for name, slices in series_slices.items():
        logging.info(f"Selecting ROIs for series {name}")
        rois = selector.select_rois(slices, name, args.num_rois)
        series_rois[name] = rois
        logging.info(f"Selected {len(rois)} ROIs for {name}")

    # 例の出力（回転前の見え方＋法線線）
    if args.export_examples == 1:
        ex_n = int(args.examples_per_series)
        for series_name, rois in series_rois.items():
            if not rois:
                continue
            # 回転前スライスアクセス用
            slice_by_index = {s.index: s for s in series_slices.get(series_name, [])}
            for j, roi in enumerate(rois[:ex_n]):
                raw_slice = slice_by_index[roi.slice_index].pixel_array
                _export_example_raw_patch(
                    raw_img=np.asarray(raw_slice, dtype=float),
                    roi=roi,
                    roi_size=(args.roi_width, args.roi_height),
                    out_dir=out_dir,
                    series_name=series_name,
                    idx_in_series=j,
                )
        logging.info(f"Saved examples under {out_dir / 'examples_raw'}")

    # AUC用の上限周波数（全シリーズのNyquistの最小×0.95）
    all_nyquists: List[float] = []
    for slices in series_slices.values():
        for s in slices:
            col_spacing = s.pixel_spacing[1]
            if col_spacing > 0:
                all_nyquists.append(0.5 / col_spacing)
    global_f_max = (min(all_nyquists) * 0.95) if all_nyquists else 0.0
    logging.info(f"Global f_max for AUC: {global_f_max:.3f} cycles/mm")

    # ROIごとの MTF 計算
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
        df_metrics.to_csv(out_dir / f"{name.lower()}_roi_metrics.csv", index=False)

    # ------------------ 図の保存 ------------------
    plot_path = out_dir / "mtf_overlaid.png"
    plot_mean_mtf(
        series_curves,          # 各シリーズの ROI→(freq, mtf) 一覧
        series_nyquists,        # 各ROIのNyquist（正規化用）
        str(plot_path),
        draw_nyquist=bool(args.draw_nyquist),
        normalize_x=bool(args.x_norm),   # ← run_mtf.sh の --x_norm に追従（既定は正規化ON）
    )
    logging.info(f"Saved mean MTF figure (normalized={bool(args.x_norm)}) to: {plot_path}")


if __name__ == "__main__":
    main()
