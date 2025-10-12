"""
Plotting utilities for MTF curves.

Supports both physical cycles/mm plotting (common band)
and normalized x-axis (f/fNyquist).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def plot_mean_mtf(
    series_mtf: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    series_nyquist: Dict[str, List[float]],
    out_path: str,
    draw_nyquist: bool = True,
    normalize_x: bool = False,
) -> None:
    """Plot the mean MTF curve for each series and save to file."""

    # ---------------------------
    #  図設定（フォント・サイズなど）
    # ---------------------------
    # Calibriが存在しなくてもエラーを出さず自動フォールバック
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))  # 縦長にして見やすく

    # 白黒でも区別できるようにラインスタイル変更（全て黒）
    style_map = {
        "LR": {"color": "black", "linestyle": "-", "linewidth": 2.0, "label": "LR"},
        "SR": {"color": "black", "linestyle": "--", "linewidth": 2.0, "label": "SR"},
        "HR": {"color": "black", "linestyle": ":", "linewidth": 2.0, "label": "HR"},
    }

    # ---------------------------
    #  X軸スケール設定
    # ---------------------------
    if normalize_x:
        x_common = np.linspace(0.0, 1.0, 512)
    else:
        nyq_medians = [float(np.median(n)) for n in series_nyquist.values() if n]
        fmax = 1.05 * float(np.max(nyq_medians)) if nyq_medians else 0.5
        x_common = np.linspace(0.0, fmax, 1024)

    # ---------------------------
    #  曲線プロット
    # ---------------------------
    for name, curves in series_mtf.items():
        if not curves:
            continue

        nyqs = series_nyquist.get(name, [])
        interp_list = []
        for i, (freq, mtf) in enumerate(curves):
            freq = np.asarray(freq, dtype=float)
            mtf = np.asarray(mtf, dtype=float)

            # Nyquist取得
            nyq = float(nyqs[i]) if i < len(nyqs) else (float(np.median(nyqs)) if nyqs else 0.0)
            if nyq <= 0:
                continue

            if normalize_x:
                f_norm = freq / nyq
                mask = (f_norm >= 0.0) & (f_norm <= 1.0)
                if mask.sum() < 4:
                    continue
                m_i = np.interp(x_common, f_norm[mask], mtf[mask], left=np.nan, right=np.nan)
                m_i[x_common > 1.0] = np.nan  # 正規化ではx>1をNaN
            else:
                mask = (freq >= 0.0) & (freq <= nyq)  # Nyquistまでで切る
                if mask.sum() < 4:
                    continue
                m_i = np.interp(x_common, freq[mask], mtf[mask], left=np.nan, right=np.nan)
                m_i[x_common > nyq] = np.nan  # 超過域をNaN

            interp_list.append(m_i)

        if not interp_list:
            continue

        arr = np.vstack(interp_list)
        mean = np.nanmean(arr, axis=0)
        ax.plot(x_common, mean, **style_map[name])

    # ---------------------------
    #  軸・ガイド線
    # ---------------------------
    if normalize_x:
        ax.set_xlabel("Normalized Spatial Frequency (f / f_Nyquist)")
        ax.set_xlim(0.0, 1.0)
        if draw_nyquist:
            ax.axvline(1.0, color="0.6", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    else:
        ax.set_xlabel("Spatial Frequency [cycles/mm]")
        if draw_nyquist:
            for name, nyqs in series_nyquist.items():
                if nyqs:
                    nyq_med = float(np.median(nyqs))
                    ax.axvline(nyq_med, color="0.6", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
        ax.set_xlim(0.0, x_common[-1])

    ax.set_ylabel("MTF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)

    # ---------------------------
    #  凡例（右上・枠線付き）
    # ---------------------------
    legend = ax.legend(
        loc="upper right",
        fontsize=13,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="black"
    )
    frame = legend.get_frame()
    frame.set_linewidth(1.5)
    frame.set_edgecolor("black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
