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
    """Plot mean MTF curve for each series and save to file (grayscale-printable)."""

    # ---- Font: Calibriが無い環境でも警告を出さない（DejaVu Sansに固定）----
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # ---- 白黒対応スタイル（線種で判別）----
    styles = {
        "LR": {"color": "black", "linestyle": "--", "linewidth": 1.8, "zorder": 3},
        "SR": {"color": "black", "linestyle": ":",  "linewidth": 1.8, "zorder": 3},
        "HR": {"color": "black", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    }

    # ---- x 軸共通座標 ----
    if normalize_x:
        x_common = np.linspace(0.0, 1.0, 512)
    else:
        nyq_medians = [float(np.median(n)) for n in series_nyquist.values() if n]
        fmax = 0.95 * float(np.min(nyq_medians)) if nyq_medians else 0.5
        x_common = np.linspace(0.0, fmax, 512)

    # ---- 各シリーズ平均 ----
    for name, curves in series_mtf.items():
        if not curves:
            continue
        nyqs = series_nyquist.get(name, [])
        interp_list = []
        for i, (freq, mtf) in enumerate(curves):
            freq = np.asarray(freq, dtype=float)
            mtf  = np.asarray(mtf,  dtype=float)
            if normalize_x:
                nyq = float(nyqs[i]) if i < len(nyqs) else (float(np.median(nyqs)) if nyqs else 0.0)
                if nyq <= 0:
                    continue
                f_norm = freq / nyq
                mask = (f_norm >= 0.0) & (f_norm <= 1.0)
                if mask.sum() < 4:
                    continue
                m_i = np.interp(x_common, f_norm[mask], mtf[mask], left=np.nan, right=np.nan)
            else:
                mask = (freq >= 0.0) & (freq <= x_common[-1])
                if mask.sum() < 4:
                    continue
                m_i = np.interp(x_common, freq[mask], mtf[mask], left=np.nan, right=np.nan)
            interp_list.append(m_i)

        if not interp_list:
            continue

        mean = np.nanmean(np.vstack(interp_list), axis=0)
        s = styles.get(name, {"color": "black", "linestyle": "-", "linewidth": 2.0, "zorder": 3})
        ax.plot(x_common, mean, label=name, **s)

    # ---- ラベル & ガイド線 ----
    if normalize_x:
        ax.set_xlabel("Normalized Spatial Frequency (f / f_Nyquist)")
        if draw_nyquist:
            ax.axvline(1.0, color="0.6", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    else:
        ax.set_xlabel("Spatial Frequency [cycles/mm]")
        if draw_nyquist:
            for name, nyqs in series_nyquist.items():
                if nyqs:
                    nyq_med = float(np.median(nyqs))
                    ax.axvline(nyq_med, color="0.6", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)

    ax.set_ylabel("MTF")
    ax.set_ylim(0.0, 1.0)

    # ---- グリッド（実線・薄め・細め、点線は使わない）----
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)
    ax.minorticks_off()

    # ---- 凡例：右上“やや内側”、白背景で読みやすく ----
    ax.legend(
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),  # ← 右上から少し内側
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="black",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
