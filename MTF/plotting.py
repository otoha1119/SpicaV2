"""
Plotting utilities for MTF curves.

Supports both physical cycles/mm plotting (common band) and normalized x-axis (f/fNyquist).
API is kept compatible with the existing pipeline:
    plot_mean_mtf(series_mtf, series_nyquist, out_path, draw_nyquist, normalize_x=False)
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
    """Plot the mean MTF curve for each series and save to file.

    Parameters
    ----------
    series_mtf : dict
        {'LR': [(freq, mtf), ...], 'SR': [...], 'HR': [...]}
        where freq, mtf are 1D numpy arrays per ROI.
    series_nyquist : dict
        {'LR': [nyq1, nyq2, ...], 'SR': [...], 'HR': [...]}
    out_path : str
        Output PNG path.
    draw_nyquist : bool
        If True, draw guideline(s).
        - normalize_x=True: a single vertical line at x=1.0.
        - normalize_x=False: per-series median Nyquist lines.
    normalize_x : bool
        If True, plot on normalized x-axis (f/fNyquist) in [0,1].
        If False, use physical cycles/mm up to min Nyquist across series.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    colors = {'LR': 'tab:blue', 'SR': 'tab:orange', 'HR': 'tab:green'}

    # Build common x-axis
    if normalize_x:
        x_common = np.linspace(0.0, 1.0, 512)
    else:
        nyq_medians = []
        for nyqs in series_nyquist.values():
            if nyqs:
                nyq_medians.append(float(np.median(nyqs)))
        fmax = 0.95 * float(np.min(nyq_medians)) if nyq_medians else 0.5
        x_common = np.linspace(0.0, fmax, 512)

    # Interpolate ROI curves onto common x, then average
    for name, curves in series_mtf.items():
        if not curves:
            continue

        nyqs = series_nyquist.get(name, [])
        interp_list = []
        for i, (freq, mtf) in enumerate(curves):
            freq = np.asarray(freq, dtype=float)
            mtf = np.asarray(mtf, dtype=float)

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

        arr = np.vstack(interp_list)
        mean = np.nanmean(arr, axis=0)
        ax.plot(x_common, mean, label=name, color=colors.get(name, None), linewidth=2.0)

    # Labels and guides
    if normalize_x:
        ax.set_xlabel("Normalized Spatial Frequency (f / f_Nyquist)")
        if draw_nyquist:
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
    else:
        ax.set_xlabel("Spatial Frequency (cycles/mm)")
        if draw_nyquist:
            for name, nyqs in series_nyquist.items():
                if nyqs:
                    nyq_med = float(np.median(nyqs))
                    ax.axvline(nyq_med, color=colors.get(name, None), linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_ylabel("MTF")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, x_common[-1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
