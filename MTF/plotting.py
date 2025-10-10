"""
Plotting utilities for MTF curves.

This module provides a function to generate a single plot containing
the mean MTF curve for each series (LR, SR, HR).  Each curve is
computed by interpolating individual ROI MTFs onto a common frequency
axis up to the minimum Nyquist frequency in the series.  Optionally,
vertical lines marking the Nyquist frequencies of each series can be
drawn.
"""
###

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Use a non‑interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_mean_mtf(
    series_mtf: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    series_nyquist: Dict[str, List[float]],
    out_path: str,
    draw_nyquist: bool = True,
) -> None:
    """
    Plot the mean MTF curve for each series and save to file.

    Parameters
    ----------
    series_mtf : dict
        Mapping from series name ('LR', 'SR', 'HR') to a list of tuples
        (freq_array, mtf_array) for each ROI.
    series_nyquist : dict
        Mapping from series name to a list of Nyquist frequencies (one per ROI).
    out_path : str
        Path where the plot image will be saved.
    draw_nyquist : bool, optional
        Whether to draw vertical lines at the median Nyquist frequency of
        each series.
    """
    plt.figure(figsize=(8, 5))
    colors = {
        "LR": "tab:blue",
        "SR": "tab:orange",
        "HR": "tab:green",
    }
    for series_name, curves in series_mtf.items():
        if not curves:
            continue
        # Determine common frequency axis: up to min Nyquist across this series
        nyquists = series_nyquist.get(series_name, [])
        if not nyquists:
            continue
        f_max = min(nyquists)
        if f_max <= 0:
            continue
        # Create frequency grid
        num_points = 2000
        freq_grid = np.linspace(0, f_max, num_points)
        # Interpolate each curve onto the grid and accumulate
        interp_mtf = np.zeros_like(freq_grid)
        for freq, mtf in curves:
            # Restrict to f_max
            if freq[-1] < f_max:
                # Extend MTF by padding last value
                freq_ext = np.append(freq, f_max)
                mtf_ext = np.append(mtf, mtf[-1])
            else:
                freq_ext = freq
                mtf_ext = mtf
            interp = np.interp(freq_grid, freq_ext, mtf_ext)
            interp_mtf += interp
        interp_mtf /= len(curves)
        plt.plot(freq_grid, interp_mtf, label=series_name, color=colors.get(series_name, None))
        # Draw Nyquist line
        if draw_nyquist:
            med_nyquist = float(np.median(nyquists))
            plt.axvline(med_nyquist, linestyle="--", color=colors.get(series_name, None), alpha=0.5)
    plt.title("Mean MTF Curves")
    plt.xlabel("Spatial frequency (cycles/mm)")
    plt.ylabel("MTF")
    plt.ylim(0, 1.0)
    plt.xlim(left=0)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()