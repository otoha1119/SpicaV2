"""
Plotting utilities for MTF curves.

- plot_mean_mtf: series-wise mean MTF overlay
  * Normalized view (f/f_Nyquist)での表示を既定
  * スタイルは LR=solid, SR=dashed, HR=dotted を固定
- plot_roi_signals: ROI 1枚の ESF / LSF / MTF を保存（必要な人向けの付加機能）
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _mean_curve_normalized(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    nyquists: List[float],
    npoints: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    各ROIの MTF を f/f_Nyquist に正規化して共通軸に射影し、平均を返す。
    """
    if not curves:
        return np.array([]), np.array([])
    x_common = np.linspace(0.0, 1.0, int(npoints), endpoint=True)
    stack = []
    for i, (freq, mtf) in enumerate(curves):
        if i < len(nyquists) and nyquists[i] and np.isfinite(nyquists[i]) and nyquists[i] > 0:
            nyq = float(nyquists[i])
        else:
            # 保険：freqの最大をNyquist代用（ほぼ1.0近辺まであるはず）
            nyq = float(np.max(freq)) if len(freq) else 0.0
        if nyq <= 0:
            continue
        x_norm = np.clip(np.asarray(freq, float) / nyq, 0.0, 1.0)
        y = np.asarray(mtf, float)
        # xが重複しないように間引き
        if len(x_norm) > 1:
            uniq_mask = np.concatenate(([True], np.diff(x_norm) > 1e-12))
            x_norm = x_norm[uniq_mask]
            y = y[uniq_mask]
        stack.append(np.interp(x_common, x_norm, y))
    if not stack:
        return np.array([]), np.array([])
    return x_common, np.mean(np.stack(stack, axis=0), axis=0)


def plot_mean_mtf(
    series_mtf: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    series_nyquist: Dict[str, List[float]],
    out_path: str,
    draw_nyquist: bool = False,
    normalize_x: bool = True,
) -> None:
    """
    “当時の図”の見た目に固定：
      - 既定は f/f_Nyquist（normalize_x=True）
      - 罫線は薄め
      - 凡例は枠あり
      - 線スタイル：LR=実線, SR=破線, HR=点線
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # スタイル固定（存在すればそのまま、無ければフォールバック）
    style_map = {
        "LR": {"linestyle": "-",  "linewidth": 2.2},
        "SR": {"linestyle": "--", "linewidth": 2.2},
        "HR": {"linestyle": ":",  "linewidth": 2.2},
    }
    fallback_styles = [
        {"linestyle": "-",  "linewidth": 2.0},
        {"linestyle": "--", "linewidth": 2.0},
        {"linestyle": ":",  "linewidth": 2.0},
        {"linestyle": "-.", "linewidth": 2.0},
    ]

    lines = []
    labels = []

    if normalize_x:
        # f/f_Nyquist 平均
        for k_i, (name, curves) in enumerate(series_mtf.items()):
            if not curves:
                continue
            nyqs = series_nyquist.get(name, [])
            x, y = _mean_curve_normalized(curves, nyqs)
            if len(x) == 0:
                continue
            st = style_map.get(name, fallback_styles[k_i % len(fallback_styles)])
            line, = ax.plot(x, y, **st, color="black", label=name)
            lines.append(line); labels.append(name)

        ax.set_xlabel("Normalized Spatial Frequency (f / f_Nyquist)")
        ax.set_xlim(0.0, 1.0)
        if draw_nyquist:
            ax.axvline(1.0, linestyle="--", linewidth=1.2, alpha=0.75)
    else:
        # 物理軸（cycles/mm）側でも動くように保持（当時図では未使用）
        for k_i, (name, curves) in enumerate(series_mtf.items()):
            if not curves:
                continue
            # 共通帯域：各ROIのmax freqの最小に合わせる
            fmax = np.inf
            for f, _ in curves:
                if len(f):
                    fmax = min(fmax, float(np.max(f)))
            if not np.isfinite(fmax) or fmax <= 0:
                continue
            x_common = np.linspace(0.0, fmax, 256, endpoint=True)
            stack = []
            for (f, m) in curves:
                f = np.asarray(f, float); m = np.asarray(m, float)
                if len(f) > 1:
                    uniq = np.concatenate(([True], np.diff(f) > 1e-12))
                    f = f[uniq]; m = m[uniq]
                stack.append(np.interp(x_common, f, m))
            y = np.mean(np.stack(stack, axis=0), axis=0)
            st = style_map.get(name, fallback_styles[k_i % len(fallback_styles)])
            line, = ax.plot(x_common, y, **st, color="black", label=name)
            lines.append(line); labels.append(name)
        ax.set_xlabel("Spatial Frequency [cycles/mm]")

    ax.set_ylabel("MTF")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)

    # 凡例は “枠あり” で固定
    if lines:
        leg = ax.legend(lines, labels, loc="upper right", frameon=True, framealpha=0.9, edgecolor="black")
        leg.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---- （任意機能：ROI単体のESF/LSF/MTF出力。使っていなければ無視してOK） ----
def plot_roi_signals(
    esf: np.ndarray,
    lsf: np.ndarray,
    freq: np.ndarray,
    mtf: np.ndarray,
    out_prefix: str,
) -> None:
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.0))
    ax1.plot(np.arange(len(esf)), esf, linewidth=1.6, color="black")
    ax1.set_title("ESF")
    ax1.set_xlabel("Samples (oversampled)")
    ax1.set_ylabel("Intensity")
    ax1.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_esf.png", dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6.0, 4.0))
    ax2.plot(np.arange(len(lsf)), lsf, linewidth=1.6, color="black")
    ax2.set_title("LSF (Hann-windowed)")
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_lsf.png", dpi=160)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6.0, 4.0))
    ax3.plot(freq, mtf, linewidth=1.8, color="black", label="MTF")
    ax3.set_title("MTF")
    ax3.set_xlabel("Spatial Frequency [cycles/mm]")
    ax3.set_ylabel("MTF")
    ax3.set_ylim(0.0, 1.05)
    ax3.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)
    ax3.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="black")
    fig3.tight_layout()
    fig3.savefig(f"{out_prefix}_mtf.png", dpi=160)
    plt.close(fig3)
